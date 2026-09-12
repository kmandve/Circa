"""Google OAuth 2.0 with offline access.

Two things here are easy to get wrong and expensive to debug:

1. **Consent screen publishing status.** A client left in "Testing" issues
   refresh tokens that expire after 7 days. The app must be published to
   "In production" (it can stay unverified, capped at 100 users). See
   docs/SETUP.md — `circa doctor` warns if a token looks like it is about to
   hit this.

2. **Scope choice.** We deliberately do *not* request the blanket
   `.../auth/calendar` scope. `calendar.app.created` lets Circa create and
   manage only the calendars it made itself, so a bug here can never modify or
   delete a real appointment. Reading the existing schedule (to tell a forced
   wake from a spontaneous one) uses a separate read-only scope.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import structlog
from cryptography.fernet import Fernet, InvalidToken

from circa.config import get_settings
from circa.db.models import OAuthToken
from circa.db.session import session_scope

log = structlog.get_logger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Google Health API - read-only, least privilege.
HEALTH_SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    # Required by users.pairedDevices, which is how the poller reads
    # `lastSyncTime`. Without it the watermark cannot be clamped to the last
    # instant data could exist, and a period when the watch had not synced would
    # be skipped permanently rather than backfilled.
    #
    # This is `settings`, not `profile` - confirmed from the API's own discovery
    # document, which declares required scopes per method:
    #   https://health.googleapis.com/$discovery/rest?version=v4
    # That document is the authoritative source; the prose docs are not.
    "https://www.googleapis.com/auth/googlehealth.settings.readonly",
]

# Calendar: create/manage ONLY calendars this app created, plus read-only
# visibility of the existing schedule for forced-wake detection.
CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.app.created",
    "https://www.googleapis.com/auth/calendar.readonly",
    # Lets Circa set the colour of the calendars it created. `app.created` alone
    # cannot: `calendarList.patch` answers 401 for it, whatever the discovery
    # document says - verified against the live API with a raw request. This is
    # the narrowest scope that works; it manages calendar *list* entries (colour,
    # visibility, notifications) and grants no access to event content anywhere.
    # Without it Circa falls back to Google's eleven preset event colours.
    "https://www.googleapis.com/auth/calendar.calendarlist",
]

# If `calendar.app.created` is rejected for this client, fall back to the
# broader scope. `circa auth --broad-calendar` opts into this explicitly.
BROAD_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.readonly",
]

PURPOSES = ("health", "calendar")

SCOPE_SETS = {"health": HEALTH_SCOPES, "calendar": CALENDAR_SCOPES}

# Refresh this long before nominal expiry, so a slow request never races it.
REFRESH_MARGIN = timedelta(minutes=5)


class AuthError(RuntimeError):
    pass


@dataclass(slots=True)
class TokenBundle:
    access_token: str
    refresh_token: str | None
    expires_at: datetime
    scopes: str


# ---------------------------------------------------------------------------
# encrypted storage
# ---------------------------------------------------------------------------


def _fernet() -> Fernet:
    return Fernet(get_settings().fernet_key())


def save_tokens(bundle: TokenBundle, purpose: str) -> None:
    now = datetime.now(UTC)
    with session_scope() as s:
        row = s.get(OAuthToken, purpose)
        if row is None:
            row = OAuthToken(purpose=purpose, issued_at=now)
            s.add(row)
        row.access_token = bundle.access_token
        row.expires_at = bundle.expires_at
        row.scopes = bundle.scopes
        row.updated_at = now
        if bundle.refresh_token:
            # A re-auth issues a new refresh token; a plain refresh does not.
            row.refresh_token_encrypted = _fernet().encrypt(bundle.refresh_token.encode())
            row.issued_at = now


def load_refresh_token(purpose: str) -> str | None:
    with session_scope() as s:
        row = s.get(OAuthToken, purpose)
        if row is None or not row.refresh_token_encrypted:
            return None
        try:
            return _fernet().decrypt(row.refresh_token_encrypted).decode()
        except InvalidToken as exc:
            raise AuthError(
                f"Stored {purpose} refresh token could not be decrypted. The "
                "encryption key (CIRCA_SECRET_KEY or <data_dir>/secret.key) has "
                "changed. Re-run `circa auth`."
            ) from exc


def token_status(purpose: str | None = None) -> dict:
    """Summary for `circa doctor` and the web app's status panel.

    With no `purpose`, reports the combined picture across both tokens -
    `authenticated` is true only when *both* are present, since the system
    needs Health to collect and Calendar to publish.
    """
    if purpose is None:
        parts = {p: token_status(p) for p in PURPOSES}
        ages = [
            v["refresh_token_age_days"]
            for v in parts.values()
            if v.get("refresh_token_age_days") is not None
        ]
        return {
            "authenticated": all(v["authenticated"] for v in parts.values()),
            "per_purpose": parts,
            "missing": [p for p, v in parts.items() if not v["authenticated"]],
            "scopes": sorted({s for v in parts.values() for s in v.get("scopes", [])}),
            "refresh_token_age_days": max(ages) if ages else None,
            "testing_status_suspected": any(
                v.get("testing_status_suspected") for v in parts.values()
            ),
        }

    with session_scope() as s:
        row = s.get(OAuthToken, purpose)
        if row is None:
            return {"authenticated": False, "purpose": purpose, "scopes": []}
        age_days = None
        if row.issued_at:
            age_days = (datetime.now(UTC) - row.issued_at).total_seconds() / 86400
        return {
            "purpose": purpose,
            "authenticated": bool(row.refresh_token_encrypted),
            "scopes": row.scopes.split() if row.scopes else [],
            "access_token_expires_at": row.expires_at,
            "refresh_token_issued_at": row.issued_at,
            "refresh_token_age_days": age_days,
            # The 7-day cliff means a "Testing" consent screen dies right here.
            "testing_status_suspected": bool(age_days and age_days > 6.5),
        }


# ---------------------------------------------------------------------------
# token exchange / refresh
# ---------------------------------------------------------------------------


def _parse_token_response(data: dict) -> TokenBundle:
    if "access_token" not in data:
        raise AuthError(f"Token endpoint returned no access_token: {data}")
    expires_in = int(data.get("expires_in", 3600))
    return TokenBundle(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        scopes=data.get("scope", ""),
    )


def exchange_code(code: str, code_verifier: str) -> TokenBundle:
    settings = get_settings()
    resp = httpx.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise AuthError(f"Code exchange failed ({resp.status_code}): {resp.text}")
    bundle = _parse_token_response(resp.json())
    if not bundle.refresh_token:
        raise AuthError(
            "Google did not return a refresh token. This happens when the app was "
            "already authorised. Revoke Circa at "
            "https://myaccount.google.com/permissions and run `circa auth` again."
        )
    return bundle


def refresh_access_token(refresh_token: str) -> TokenBundle:
    settings = get_settings()
    resp = httpx.post(
        TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        detail = resp.text
        if "invalid_grant" in detail:
            raise AuthError(
                "Refresh token rejected (invalid_grant). The usual cause is an OAuth "
                "consent screen still in 'Testing' status, which expires refresh "
                "tokens after 7 days. Publish the app to 'In production' in Google "
                "Cloud Console, then re-run `circa auth`."
            )
        raise AuthError(f"Token refresh failed ({resp.status_code}): {detail}")
    return _parse_token_response(resp.json())


def get_access_token(purpose: str = "health", force_refresh: bool = False) -> str:
    """Return a currently-valid access token for one API family.

    `purpose` is not cosmetic: handing a Health request a token that carries
    Calendar scopes produces a 403 DISALLOWED_OAUTH_SCOPES, so the two must
    never be interchanged.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"unknown token purpose {purpose!r}")

    with session_scope() as s:
        row = s.get(OAuthToken, purpose)
        if row is None or not row.refresh_token_encrypted:
            raise AuthError(
                f"Not authenticated for {purpose}. Run `circa auth` first."
            )
        cached, expires_at = row.access_token, row.expires_at

    if not force_refresh and cached and expires_at and datetime.now(UTC) < expires_at - REFRESH_MARGIN:
        return cached

    refresh_token = load_refresh_token(purpose)
    if refresh_token is None:
        raise AuthError(f"Not authenticated for {purpose}. Run `circa auth` first.")
    bundle = refresh_access_token(refresh_token)
    save_tokens(bundle, purpose)
    log.debug("access_token.refreshed", purpose=purpose,
              expires_at=bundle.expires_at.isoformat())
    return bundle.access_token


# ---------------------------------------------------------------------------
# interactive flow
# ---------------------------------------------------------------------------


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def build_auth_url(state: str, code_challenge: str, scopes: list[str]) -> str:
    settings = get_settings()
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        # Both are required to reliably receive a refresh token.
        "access_type": "offline",
        "prompt": "consent",
        # NOT include_granted_scopes. Incremental authorisation deliberately
        # merges previously-granted scopes onto the new token, which would put
        # Calendar scopes on the Health token and trigger
        # 403 DISALLOWED_OAUTH_SCOPES on every Health request.
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


_RESULT: dict[str, str] = {}
_DONE = threading.Event()

_PAGE = """<!doctype html><meta charset="utf-8">
<title>Circa</title>
<style>
 body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      display:grid;place-items:center;height:100vh;margin:0;background:#0f1115;color:#e6e8ee}}
 .card{{max-width:34rem;padding:2.5rem;text-align:center}}
 h1{{font-size:1.4rem;margin:0 0 .6rem;letter-spacing:-.01em}}
 p{{margin:0;color:#9aa3b2}}
 .ok{{color:#7ee2a8}} .bad{{color:#ff8a8a}}
</style>
<div class="card"><h1 class="{cls}">{title}</h1><p>{body}</p></div>
"""


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/oauth/callback":
            self.send_response(404)
            self.end_headers()
            return
        qs = urllib.parse.parse_qs(parsed.query)
        _RESULT.update({k: v[0] for k, v in qs.items()})

        if "error" in _RESULT:
            html = _PAGE.format(
                cls="bad",
                title="Authorisation failed",
                body=f"Google returned: {_RESULT['error']}. You can close this tab.",
            )
        else:
            html = _PAGE.format(
                cls="ok",
                title="Circa is connected",
                body="You can close this tab and return to the terminal.",
            )
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        _DONE.set()

    def log_message(self, *args) -> None:  # silence stdlib request logging
        pass


def run_local_auth_flow(
    purpose: str,
    open_browser: bool = True,
    broad_calendar: bool = False,
) -> TokenBundle:
    """Run one interactive consent flow and store the resulting token.

    Called once per purpose - Health and Calendar cannot share a token.

    On a headless VM, forward the port from your laptop first::

        ssh -L 8721:localhost:8721 user@vm
        circa auth --no-browser

    then open the printed URL in your own browser.
    """
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise AuthError(
            "CIRCA_GOOGLE_CLIENT_ID / CIRCA_GOOGLE_CLIENT_SECRET are not set. "
            "See docs/SETUP.md for how to create the OAuth client."
        )
    if purpose not in PURPOSES:
        raise ValueError(f"unknown token purpose {purpose!r}")

    scopes = SCOPE_SETS[purpose]
    if purpose == "calendar" and broad_calendar:
        scopes = BROAD_CALENDAR_SCOPES

    _RESULT.clear()
    _DONE.clear()
    state = secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()
    url = build_auth_url(state, challenge, scopes)

    server = HTTPServer((settings.oauth_host, settings.oauth_port), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    label = {"health": "Fitbit / Google Health data", "calendar": "Google Calendar"}[purpose]
    print(f"\nStep: authorise {label}\n")
    print("Open this URL:\n")
    print(f"  {url}\n")
    print("Google will warn that the app is unverified — that is expected for a")
    print("personal project. Choose 'Advanced' then 'Go to Circa (unsafe)'.\n")
    if open_browser:
        webbrowser.open(url)

    try:
        if not _DONE.wait(timeout=300):
            raise AuthError("Timed out after 5 minutes waiting for the OAuth callback.")
    finally:
        server.shutdown()
        server.server_close()

    if "error" in _RESULT:
        raise AuthError(f"Authorisation denied: {_RESULT['error']}")
    if _RESULT.get("state") != state:
        raise AuthError("OAuth state mismatch — aborting (possible CSRF).")

    bundle = exchange_code(_RESULT["code"], verifier)
    save_tokens(bundle, purpose)
    return bundle

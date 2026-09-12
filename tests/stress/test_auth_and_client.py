"""Token lifecycle and the API client's failure handling.

Refresh tokens are the single point of failure for the whole collector: if one
silently stops working the app keeps running and quietly collects nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from circa.ingest.oauth import (
    PURPOSES,
    AuthError,
    TokenBundle,
    get_access_token,
    load_refresh_token,
    save_tokens,
    token_status,
)


def _bundle(access="at-1", refresh="rt-1", ttl_seconds=3600):
    return TokenBundle(
        access_token=access,
        refresh_token=refresh,
        expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        scopes="https://www.googleapis.com/auth/googlehealth.sleep.readonly",
    )


# --- storage ----------------------------------------------------------------


@pytest.mark.parametrize("purpose", PURPOSES)
def test_refresh_token_is_encrypted_at_rest(db, purpose):
    from sqlalchemy import select

    from circa.db.models import OAuthToken

    save_tokens(_bundle(refresh="super-secret-refresh"), purpose)
    with db() as s:
        row = s.scalar(select(OAuthToken).where(OAuthToken.purpose == purpose))
        blob = bytes(row.refresh_token_encrypted)
    assert b"super-secret-refresh" not in blob
    assert load_refresh_token(purpose) == "super-secret-refresh"


def test_a_plain_refresh_does_not_wipe_the_stored_refresh_token():
    """Google returns a refresh token on consent but not on a plain refresh.

    Overwriting with None would silently unauthenticate the app until the next
    manual re-auth - and the failure would only surface hours later.
    """
    save_tokens(_bundle(refresh="original"), "health")
    save_tokens(TokenBundle("new-access", None, datetime.now(UTC) + timedelta(hours=1), ""), "health")
    assert load_refresh_token("health") == "original"


def test_the_two_purposes_never_share_a_token():
    """Health and Calendar scopes cannot live on one token: the Health API
    rejects the whole request with DISALLOWED_OAUTH_SCOPES."""
    save_tokens(_bundle(access="health-at", refresh="health-rt"), "health")
    save_tokens(_bundle(access="cal-at", refresh="cal-rt"), "calendar")
    assert load_refresh_token("health") == "health-rt"
    assert load_refresh_token("calendar") == "cal-rt"


def test_an_undecryptable_token_raises_something_actionable(monkeypatch):
    from sqlalchemy import select

    from circa.db.models import OAuthToken
    from circa.db.session import session_scope

    save_tokens(_bundle(), "health")
    with session_scope() as s:
        row = s.scalar(select(OAuthToken).where(OAuthToken.purpose == "health"))
        row.refresh_token_encrypted = b"not-a-valid-fernet-token"

    with pytest.raises(AuthError) as exc:
        load_refresh_token("health")
    assert "decrypt" in str(exc.value).lower()


# --- refresh ----------------------------------------------------------------


def test_a_valid_cached_access_token_is_reused(monkeypatch):
    calls = []

    def fake_refresh(_token):
        calls.append(1)
        return _bundle(access="refreshed")

    monkeypatch.setattr("circa.ingest.oauth.refresh_access_token", fake_refresh)
    save_tokens(_bundle(access="cached", ttl_seconds=3600), "health")
    assert get_access_token("health") == "cached"
    assert not calls, "refreshed a token that was still valid"


def test_an_expiring_access_token_is_refreshed_before_it_dies(monkeypatch):
    """Refreshing only after expiry means every poll eats one guaranteed 401."""
    monkeypatch.setattr(
        "circa.ingest.oauth.refresh_access_token", lambda _t: _bundle(access="refreshed")
    )
    save_tokens(_bundle(access="nearly-dead", ttl_seconds=30), "health")
    assert get_access_token("health") == "refreshed"


def test_missing_credentials_name_the_command_that_fixes_them():
    with pytest.raises(AuthError) as exc:
        get_access_token("health")
    assert "auth" in str(exc.value).lower()


def test_token_status_reports_both_purposes_before_any_auth():
    status = token_status()
    assert status["authenticated"] is False
    assert sorted(status["missing"]) == sorted(PURPOSES)
    for purpose in PURPOSES:
        assert status["per_purpose"][purpose]["authenticated"] is False


def test_token_status_tracks_refresh_token_age():
    """A token issued under a 'Testing' consent screen dies at seven days, so
    its age is the thing that predicts the outage."""
    save_tokens(_bundle(), "health")
    status = token_status("health")
    assert status["refresh_token_age_days"] is not None
    assert status["refresh_token_age_days"] < 1


# --- API client -------------------------------------------------------------


def _client(handler):
    from circa.ingest.health_api import HealthClient

    client = HealthClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler),
                                  base_url="https://health.googleapis.com/v4")
    return client


def test_a_403_surfaces_googles_own_message(monkeypatch):
    """An earlier handler replaced Google's text with a guess, which hid
    DISALLOWED_OAUTH_SCOPES and cost hours of misdiagnosis."""
    from circa.ingest.health_api import HealthApiError

    monkeypatch.setattr("circa.ingest.health_api.get_access_token", lambda *a, **k: "tok")

    def handler(request):
        return httpx.Response(403, json={"error": {
            "status": "PERMISSION_DENIED",
            "message": "Request had insufficient authentication scopes.",
            "details": [{"reason": "DISALLOWED_OAUTH_SCOPES"}],
        }})

    from circa.ingest.datatypes import BY_NAME

    client = _client(handler)
    with pytest.raises(HealthApiError) as exc:
        client.fetch(BY_NAME["sleep"], datetime.now(UTC) - timedelta(days=1), datetime.now(UTC))
    text = str(exc.value)
    assert "DISALLOWED_OAUTH_SCOPES" in text or "insufficient authentication scopes" in text


def test_pagination_follows_every_page(monkeypatch):
    monkeypatch.setattr("circa.ingest.health_api.get_access_token", lambda *a, **k: "tok")
    pages = {None: "p2", "p2": "p3", "p3": None}
    seen = []

    def handler(request):
        token = dict(request.url.params).get("pageToken")
        seen.append(token)
        body = {"dataPoints": [{"sleep": {"name": f"s/{token}"}}]}
        nxt = pages.get(token)
        if nxt:
            body["nextPageToken"] = nxt
        return httpx.Response(200, json=body)

    from circa.ingest.datatypes import BY_NAME

    result = _client(handler).fetch(
        BY_NAME["sleep"], datetime.now(UTC) - timedelta(days=1), datetime.now(UTC)
    )
    assert len(result.points) == 3
    assert seen == [None, "p2", "p3"]


def test_pagination_stops_at_max_pages(monkeypatch):
    """A server that always returns a next-page token must not loop forever."""
    monkeypatch.setattr("circa.ingest.health_api.get_access_token", lambda *a, **k: "tok")

    def handler(request):
        return httpx.Response(200, json={
            "dataPoints": [{"sleep": {"name": "s"}}],
            "nextPageToken": "always-more",
        })

    from circa.ingest.datatypes import BY_NAME

    result = _client(handler).fetch(
        BY_NAME["sleep"], datetime.now(UTC) - timedelta(days=1),
        datetime.now(UTC), max_pages=5,
    )
    assert result.pages <= 5

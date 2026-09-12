"""Client for the Google Health API v4.

Design notes:

* **Raw-first.** Every response is persisted verbatim before anything tries to
  interpret it. Google's own docs describe v4 as "actively evolving" and
  contradict themselves in places (skin temperature is listed both as a
  standalone type and as a daily derivation), so the normalisers must be
  allowed to be wrong without costing us data.

* **Filters may not exist as documented.** Google publishes filter examples
  only for interval types. If a filter expression is rejected we fall back to
  an unfiltered fetch and trim client-side, and record that fact so `circa
  probe` can report which types actually support server-side filtering.

* **Throttling.** The per-user ceiling is 300 req/min, but unverified apps are
  held to ~2.5 QPS, so we pace at 2 QPS and back off on 429.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from circa.ingest.datatypes import DataType
from circa.ingest.oauth import AuthError, get_access_token

log = structlog.get_logger(__name__)

BASE_URL = "https://health.googleapis.com/v4"
DISCOVERY_URL = "https://health.googleapis.com/$discovery/rest?version=v4"

MIN_REQUEST_INTERVAL = 0.5  # seconds -> 2 QPS, under the 2.5 QPS unverified cap
MAX_RETRIES = 5
DEFAULT_PAGE_SIZE = 1000


class HealthApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class FetchResult:
    data_type: str
    points: list[dict] = field(default_factory=list)
    pages: int = 0
    server_filtered: bool = True
    filter_error: str | None = None


class HealthClient:
    """Synchronous client. One user, low volume — async buys nothing here."""

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=60.0)
        self._last_request = 0.0
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HealthClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- plumbing ---------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        force_refresh = False

        for attempt in range(MAX_RETRIES):
            self._throttle()
            token = get_access_token("health", force_refresh=force_refresh)
            force_refresh = False
            resp = self._client.get(
                url, params=params, headers={"Authorization": f"Bearer {token}"}
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 401:
                # Access token rejected — force one refresh, then give up.
                if attempt == 0:
                    log.info("health_api.token_rejected_retrying")
                    force_refresh = True
                    continue
                raise AuthError(f"Unauthorised after refresh: {resp.text}")

            if resp.status_code == 403:
                # Surface Google's own message. An earlier version replaced it
                # with a guess, which hid the actual cause completely.
                detail = resp.text
                hint = ""
                if "DISALLOWED_OAUTH_SCOPES" in detail:
                    hint = (
                        "\n\nThe access token carries scopes the Health API refuses "
                        "- almost always Calendar scopes. Health and Calendar must "
                        "use separate tokens. Re-run `circa auth`."
                    )
                elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in detail:
                    hint = "\n\nA required googlehealth scope was not granted. Re-run `circa auth`."
                elif "SERVICE_DISABLED" in detail or "has not been used" in detail:
                    hint = (
                        "\n\nThe Google Health API is not enabled on this Cloud "
                        "project. Enable it at "
                        "https://console.cloud.google.com/apis/library/health.googleapis.com"
                    )
                raise HealthApiError(
                    f"Forbidden (403) from {path}: {detail}{hint}",
                    status=403, body=detail,
                )

            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = min(2**attempt, 30)
                log.warning("health_api.retrying", status=resp.status_code,
                            attempt=attempt + 1, sleep=backoff)
                time.sleep(backoff)
                continue

            raise HealthApiError(
                f"Request to {path} failed ({resp.status_code})",
                status=resp.status_code, body=resp.text,
            )

        raise HealthApiError(f"Giving up on {path} after {MAX_RETRIES} attempts")

    # -- endpoints --------------------------------------------------------

    def paired_devices(self) -> list[dict]:
        """`users.pairedDevices.list` — the sync freshness gate.

        Google's own troubleshooting advice is to check `lastSyncTime` before
        concluding data is missing, so the poller uses this to avoid hammering
        the API when the watch simply has not synced.
        """
        data = self._request("users/me/pairedDevices")
        return data.get("pairedDevices", data.get("devices", []))

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict]:
        page_token: str | None = None
        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
            data = self._request(path, page_params)
            yield data
            page_token = data.get("nextPageToken")
            if not page_token:
                return

    @staticmethod
    def _extract_points(page: dict) -> list[dict]:
        """Pull the point list out of a page, whatever Google decided to call it."""
        for key in ("dataPoints", "points", "data"):
            if key in page and isinstance(page[key], list):
                return page[key]
        # Unknown envelope: hand back the whole page so it still reaches the raw
        # table, and let `circa probe` surface the shape.
        return [page] if page and "nextPageToken" not in page else []

    def fetch(
        self,
        dt: DataType,
        start: datetime,
        end: datetime,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 500,
    ) -> FetchResult:
        """Fetch one data type over [start, end).

        Tries a server-side filter first; on rejection, refetches unfiltered and
        trims locally so a wrong guess about filter syntax degrades performance
        rather than losing data.
        """
        result = FetchResult(data_type=dt.name)
        base = {"pageSize": page_size}

        if dt.time_filter:
            expr = (
                f'{dt.time_filter} >= "{_filter_value(start, dt.filter_format)}" AND '
                f'{dt.time_filter} < "{_filter_value(end, dt.filter_format)}"'
            )
            try:
                for page in self._paginate(dt.path, {**base, "filter": expr}):
                    result.points.extend(self._extract_points(page))
                    result.pages += 1
                    if result.pages >= max_pages:
                        log.warning("health_api.page_cap", data_type=dt.name)
                        break
                return result
            except HealthApiError as exc:
                if exc.status not in (400, 404):
                    raise
                log.warning("health_api.filter_rejected", data_type=dt.name,
                            filter=expr, detail=(exc.body or "")[:400])
                result.filter_error = f"{dt.time_filter}: {exc.body}"
                result.points.clear()
                result.pages = 0

        result.server_filtered = False
        for page in self._paginate(dt.path, base):
            result.points.extend(self._extract_points(page))
            result.pages += 1
            if result.pages >= max_pages:
                log.warning("health_api.page_cap", data_type=dt.name)
                break
        return result

    def sample_page(self, dt: DataType, page_size: int = 3) -> dict:
        """One tiny unfiltered page — used by `circa probe` to inspect shapes."""
        return self._request(dt.path, {"pageSize": page_size})


def _filter_value(ts: datetime, fmt: str) -> str:
    """Format the right-hand side of a v4 filter expression.

    Two shapes, both confirmed against the live API: daily types compare
    against a bare calendar date, everything else against an RFC3339 UTC
    instant. Google's published example used an offset-free timestamp, which
    the members we actually filter on reject.
    """
    if fmt == "date":
        return ts.astimezone(UTC).strftime("%Y-%m-%d")
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

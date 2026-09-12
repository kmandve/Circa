"""Google Calendar API client.

Scoped deliberately narrowly. Circa holds `calendar.app.created` (create and
manage only calendars this app made) plus `calendar.readonly` (see the existing
schedule). It never holds blanket write access, so a bug here cannot modify or
delete a real appointment — the worst case is a mess on Circa's own calendars,
which can be deleted wholesale.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import structlog

from circa.ingest.oauth import AuthError, get_access_token

log = structlog.get_logger(__name__)

BASE_URL = "https://www.googleapis.com/calendar/v3"
MIN_REQUEST_INTERVAL = 0.12
MAX_RETRIES = 5


class CalendarError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def reason(self) -> str:
        """Google's own sentence, which is the only part worth logging.

        A bare "failed (400)" cost an afternoon: the body said "Invalid
        foreground color" the whole time and nothing ever printed it.
        """
        try:
            message = json.loads(self.body or "{}")["error"]["message"]
        except (ValueError, KeyError, TypeError):
            return (self.body or "")[:200]
        return str(message)


class CalendarClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=30.0)
        self._owns = client is None
        self._last = 0.0

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def __enter__(self) -> CalendarClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last = time.monotonic()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        force_refresh = False

        for attempt in range(MAX_RETRIES):
            self._throttle()
            token = get_access_token("calendar", force_refresh=force_refresh)
            force_refresh = False
            resp = self._client.request(
                method, url, params=params, json=json,
                headers={"Authorization": f"Bearer {token}"},
            )

            if resp.status_code in (200, 201):
                return resp.json()
            if resp.status_code == 204:
                return {}
            if resp.status_code == 401 and attempt == 0:
                force_refresh = True
                continue
            if resp.status_code == 403 and "insufficient" in resp.text.lower():
                raise AuthError(
                    "Calendar scope insufficient. Re-run `circa auth`. If Google "
                    "rejects the narrow `calendar.app.created` scope for your "
                    "client, use `circa auth --broad-calendar`."
                )
            if resp.status_code == 404:
                raise CalendarError("not found", status=404, body=resp.text)
            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = min(2**attempt, 20)
                log.warning("gcal.retrying", status=resp.status_code, sleep=backoff)
                time.sleep(backoff)
                continue
            raise CalendarError(
                f"{method} {path} failed ({resp.status_code})",
                status=resp.status_code, body=resp.text,
            )
        raise CalendarError(f"giving up on {method} {path}")

    # -- calendars ---------------------------------------------------------

    def list_calendars(self) -> list[dict]:
        items: list[dict] = []
        page_token = None
        while True:
            params = {"maxResults": 250}
            if page_token:
                params["pageToken"] = page_token
            data = self._request("GET", "users/me/calendarList", params=params)
            items.extend(data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                return items

    def create_calendar(self, summary: str, description: str, timezone: str) -> dict:
        return self._request(
            "POST", "calendars",
            json={"summary": summary, "description": description, "timeZone": timezone},
        )

    def set_calendar_colour(self, calendar_id: str, color_id: str) -> dict:
        return self._request(
            "PATCH", f"users/me/calendarList/{calendar_id}",
            params={"colorRgbFormat": "false"},
            json={"colorId": color_id},
        )

    def set_calendar_rgb(
        self, calendar_id: str, background: str, foreground: str
    ) -> dict:
        """Set an arbitrary colour, rather than one of Google's eleven presets.

        `colorRgbFormat=true` is what unlocks real hex. It needs the
        `calendar.calendarlist` scope; with only `calendar.app.created` this
        answers 401.
        """
        return self._request(
            "PATCH", f"users/me/calendarList/{calendar_id}",
            params={"colorRgbFormat": "true"},
            json={"backgroundColor": background, "foregroundColor": foreground},
        )

    def delete_calendar(self, calendar_id: str) -> None:
        self._request("DELETE", f"calendars/{calendar_id}")

    # -- events ------------------------------------------------------------

    def list_events(
        self,
        calendar_id: str,
        time_min: str,
        time_max: str,
        private_property: str | None = None,
        single_events: bool = True,
    ) -> list[dict]:
        items: list[dict] = []
        page_token = None
        while True:
            params: dict[str, Any] = {
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": 2500,
                "singleEvents": str(single_events).lower(),
            }
            if single_events:
                params["orderBy"] = "startTime"
            if private_property:
                params["privateExtendedProperty"] = private_property
            if page_token:
                params["pageToken"] = page_token
            data = self._request("GET", f"calendars/{calendar_id}/events", params=params)
            items.extend(data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                return items

    def insert_event(self, calendar_id: str, body: dict) -> dict:
        return self._request("POST", f"calendars/{calendar_id}/events", json=body)

    def patch_event(self, calendar_id: str, event_id: str, body: dict) -> dict:
        return self._request(
            "PATCH", f"calendars/{calendar_id}/events/{event_id}", json=body
        )

    def delete_event(self, calendar_id: str, event_id: str) -> None:
        try:
            self._request("DELETE", f"calendars/{calendar_id}/events/{event_id}")
        except CalendarError as exc:
            if exc.status != 404:  # already gone is success
                raise

    def freebusy(self, calendar_ids: list[str], time_min: str, time_max: str) -> dict:
        return self._request(
            "POST", "freeBusy",
            json={
                "timeMin": time_min,
                "timeMax": time_max,
                "items": [{"id": cid} for cid in calendar_ids],
            },
        )

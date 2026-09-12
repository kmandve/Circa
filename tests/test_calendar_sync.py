"""Calendar sync idempotency.

The property under test: Circa recomputes on every poll, so an unchanged
forecast must produce **zero** calendar API writes. Without that, event IDs
churn and Google fires notifications many times a day.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from circa.gcal.blocks import Block
from circa.gcal.sync import push
from circa.settings_store import RuntimeSettings


class FakeCalendarClient:
    """Records every call so the test can assert on API traffic, not just state."""

    def __init__(self):
        self.calendars: dict[str, dict] = {}
        self.events: dict[str, dict] = {}
        self.calls: list[str] = []
        self._next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def list_calendars(self):
        self.calls.append("list_calendars")
        return [{"id": k, "summary": v["summary"]} for k, v in self.calendars.items()]

    def create_calendar(self, summary, description, timezone):
        self._next_id += 1
        cid = f"cal{self._next_id}@group.calendar.google.com"
        self.calendars[cid] = {"summary": summary}
        self.calls.append(f"create_calendar:{summary}")
        return {"id": cid, "summary": summary}

    def set_calendar_colour(self, calendar_id, color_id):
        self.calls.append("set_colour")
        return {}

    def set_calendar_rgb(self, calendar_id, background, foreground):
        self.calls.append(f"set_rgb:{background}")
        self.calendars.setdefault(calendar_id, {})["backgroundColor"] = background
        return {}

    def insert_event(self, calendar_id, body):
        self._next_id += 1
        eid = f"evt{self._next_id}"
        self.events[eid] = {"calendar": calendar_id, "body": body}
        self.calls.append(f"insert:{body['extendedProperties']['private']['block_key']}")
        return {"id": eid}

    def patch_event(self, calendar_id, event_id, body):
        self.events[event_id] = {"calendar": calendar_id, "body": body}
        self.calls.append(f"patch:{body['extendedProperties']['private']['block_key']}")
        return {"id": event_id}

    def delete_event(self, calendar_id, event_id):
        self.events.pop(event_id, None)
        self.calls.append(f"delete:{event_id}")

    def delete_calendar(self, calendar_id):
        self.calendars.pop(calendar_id, None)


def _block(key: str, hours_ahead: float = 4.0, title: str = "Peak Focus (±30m)") -> Block:
    start = datetime.now(UTC) + timedelta(hours=hours_ahead)
    return Block(
        key=key, category="focus", kind="peak_focus",
        start=start, end=start + timedelta(hours=2),
        title=title, description="why this block exists",
    )


@pytest.fixture
def client():
    return FakeCalendarClient()


def test_first_push_creates_calendars_and_events(client, db):
    settings = RuntimeSettings()
    with db() as s:
        report = push(s, [_block("focus:peak_focus:2026-09-02")], settings, client=client)
    assert report.created == 1
    assert report.updated == 0
    assert any(c.startswith("create_calendar") for c in client.calls)


def test_identical_second_push_makes_no_api_writes(client, db):
    """The core idempotency guarantee."""
    settings = RuntimeSettings()
    block = _block("focus:peak_focus:2026-09-02")

    with db() as s:
        push(s, [block], settings, client=client)
    client.calls.clear()

    with db() as s:
        report = push(s, [block], settings, client=client)

    assert report.unchanged == 1
    assert report.created == 0 and report.updated == 0
    writes = [c for c in client.calls if c.startswith(("insert:", "patch:", "delete:"))]
    assert writes == [], f"expected no writes, got {writes}"


def test_changed_block_patches_rather_than_recreating(client, db):
    """Patching keeps the event ID stable, which is what avoids a notification storm."""
    settings = RuntimeSettings()
    key = "focus:peak_focus:2026-09-02"

    with db() as s:
        push(s, [_block(key)], settings, client=client)
    original_ids = set(client.events)
    client.calls.clear()

    with db() as s:
        report = push(s, [_block(key, hours_ahead=5.0)], settings, client=client)

    assert report.updated == 1 and report.created == 0
    assert set(client.events) == original_ids  # same event, moved
    assert any(c.startswith("patch:") for c in client.calls)


def test_dropped_block_is_deleted(client, db):
    settings = RuntimeSettings()
    with db() as s:
        push(s, [_block("focus:peak_focus:a"), _block("focus:circadian_dip:a", 8.0)],
             settings, client=client)
    with db() as s:
        report = push(s, [_block("focus:peak_focus:a")], settings, client=client)
    assert report.deleted == 1


def test_disabling_a_category_clears_its_future_events(client, db):
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)
    with db() as s:
        report = push(s, [], RuntimeSettings(enable_focus=False), client=client)
    assert report.deleted >= 1


def test_events_are_transparent_so_they_never_mark_you_busy(client, db):
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)
    body = next(iter(client.events.values()))["body"]
    assert body["transparency"] == "transparent"
    assert body["extendedProperties"]["private"]["app"] == "circa"
    assert body["extendedProperties"]["private"]["block_key"] == "focus:peak_focus:a"


def test_blocks_beyond_the_horizon_are_not_written(client, db):
    settings = RuntimeSettings(forecast_horizon_hours=24)
    with db() as s:
        report = push(s, [_block("focus:far", hours_ahead=72)], settings, client=client)
    assert report.created == 0


def test_calendars_are_reused_not_duplicated_on_restart(client, db):
    settings = RuntimeSettings()
    with db() as s:
        push(s, [_block("focus:a")], settings, client=client)
    created_before = len([c for c in client.calls if c.startswith("create_calendar")])
    client.calls.clear()

    # Simulate a fresh database that has lost its calendar links but where the
    # calendars still exist on the Google account.
    from sqlalchemy import delete

    from circa.db.models import CalendarLink

    with db() as s:
        s.execute(delete(CalendarLink))
    with db() as s:
        push(s, [_block("focus:a")], settings, client=client)

    assert created_before > 0
    assert not any(c.startswith("create_calendar") for c in client.calls), (
        "calendars must be matched by name, not blindly recreated"
    )

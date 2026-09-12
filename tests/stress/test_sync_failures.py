"""Calendar sync under failure, not under sunshine.

Sync talks to a network service in the middle of a database transaction it has
deliberately released. Every way that call can fail is a way local state and
Google can disagree - and the failure mode that matters is duplicate events on
the user's real calendar.
"""

from __future__ import annotations

import pytest

from circa.gcal.client import CalendarError
from circa.gcal.sync import push
from circa.settings_store import NotificationPolicy, RuntimeSettings
from tests.test_calendar_sync import FakeCalendarClient, _block


class FlakyClient(FakeCalendarClient):
    """A fake that can be told exactly how to fail."""

    def __init__(self, fail_on_insert_after=None, insert_error=None,
                 patch_error=None, delete_error=None, raise_plain=False):
        super().__init__()
        self.fail_on_insert_after = fail_on_insert_after
        self.insert_error = insert_error
        self.patch_error = patch_error
        self.delete_error = delete_error
        self.raise_plain = raise_plain
        self.inserts = 0

    def insert_event(self, calendar_id, body):
        self.inserts += 1
        if self.fail_on_insert_after is not None and self.inserts > self.fail_on_insert_after:
            if self.raise_plain:
                raise ConnectionError("connection reset by peer")
            raise CalendarError("boom", status=self.insert_error or 500)
        return super().insert_event(calendar_id, body)

    def patch_event(self, calendar_id, event_id, body):
        if self.patch_error:
            raise CalendarError("gone", status=self.patch_error)
        return super().patch_event(calendar_id, event_id, body)

    def delete_event(self, calendar_id, event_id):
        if self.delete_error:
            raise CalendarError("nope", status=self.delete_error)
        return super().delete_event(calendar_id, event_id)


def _blocks(n, title="Peak Focus (±30m)"):
    return [_block(f"focus:peak_focus:b{i}", hours_ahead=4 + i, title=title) for i in range(n)]


# --- partial failure --------------------------------------------------------


def test_events_created_before_a_crash_are_not_duplicated_next_run(db):
    """The network phase buffers writes, so a crash halfway loses the record.

    Events already in Google would then be inserted a second time on the next
    poll, and the user gets duplicates on a real calendar with no way to tell
    which is live.
    """
    settings = RuntimeSettings()
    blocks = _blocks(4)

    first = FlakyClient(fail_on_insert_after=2, raise_plain=True)
    with db() as s:
        report = push(s, blocks, settings, client=first)

    # A transport error is not a CalendarError, so it must still be caught and
    # reported rather than escaping and stranding the two events that worked.
    assert len(report.errors) == 2, report.errors
    assert report.created == 2
    assert len(first.events) == 2

    second = FakeCalendarClient()
    second.calendars = first.calendars
    second.events = dict(first.events)
    with db() as s:
        report = push(s, blocks, settings, client=second)

    total = len([e for e in second.events.values()
                 if e["body"]["extendedProperties"]["private"]["block_key"].startswith("focus:")])
    assert total == 4, f"expected 4 events, found {total} - duplicates were created"


def test_a_single_failing_event_does_not_abort_the_rest(db):
    settings = RuntimeSettings()
    client = FlakyClient(fail_on_insert_after=1, insert_error=500)
    with db() as s:
        report = push(s, _blocks(3), settings, client=client)
    assert report.created == 1
    assert len(report.errors) == 2, report.errors


def test_recreate_after_404_that_also_fails_is_reported_not_raised(db):
    """A 404 triggers a recreate; if that recreate fails too, the handler must
    not let the exception escape and abandon every remaining block."""
    settings = RuntimeSettings()
    blocks = _blocks(3)
    with db() as s:
        push(s, blocks, settings, client=FakeCalendarClient())

    client = FlakyClient(patch_error=404, fail_on_insert_after=0, insert_error=500)
    changed = [_block(b.key, hours_ahead=9 + i, title="Peak Focus (±45m)")
               for i, b in enumerate(blocks)]
    with db() as s:
        report = push(s, changed, settings, client=client)
    assert len(report.errors) == 3, report.errors


# --- settings that must reach Google ---------------------------------------


def test_changing_notification_policy_reaches_the_calendar(db):
    """`_unchanged` compares text and times but not reminders.

    Turning notifications on therefore changed the event body while sync still
    considered the event identical, so the setting silently never applied.
    """
    from dataclasses import replace

    settings = RuntimeSettings(notifications=NotificationPolicy.NONE)
    block = _block("focus:peak_focus:x")
    block.notify_minutes = None
    client = FakeCalendarClient()
    with db() as s:
        push(s, [block], settings, client=client)

    # Identical in every respect except the reminder, so nothing else can be
    # what triggers the update.
    loud = replace(block, notify_minutes=15)
    with db() as s:
        report = push(s, [loud], settings, client=client)

    assert report.updated == 1, "reminder change was treated as no change"
    body = next(iter(client.events.values()))["body"]
    assert body["reminders"]["overrides"] == [{"method": "popup", "minutes": 15}]


def test_identical_push_still_makes_no_writes_after_the_reminder_fix(db):
    settings = RuntimeSettings()
    blocks = _blocks(3)
    client = FakeCalendarClient()
    with db() as s:
        push(s, blocks, settings, client=client)
    client.calls.clear()
    with db() as s:
        report = push(s, blocks, settings, client=client)
    assert report.unchanged == 3
    assert not [c for c in client.calls if c.startswith(("insert", "patch", "delete"))]


# --- degenerate inputs ------------------------------------------------------


def test_push_with_no_blocks_is_harmless(db):
    settings = RuntimeSettings()
    with db() as s:
        report = push(s, [], settings, client=FakeCalendarClient())
    assert report.created == 0 and not report.errors


def test_delete_failure_does_not_mark_the_block_gone_locally(db):
    """If Google refuses the delete, local state must keep tracking the event."""
    from sqlalchemy import select

    from circa.db.models import CalendarBlock

    settings = RuntimeSettings()
    client = FakeCalendarClient()
    blocks = _blocks(2)
    with db() as s:
        push(s, blocks, settings, client=client)

    failing = FlakyClient(delete_error=500)
    failing.calendars = client.calendars
    failing.events = dict(client.events)
    with db() as s:
        report = push(s, blocks[:1], settings, client=failing)
        assert report.errors
        row = s.scalar(select(CalendarBlock).where(CalendarBlock.block_key == blocks[1].key))
        assert row is not None
        assert not row.deleted, "block marked deleted locally though Google still has it"


# --- resurrection -----------------------------------------------------------


def test_a_block_that_returns_after_being_dropped_comes_back(db):
    """Google accepts a PATCH against a cancelled event and returns 200 without
    reviving it. Patching one therefore marked the block live locally while
    nothing appeared on the calendar, and the next poll saw it as unchanged and
    sent nothing - so any block ever dropped and later restored was gone for
    good. Found on the real calendar: a bright-light window that had been
    dropped by a clipping bug never reappeared after the bug was fixed.
    """
    from sqlalchemy import select

    from circa.db.models import CalendarBlock

    settings = RuntimeSettings()
    client = FakeCalendarClient()
    block = _block("light:morning_light:a")

    with db() as s:
        push(s, [block], settings, client=client)
    first_id = next(iter(client.events))

    # It stops being generated, so sync deletes it.
    with db() as s:
        push(s, [], settings, client=client)
    with db() as s:
        row = s.scalar(select(CalendarBlock).where(CalendarBlock.block_key == block.key))
        assert row.deleted is True
    assert first_id not in client.events

    # It comes back.
    with db() as s:
        report = push(s, [block], settings, client=client)

    assert report.created == 1, "revived block was patched instead of re-inserted"
    live = [
        e["body"]["extendedProperties"]["private"]["block_key"]
        for e in client.events.values()
    ]
    assert block.key in live, "the block never made it back onto the calendar"


def test_an_event_cancelled_behind_our_back_is_recreated(db):
    """The local deleted flag can be lost; the patch response still tells us."""

    class CancelledOnPatch(FakeCalendarClient):
        def patch_event(self, calendar_id, event_id, body):
            super().patch_event(calendar_id, event_id, body)
            return {"id": event_id, "status": "cancelled"}

    settings = RuntimeSettings()
    client = CancelledOnPatch()
    block = _block("light:morning_light:b")
    with db() as s:
        push(s, [block], settings, client=client)

    changed = _block(block.key, hours_ahead=9, title="Bright light window (±50m)")
    with db() as s:
        report = push(s, [changed], settings, client=client)
    assert report.created == 1, "a cancelled event was left cancelled"


# --- the past ---------------------------------------------------------------


def _seed_stored_event(db, client, key, kind, start, end, category="focus"):
    """Put a block into stored state directly.

    `push` refuses to write a forecast that has already started, so a spent one
    cannot be created through it - but it is exactly the state that accumulates
    in real use, as yesterday's blocks age out.
    """
    from circa.db.models import CalendarBlock, CalendarLink

    cal_id = f"cal-{category}@group.calendar.google.com"
    client.calendars[cal_id] = {"summary": f"Circa · {category.title()}"}
    event_id = f"evt-{key}"
    client.events[event_id] = {
        "calendar": cal_id,
        "body": {"extendedProperties": {"private": {"block_key": key, "kind": kind}}},
    }
    with db() as s:
        if s.get(CalendarLink, category) is None:
            s.add(CalendarLink(category=category, calendar_id=cal_id,
                               summary=f"Circa · {category.title()}",
                               color_id="4", enabled=True))
        s.add(CalendarBlock(
            block_key=key, category=category, kind=kind,
            target_date=start.date(), start_ts=start, end_ts=end,
            title="seeded", description="", google_event_id=event_id,
            model_version="0.2.0", deleted=False,
        ))
    return event_id


def test_a_spent_forecast_is_removed_from_the_calendar(db):
    """Forecasts used to be left in place once they had happened, so the
    calendar silted up with predictions nobody could act on - twenty-one of them
    after a few days."""
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from circa.db.models import CalendarBlock

    now = datetime.now(UTC)
    client = FakeCalendarClient()
    _seed_stored_event(db, client, "focus:peak_focus:yesterday", "peak_focus",
                       now - timedelta(hours=6), now - timedelta(hours=4))

    with db() as s:
        report = push(s, [_block("focus:peak_focus:today", hours_ahead=3)],
                      RuntimeSettings(), client=client)
    assert report.deleted == 1
    with db() as s:
        row = s.scalar(select(CalendarBlock).where(
            CalendarBlock.block_key == "focus:peak_focus:yesterday"))
        assert row.deleted is True


def test_a_block_in_progress_is_not_pulled_out_from_under_you(db):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    client = FakeCalendarClient()
    _seed_stored_event(db, client, "focus:peak_focus:running", "peak_focus",
                       now - timedelta(hours=1), now + timedelta(hours=1))

    with db() as s:
        report = push(s, [_block("focus:second_wind:later", hours_ahead=6)],
                      RuntimeSettings(), client=client)
    assert report.deleted == 0, "deleted a block that was still running"


def test_a_failed_run_does_not_wipe_the_calendar(db):
    """If block generation fails and produces nothing, leaving the calendar as
    it was is far better than clearing it."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    client = FakeCalendarClient()
    _seed_stored_event(db, client, "focus:peak_focus:a", "peak_focus",
                       now - timedelta(hours=6), now - timedelta(hours=4))

    with db() as s:
        report = push(s, [], RuntimeSettings(), client=client)   # nothing generated
    assert report.deleted == 0
    assert len(client.events) == 1


def test_recorded_nights_are_kept_even_though_they_are_in_the_past(db):
    """They are the half of the picture the calendar exists to show."""
    from datetime import UTC, datetime, timedelta

    from circa.gcal.blocks import SUPERSEDED_BY_RECORD, Block

    now = datetime.now(UTC)
    settings = RuntimeSettings()
    client = FakeCalendarClient()
    record = Block(
        key="sleep:sleep_actual:last-night", category="sleep", kind="sleep_actual",
        start=now - timedelta(hours=14), end=now - timedelta(hours=6),
        title="Slept 8h 00m", description="recorded",
        supersedes_kinds=SUPERSEDED_BY_RECORD,
    )
    with db() as s:
        push(s, [record], settings, client=client)
    # It is still generated on the next run, so it is never a deletion candidate.
    with db() as s:
        report = push(s, [record], settings, client=client)
    assert report.deleted == 0
    assert report.unchanged == 1


@pytest.mark.parametrize("status", [404, 410])
def test_deleting_an_event_that_is_already_gone_counts_as_done(db, status):
    """Google answers 404 for a deleted event and 410 for a cancelled one.

    Treating either as a failure left the local row marked live, so every
    subsequent run retried the same delete and logged the same error - sixteen
    of them per poll on the real calendar, permanently.
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from circa.db.models import CalendarBlock

    now = datetime.now(UTC)
    client = FlakyClient(delete_error=status)
    _seed_stored_event(db, client, "focus:peak_focus:gone", "peak_focus",
                       now - timedelta(hours=6), now - timedelta(hours=4))

    with db() as s:
        report = push(s, [_block("focus:peak_focus:new", hours_ahead=3)],
                      RuntimeSettings(), client=client)
    assert not report.errors, report.errors
    assert report.deleted == 1
    with db() as s:
        row = s.scalar(select(CalendarBlock).where(
            CalendarBlock.block_key == "focus:peak_focus:gone"))
        assert row.deleted is True, "row still live, so the delete will be retried forever"


def test_a_real_delete_failure_is_still_reported(db):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    client = FlakyClient(delete_error=500)
    _seed_stored_event(db, client, "focus:peak_focus:broken", "peak_focus",
                       now - timedelta(hours=6), now - timedelta(hours=4))
    with db() as s:
        report = push(s, [_block("focus:peak_focus:new", hours_ahead=3)],
                      RuntimeSettings(), client=client)
    assert report.errors

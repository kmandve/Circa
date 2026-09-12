"""Forced-wake detection and the passive validation probes.

Forced-wake detection is the single highest-value use of calendar read access -
it decides which nights count as evidence about the endogenous clock. Getting it
wrong does not fail loudly; it quietly reweights the whole sleep channel.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from circa.db.models import SleepSession
from circa.gcal.schedule import free_intervals, mark_forced_wakes
from circa.settings_store import RuntimeSettings

TZ = ZoneInfo("America/Chicago")
NOW = datetime.now(UTC)


class BusyClient:
    """A calendar client that reports whatever busy intervals a test wants."""

    def __init__(self, busy=(), calendars=None):
        self.busy = list(busy)
        self._calendars = calendars if calendars is not None else [
            {"id": "primary", "summary": "Work", "selected": True}
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def list_calendars(self):
        return self._calendars

    def freebusy(self, ids, time_min, time_max):
        return {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": s.isoformat().replace("+00:00", "Z"),
                         "end": e.isoformat().replace("+00:00", "Z")}
                        for s, e in self.busy
                    ]
                }
            }
        }


def _night(db, wake_local_hour, days_ago=1, hours=8.0, external_id=None):
    day = (NOW - timedelta(days=days_ago)).astimezone(TZ).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = (day + timedelta(hours=wake_local_hour)).astimezone(UTC)
    start = end - timedelta(hours=hours)
    with db() as s:
        s.add(SleepSession(
            external_id=external_id or f"n/{days_ago}/{wake_local_hour}",
            start_ts=start, end_ts=end, tz_name="America/Chicago",
            utc_offset_seconds=int(end.astimezone(TZ).utcoffset().total_seconds()),
            sleep_date=end.astimezone(TZ).date(), is_main_sleep=True,
            tst_minutes=hours * 60, midpoint_ts=start + (end - start) / 2,
        ))
    return start, end


def _forced(db):
    with db() as s:
        return {n.external_id: n.wake_forced for n in s.scalars(select(SleepSession))}


# --- the basics -------------------------------------------------------------


def test_a_meeting_just_after_wake_marks_it_forced(db):
    _, end = _night(db, wake_local_hour=7.0, external_id="meeting")
    client = BusyClient([(end + timedelta(minutes=30), end + timedelta(hours=1))])
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=client)
    assert _forced(db)["meeting"] is True


def test_a_clear_calendar_leaves_a_late_wake_free(db):
    _night(db, wake_local_hour=9.5, external_id="free")
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient([]))
    assert _forced(db)["free"] is False


def test_a_meeting_much_later_in_the_day_does_not_count(db):
    _, end = _night(db, wake_local_hour=9.0, external_id="afternoon")
    client = BusyClient([(end + timedelta(hours=6), end + timedelta(hours=7))])
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient([]) if False else client)
    assert _forced(db)["afternoon"] is False


# --- the ways it goes wrong -------------------------------------------------


def test_an_all_day_busy_event_does_not_mark_every_wake_as_forced(db):
    """A multi-day "Vacation" or "Out of office" marked busy covers every wake
    in it. Those are exactly the days when the wake is *least* forced, so
    treating them as alarm-driven inverts the signal on the most valuable
    nights available.
    """
    _night(db, wake_local_hour=9.5, days_ago=1, external_id="holiday-1")
    _night(db, wake_local_hour=10.0, days_ago=2, external_id="holiday-2")
    week = [(NOW - timedelta(days=4), NOW + timedelta(days=2))]
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient(week))
    forced = _forced(db)
    assert forced["holiday-1"] is False, "a week-long busy block forced a 09:30 wake"
    assert forced["holiday-2"] is False


@pytest.mark.parametrize("hour", list(range(0, 24)))
def test_every_legal_early_hour_setting_works(db, hour):
    """`forced_wake_before_hour - 3` is used to build a `time()`, which raises
    for any setting under 3."""
    _night(db, wake_local_hour=8.0, external_id="h")
    with db() as s:
        mark_forced_wakes(
            s, RuntimeSettings(forced_wake_before_hour=hour), client=BusyClient([])
        )


def test_a_very_early_wake_is_forced_even_with_an_empty_calendar(db):
    _night(db, wake_local_hour=4.5, external_id="dawn")
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient([]))
    assert _forced(db)["dawn"] is True


def test_a_failing_calendar_does_not_mark_everything_forced(db):
    """Losing calendar access must degrade to "unknown", not to "all forced"."""
    class Broken(BusyClient):
        def list_calendars(self):
            from circa.gcal.client import CalendarError

            raise CalendarError("nope", status=500)

    _night(db, wake_local_hour=9.0, external_id="broken")
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=Broken())
    assert _forced(db)["broken"] is False


def test_circa_own_calendars_are_never_read_as_commitments(db):
    """Circa writes a "Biological night" every day. Reading its own output back
    as a commitment would mark every single wake forced."""
    _, end = _night(db, wake_local_hour=9.0, external_id="self")
    client = BusyClient(
        [(end + timedelta(minutes=10), end + timedelta(hours=1))],
        calendars=[{"id": "circa-sleep", "summary": "Circa · Sleep", "selected": True}],
    )
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=client)
    assert _forced(db)["self"] is False


def test_marking_is_idempotent(db):
    _, end = _night(db, wake_local_hour=7.0, external_id="idem")
    client = BusyClient([(end + timedelta(minutes=20), end + timedelta(hours=1))])
    with db() as s:
        first = mark_forced_wakes(s, RuntimeSettings(), client=client)
    with db() as s:
        second = mark_forced_wakes(s, RuntimeSettings(), client=client)
    assert first == 1 and second == 0


# --- free intervals ---------------------------------------------------------


def test_free_intervals_are_the_exact_complement():
    start = datetime(2026, 9, 10, 8, tzinfo=UTC)
    end = datetime(2026, 9, 10, 18, tzinfo=UTC)
    busy = [
        (datetime(2026, 9, 10, 9, tzinfo=UTC), datetime(2026, 9, 10, 10, tzinfo=UTC)),
        (datetime(2026, 9, 10, 9, 30, tzinfo=UTC), datetime(2026, 9, 10, 11, tzinfo=UTC)),
        (datetime(2026, 9, 10, 15, tzinfo=UTC), datetime(2026, 9, 10, 16, tzinfo=UTC)),
    ]
    free = free_intervals(busy, start, end)
    assert free == [
        (start, datetime(2026, 9, 10, 9, tzinfo=UTC)),
        (datetime(2026, 9, 10, 11, tzinfo=UTC), datetime(2026, 9, 10, 15, tzinfo=UTC)),
        (datetime(2026, 9, 10, 16, tzinfo=UTC), end),
    ]
    for lo, hi in free:
        for b_lo, b_hi in busy:
            assert not (lo < b_hi and hi > b_lo), "a free interval overlaps a busy one"


def test_free_intervals_handle_a_fully_booked_day():
    start = datetime(2026, 9, 10, 8, tzinfo=UTC)
    end = datetime(2026, 9, 10, 18, tzinfo=UTC)
    assert free_intervals([(start, end)], start, end) == []


def test_free_intervals_handle_no_commitments():
    start = datetime(2026, 9, 10, 8, tzinfo=UTC)
    end = datetime(2026, 9, 10, 18, tzinfo=UTC)
    assert free_intervals([], start, end) == [(start, end)]


def test_the_local_heuristic_runs_without_calendar_access(db):
    """`wake_forced` starts NULL and `free_wake_probe` excludes NULL, so a run
    that never classifies a night leaves the strongest passive phase anchor
    permanently empty."""
    from circa.validate.probes import free_wake_probe

    _night(db, wake_local_hour=9.5, external_id="offline")
    with db() as s:
        assert free_wake_probe(s) == []
        mark_forced_wakes(s, RuntimeSettings(), use_calendar=False)
    assert _forced(db)["offline"] is False
    with db() as s:
        assert len(free_wake_probe(s)) == 1


def test_the_offline_heuristic_still_catches_a_dawn_wake(db):
    _night(db, wake_local_hour=4.0, external_id="dawn-offline")
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), use_calendar=False)
    assert _forced(db)["dawn-offline"] is True


def test_a_read_only_pipeline_run_classifies_nights(db):
    """`circa run --no-calendar`, and every /api/curve request."""
    from circa.db.models import SleepSession

    _night(db, wake_local_hour=9.0, external_id="readonly")
    from circa.pipeline import run

    with db() as s:
        run(s, push_calendar=False)
    with db() as s:
        night = s.scalar(select(SleepSession).where(SleepSession.external_id == "readonly"))
        assert night.wake_forced is not None, "night left unclassified by a read-only run"


def test_the_offline_heuristic_never_overwrites_calendar_evidence(db):
    """It cannot see commitments, so anything it decides is weaker.

    Letting it write over a conclusion drawn from the calendar turned every
    alarm-driven wake back into a "free" wake - and free wakes are the strongest
    passive phase anchor there is, so they must not be manufactured.
    """
    _, end = _night(db, wake_local_hour=8.0, external_id="after-meeting")
    client = BusyClient([(end + timedelta(minutes=20), end + timedelta(hours=1))])
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=client)
    assert _forced(db)["after-meeting"] is True

    # A later run with no calendar access must leave it alone.
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), use_calendar=False)
    assert _forced(db)["after-meeting"] is True, "offline run erased calendar evidence"


def test_a_calendar_run_may_still_correct_itself(db):
    """The restriction is only on the offline path; real evidence can change."""
    _, end = _night(db, wake_local_hour=8.0, external_id="cancelled")
    busy = [(end + timedelta(minutes=20), end + timedelta(hours=1))]
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient(busy))
    assert _forced(db)["cancelled"] is True
    with db() as s:
        mark_forced_wakes(s, RuntimeSettings(), client=BusyClient([]))
    assert _forced(db)["cancelled"] is False

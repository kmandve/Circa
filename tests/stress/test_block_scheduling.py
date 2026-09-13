"""Blocks must be useful whatever time of day the poll happens to run.

The phase markers recur daily, but the block builders hang off a single
instant. Getting that instant wrong is invisible in a unit test that only ever
runs at one clock time, and shows up on the calendar as a missing night.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from circa.alertness.model import compute, grid
from circa.alertness.process_s import SleepWakeHistory
from circa.gcal.blocks import WAKE_REQUIRED_KINDS, build_all
from circa.phase.confidence import assess
from circa.settings_store import RuntimeSettings

TZ = ZoneInfo("America/Chicago")
OFFSET = -5 * 3600
HORIZON = 48


def _build(as_of, dlmo_local_h, sleep_start_h=23.4, sleep_hours=8.0, conf=None):
    settings = RuntimeSettings()
    conf = conf or assess(30, 95, 0.9, 20)
    base = datetime(2026, 9, 1, tzinfo=TZ)
    episodes = [
        (
            (base + timedelta(days=d, hours=sleep_start_h)).astimezone(UTC),
            (base + timedelta(days=d, hours=sleep_start_h + sleep_hours)).astimezone(UTC),
        )
        for d in range(25)
    ]
    history = SleepWakeHistory(
        episodes, as_of - timedelta(days=10), as_of + timedelta(hours=HORIZON)
    )
    times = grid(as_of - timedelta(hours=12), as_of + timedelta(hours=HORIZON), minutes=10)
    curve = compute(
        times, OFFSET, history,
        np.random.default_rng(0).normal((dlmo_local_h + 7) % 24, 0.4, 200),
    )
    # Mirror the engine: DLMO snapped to the nearest daily recurrence, CBTmin
    # placed relative to it.
    day = datetime(2026, 9, 10, tzinfo=TZ) + timedelta(hours=dlmo_local_h)
    dlmo = min(
        (day.astimezone(UTC) + timedelta(days=k) for k in (-2, -1, 0, 1, 2)),
        key=lambda c: abs((c - as_of).total_seconds()),
    )
    blocks = build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=conf, settings=settings, offset=OFFSET,
        from_ts=as_of, to_ts=as_of + timedelta(hours=HORIZON),
    )
    return blocks, history, as_of


POLL_HOURS = list(range(0, 24))
DLMO_HOURS = [19.0, 21.4, 23.0, 0.5, 2.0]


@pytest.mark.parametrize("dlmo_h", DLMO_HOURS)
@pytest.mark.parametrize("poll_hour", POLL_HOURS)
def test_upcoming_night_is_always_scheduled(poll_hour, dlmo_h):
    """A poll at any hour must leave a biological night the user can still use.

    `dlmo_ts` is the recurrence nearest to now, so for a night owl every poll
    between midnight and 10:00 pointed at *last* night's melatonin onset. The
    sleep window built from it had already ended, sync dropped it as past, and
    tonight's was never generated - the Sleep calendar simply went empty for
    most of the morning.
    """
    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    blocks, _, _ = _build(as_of, dlmo_h)
    nights = [b for b in blocks if b.kind == "sleep_window"]
    assert nights, f"no biological night generated at {poll_hour:02d}:00 local"
    assert any(b.end > as_of for b in nights), (
        f"every biological night at {poll_hour:02d}:00 local had already ended: "
        f"{[(b.start.isoformat(), b.end.isoformat()) for b in nights]}"
    )


@pytest.mark.parametrize("dlmo_h", DLMO_HOURS)
@pytest.mark.parametrize("poll_hour", [0, 5, 9, 11, 14, 17, 20, 23])
def test_blocks_stay_inside_the_forecast_horizon(poll_hour, dlmo_h):
    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    blocks, _, _ = _build(as_of, dlmo_h)
    horizon_end = as_of + timedelta(hours=HORIZON)
    for b in blocks:
        assert b.start >= as_of - timedelta(hours=26), f"{b.kind} far in the past"
        assert b.start <= horizon_end, f"{b.kind} beyond the horizon"


@pytest.mark.parametrize("dlmo_h", DLMO_HOURS)
@pytest.mark.parametrize("poll_hour", [0, 4, 8, 10, 13, 16, 19, 22])
def test_waking_advice_never_lands_during_sleep_at_any_poll_hour(poll_hour, dlmo_h):
    """The multi-day anchors must not smuggle blocks past the wake check."""
    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    blocks, history, _ = _build(as_of, dlmo_h)
    for b in blocks:
        if b.kind not in WAKE_REQUIRED_KINDS:
            continue
        t = b.start
        while t < b.end:
            assert not history.is_asleep(t), (
                f"{b.kind} at {b.start.astimezone(TZ)} overlaps sleep "
                f"(poll {poll_hour:02d}:00, dlmo {dlmo_h})"
            )
            t += timedelta(minutes=15)


@pytest.mark.parametrize("poll_hour", [0, 6, 9, 12, 15, 18, 21])
def test_block_keys_are_unique_per_poll(poll_hour):
    """Duplicate keys would make two events fight over one calendar entry."""
    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    blocks, _, _ = _build(as_of, 21.4)
    keys = [b.key for b in blocks]
    assert len(keys) == len(set(keys)), f"duplicate block keys: {sorted(keys)}"


@pytest.mark.parametrize("poll_hour", [3, 8, 11, 16, 21])
def test_cbtmin_stays_inside_the_night_it_describes(poll_hour):
    """CBTmin is ~7h after DLMO; it must never be printed from another night."""
    from circa.phase.engine import _local_hour_after, _local_hours_to_ts

    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    dlmo = _local_hours_to_ts(21.4, as_of, OFFSET)
    cbt = _local_hour_after(4.4, dlmo, OFFSET)
    gap = (cbt - dlmo).total_seconds() / 3600
    assert 4.0 <= gap <= 10.0, f"CBTmin {gap:+.1f}h from its own DLMO"


# --- a spent forecast must leave the calendar --------------------------------


def _marker(key: str, kind: str, starts_in_hours: float, minutes: int = 15):
    """A short marker - the shape the start-anchored window got wrong."""
    from datetime import UTC, datetime, timedelta

    from circa.gcal.blocks import Block

    start = datetime.now(UTC) + timedelta(hours=starts_in_hours)
    return Block(
        key=key, category="light", kind=kind,
        start=start, end=start + timedelta(minutes=minutes),
        title="Dim lights", description="why",
    )


def test_a_spent_marker_stays_for_the_rest_of_the_day(db):
    """The rule this replaced deleted a forecast the moment it ended, which made
    the day unreadable by the evening: by 8pm there was no record of what the
    plan had been at 8am.

    A block now stays for the circadian day it belongs to whether or not it has
    happened, and is still refreshed while it stands, so a revised estimate
    moves it rather than leaving it stale.
    """
    from circa.gcal.sync import push
    from circa.settings_store import RuntimeSettings
    from tests.test_calendar_sync import FakeCalendarClient, _key

    client = FakeCalendarClient()
    settings = RuntimeSettings()
    key = _key("light", "dim_light")

    with db() as s:
        push(s, [_marker(key, "dim_light", 0.25)], settings, client=client)
    assert len(client.events) == 1

    # An hour on, it is over. It belongs to today, so it stays - and moving it
    # still works.
    with db() as s:
        report = push(s, [_marker(key, "dim_light", -1.0)], settings, client=client)

    assert report.deleted == 0, "a spent block was removed before the day ended"
    assert report.updated == 1, "a spent block stopped tracking the model"


def test_a_long_block_still_in_progress_is_kept_and_refreshed(db):
    """The other half of the same rule. A sleep window is eight hours long, so a
    two-hour grace measured from its start dropped it out of the refresh in the
    middle of the night it described - present on the calendar, but no longer
    tracking the model."""
    from datetime import UTC, datetime, timedelta

    from circa.gcal.blocks import Block
    from circa.gcal.sync import push
    from circa.settings_store import RuntimeSettings
    from tests.test_calendar_sync import FakeCalendarClient

    client = FakeCalendarClient()
    settings = RuntimeSettings()
    started = datetime.now(UTC) - timedelta(hours=4)

    def night(title: str) -> Block:
        return Block(
            key="sleep:sleep_window:a", category="sleep", kind="sleep_window",
            start=started, end=started + timedelta(hours=8),
            title=title, description="why",
        )

    with db() as s:
        push(s, [night("Sleep")], settings, client=client)
    with db() as s:
        report = push(s, [night("Sleep (revised)")], settings, client=client)

    assert report.deleted == 0, "an in-progress night was dropped"
    assert report.updated == 1, "an in-progress night stopped tracking the model"
    body = next(iter(client.events.values()))["body"]
    assert body["summary"] == "Sleep (revised)"


# --- the day turns over at the wake, not at midnight -------------------------


def _sleep_row(s, start, end):
    from circa.db.models import SleepSession

    s.add(SleepSession(
        external_id=f"t/{start.isoformat()}", start_ts=start, end_ts=end,
        tz_name="America/Chicago", utc_offset_seconds=-18000,
        sleep_date=end.date(), is_main_sleep=True,
        tst_minutes=(end - start).total_seconds() / 60 - 20,
        time_in_bed_minutes=(end - start).total_seconds() / 60,
        midpoint_ts=start + (end - start) / 2, wake_forced=False,
    ))


def test_the_circadian_day_is_the_day_you_woke_into(db):
    """Not the calendar date. A night that runs past midnight belongs to the day
    you woke up on, which is why tonight's sleep window and this morning's
    grogginess carry the same day."""
    from datetime import UTC, datetime, timedelta
    from zoneinfo import ZoneInfo

    from circa.db.session import session_scope
    from circa.gcal.sync import circadian_day

    tz = ZoneInfo("America/Chicago")
    now = datetime.now(UTC)
    offset = int(now.astimezone(tz).utcoffset().total_seconds())
    woke = now - timedelta(hours=6)

    with session_scope() as s:
        _sleep_row(s, woke - timedelta(hours=8), woke)
    with session_scope() as s:
        assert circadian_day(s, now, offset) == (woke + timedelta(seconds=offset)).date()


def test_a_wake_that_never_arrives_does_not_freeze_the_calendar(db):
    """If the watch is not worn there is no wake to anchor on. Falling back to
    the calendar day keeps the day turning over; without it the calendar would
    stall on a day that finished, forever."""
    from datetime import UTC, datetime, timedelta

    from circa.db.session import session_scope
    from circa.gcal.sync import CIRCADIAN_DAY_STALE_HOURS, circadian_day

    now = datetime.now(UTC)
    offset = -18000
    stale = now - timedelta(hours=CIRCADIAN_DAY_STALE_HOURS + 6)

    with session_scope() as s:
        _sleep_row(s, stale - timedelta(hours=8), stale)
    with session_scope() as s:
        assert circadian_day(s, now, offset) == (now + timedelta(seconds=offset)).date()

    # And with no sleep recorded at all.
    with session_scope() as s:
        from circa.db.models import SleepSession
        s.query(SleepSession).delete()
    with session_scope() as s:
        assert circadian_day(s, now, offset) == (now + timedelta(seconds=offset)).date()


def test_waking_replaces_the_whole_previous_day(db):
    """The rule in one test: yesterday's plan goes and today's appears together,
    at the wake - not at midnight, and not block by block as each one expires.

    Played out on a frozen clock as it actually happens: a day's plan is written
    the morning it belongs to, and is still there that evening. The next
    morning's sleep is logged, and the whole set turns over at once.
    """
    from datetime import UTC, datetime, timedelta
    from zoneinfo import ZoneInfo

    from freezegun import freeze_time

    from circa.db.session import session_scope
    from circa.gcal.blocks import Block
    from circa.gcal.sync import push
    from circa.settings_store import RuntimeSettings
    from tests.test_calendar_sync import FakeCalendarClient

    tz = ZoneInfo("America/Chicago")
    client = FakeCalendarClient()
    settings = RuntimeSettings()

    def plan(day, kinds, at):
        return [
            Block(key=f"focus:{k}:{day.isoformat()}", category="focus", kind=k,
                  start=at + timedelta(hours=i + 1),
                  end=at + timedelta(hours=i + 1, minutes=45),
                  title=k, description="why")
            for i, k in enumerate(kinds)
        ]

    # --- Monday, 9am local: woke an hour ago, the day's plan goes up ---------
    monday_9am = datetime(2026, 9, 14, 14, 0, tzinfo=UTC)      # 9am Chicago
    monday = monday_9am.astimezone(tz).date()
    with freeze_time(monday_9am):
        with session_scope() as s:
            _sleep_row(s, monday_9am - timedelta(hours=9), monday_9am - timedelta(hours=1))
        with db() as s:
            push(s, plan(monday, ["peak_focus", "circadian_dip"], monday_9am),
                 settings, client=client)
    assert len([e for e in client.events.values() if not e.get("deleted")]) == 2

    # --- Monday, 10pm: both are long over, and both are still there ----------
    #
    # A third block, still ahead, matters here: with nothing left to write the
    # prune is skipped anyway by the guard against wiping a calendar on an empty
    # run, and the test would pass without proving anything.
    monday_10pm = monday_9am + timedelta(hours=13)
    with freeze_time(monday_10pm):
        evening = plan(monday, ["wind_down"], monday_10pm)
        with db() as s:
            report = push(
                s,
                plan(monday, ["peak_focus", "circadian_dip"], monday_9am) + evening,
                settings, client=client,
            )
        assert report.deleted == 0, "the day's plan was pruned before the day ended"
    assert len([e for e in client.events.values() if not e.get("deleted")]) == 3

    # --- Tuesday, 9am: this morning's sleep lands. The day turns over. -------
    tuesday_9am = monday_9am + timedelta(days=1)
    tuesday = tuesday_9am.astimezone(tz).date()
    assert tuesday != monday
    with freeze_time(tuesday_9am):
        with session_scope() as s:
            _sleep_row(s, tuesday_9am - timedelta(hours=9), tuesday_9am - timedelta(hours=1))
        with db() as s:
            report = push(s, plan(tuesday, ["peak_focus", "circadian_dip", "second_wind"],
                                  tuesday_9am), settings, client=client)

    assert report.deleted == 3, f"Monday's plan survived the wake: {report}"
    assert report.created == 3, f"Tuesday's plan was not written: {report}"
    assert len([e for e in client.events.values() if not e.get("deleted")]) == 3

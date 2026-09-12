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

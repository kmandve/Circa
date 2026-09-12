"""Coherence invariants swept across configurations.

These exist because a whole class of bug kept escaping example-based tests:
outputs that are individually well-formed but contradict each other. The one
that reached the real calendar told the user to seek bright light at 04:15
while the Sleep calendar simultaneously showed "Biological night" until 06:30.

Rather than assert one scenario, these sweep sleep schedules, chronotypes,
confidence tiers, DST transitions, midnight-wrapping phases and every policy
setting, and assert properties that must hold for all of them.
"""

from __future__ import annotations

import itertools
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from circa.alertness.model import compute, grid
from circa.alertness.process_s import SleepWakeHistory
from circa.gcal.blocks import WAKE_REQUIRED_KINDS, build_all
from circa.phase.confidence import assess
from circa.settings_store import (
    Chronotype,
    LowConfidencePolicy,
    NotificationPolicy,
    RuntimeSettings,
)

# Raw health values must never reach the calendar: it is a separate disclosure
# surface (shared views, other devices, lock-screen notifications).
MEASURED_VALUE = re.compile(r"\d+\s*(bpm|ms\b|°C)", re.I)


def _scenario(sleep_start_h, sleep_hours, dlmo_h, conf, settings, anchor, offset=0):
    episodes = [
        (anchor + timedelta(days=d, hours=sleep_start_h),
         anchor + timedelta(days=d, hours=sleep_start_h + sleep_hours))
        for d in range(10)
    ]
    history = SleepWakeHistory(episodes, anchor, anchor + timedelta(days=10))
    times = grid(anchor + timedelta(days=5), anchor + timedelta(days=7), minutes=10)
    dlmo = anchor + timedelta(days=5, hours=dlmo_h)
    cbt = dlmo + timedelta(hours=7)
    curve = compute(
        times, offset, history,
        np.random.default_rng(0).normal((dlmo_h + 7) % 24, 0.4, 200),
    )
    blocks = build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=cbt,
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=conf, settings=settings, offset=offset,
        from_ts=anchor + timedelta(days=5),
    )
    return blocks, history


def _overlaps_sleep(block, history) -> bool:
    t = block.start
    while t < block.end:
        if history.is_asleep(t):
            return True
        t += timedelta(minutes=15)
    return False


SCHEDULES = [(0.0, 8.0), (1.5, 7.0), (22.5, 8.5), (3.0, 6.0), (23.0, 9.0)]
CONFIDENCES = [
    assess(30, 85, 0.95, 10),    # mature
    assess(14, 130, 0.8, 40),    # personalized
    assess(3, 260, 0.5, None),   # cold start, poor data
    assess(40, 95, 0.9, 200),    # mature but channels disagree
]


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("conf", CONFIDENCES)
@pytest.mark.parametrize("chronotype", [Chronotype.NIGHT_OWL, Chronotype.MORNING])
def test_waking_advice_never_lands_during_sleep(schedule, conf, chronotype):
    """The core invariant: advice you are asleep for is not advice."""
    sleep_start, sleep_hours = schedule
    settings = RuntimeSettings(chronotype=chronotype, target_sleep_hours=sleep_hours)
    wake_h = (sleep_start + sleep_hours) % 24
    dlmo_h = (wake_h - 9) % 24
    blocks, history = _scenario(
        sleep_start, sleep_hours, dlmo_h, conf, settings, datetime(2026, 9, 1, tzinfo=UTC)
    )
    for block in blocks:
        if block.kind in WAKE_REQUIRED_KINDS:
            assert not _overlaps_sleep(block, history), (
                f"{block.kind} {block.start}..{block.end} overlaps predicted sleep "
                f"({sleep_start}+{sleep_hours}h)"
            )


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_blocks_are_internally_well_formed(schedule):
    sleep_start, sleep_hours = schedule
    settings = RuntimeSettings(target_sleep_hours=sleep_hours)
    conf = assess(30, 90, 0.95, 15)
    blocks, _ = _scenario(sleep_start, sleep_hours, (sleep_start - 2) % 24, conf,
                          settings, datetime(2026, 9, 1, tzinfo=UTC))
    keys = [b.key for b in blocks]
    assert len(keys) == len(set(keys)), "block keys must be unique for idempotency"
    for b in blocks:
        assert b.end > b.start
        assert (b.end - b.start) <= timedelta(hours=12)
        assert b.title.strip() and (b.description or "").strip()


# DST transitions and a phase sitting exactly on the midnight wrap.
ANCHORS = [
    datetime(2026, 9, 1, tzinfo=UTC),
    datetime(2026, 3, 6, tzinfo=UTC),    # DST begins Mar 8
    datetime(2026, 10, 29, tzinfo=UTC),  # DST ends Nov 1
]


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("dlmo_h", [22.0, 0.0, 1.5, 19.0])
def test_survives_dst_and_the_midnight_wrap(anchor, dlmo_h):
    settings = RuntimeSettings()
    conf = assess(30, 95, 0.9, 20)
    blocks, history = _scenario(0.0, 8.0, dlmo_h, conf, settings, anchor, offset=-6 * 3600)
    assert blocks, "a mature estimate should produce blocks"
    for b in blocks:
        assert b.end > b.start
        assert (b.end - b.start) < timedelta(hours=26)
        if b.kind in WAKE_REQUIRED_KINDS:
            assert not _overlaps_sleep(b, history)


@pytest.mark.parametrize(
    "notifications,low_confidence",
    list(itertools.product(list(NotificationPolicy), list(LowConfidencePolicy))),
)
def test_policies_are_enforced(notifications, low_confidence):
    settings = RuntimeSettings(
        notifications=notifications, low_confidence_policy=low_confidence
    )
    low = assess(3, 260, 0.5, None)
    blocks, _ = _scenario(0.0, 8.0, 22.0, low, settings, datetime(2026, 9, 1, tzinfo=UTC))

    if notifications is NotificationPolicy.NONE:
        assert all(b.notify_minutes is None for b in blocks)
    if low_confidence is LowConfidencePolicy.WRITE_NOTHING:
        assert blocks == []
    if low_confidence is LowConfidencePolicy.ROBUST_ONLY:
        assert {b.category for b in blocks} <= {"sleep", "light", "debug"}


def test_calendar_never_carries_measured_health_values():
    """Google Calendar is a separate disclosure surface from the app."""
    settings = RuntimeSettings(enable_debug_calendar=False)
    conf = assess(30, 90, 0.95, 15)
    blocks, _ = _scenario(0.0, 8.0, 22.0, conf, settings, datetime(2026, 9, 1, tzinfo=UTC))
    assert blocks
    for b in blocks:
        assert not MEASURED_VALUE.search(b.title), b.title
        assert not MEASURED_VALUE.search(b.description or ""), b.kind


def test_high_confidence_produces_the_expected_block_set():
    """Guards against a clipping rule silently deleting whole categories."""
    settings = RuntimeSettings()
    conf = assess(40, 85, 0.95, 10)
    blocks, _ = _scenario(0.0, 8.0, 22.0, conf, settings, datetime(2026, 9, 1, tzinfo=UTC))
    kinds = {b.kind for b in blocks}
    for expected in ("sleep_window", "wind_down", "morning_light", "dim_light",
                     "caffeine_cutoff", "last_meal", "workout", "peak_focus"):
        assert expected in kinds, f"{expected} missing: {sorted(kinds)}"

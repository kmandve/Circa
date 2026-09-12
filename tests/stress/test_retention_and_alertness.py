"""Retention must never delete what the models read, and the alertness model
must stay finite on degenerate sleep histories.

Retention runs unattended on a schedule. A mistake there is silent and
irreversible, so the property is stated directly: everything the phase and
alertness layers consume survives a prune.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import func, select

from circa.db.models import (
    CalendarBlock,
    DailyMetric,
    HeartRateMinute,
    HeartRateSample,
    PhaseEstimate,
    SleepSession,
    SleepStage,
    StepMinute,
)

NOW = datetime.now(UTC)
OLD = NOW - timedelta(days=400)


def _seed_old_everything(db):
    with db() as s:
        s.add(SleepSession(
            external_id="ancient", start_ts=OLD, end_ts=OLD + timedelta(hours=8),
            tz_name="America/Chicago", utc_offset_seconds=-18000,
            sleep_date=(OLD + timedelta(hours=8)).date(), is_main_sleep=True,
            tst_minutes=440.0, midpoint_ts=OLD + timedelta(hours=4),
        ))
        s.flush()
        sid = s.scalar(select(SleepSession.id))
        s.add(SleepStage(session_id=sid, start_ts=OLD, end_ts=OLD + timedelta(hours=2),
                         stage="DEEP"))
        for i in range(50):
            s.add(HeartRateMinute(
                ts=OLD + timedelta(minutes=i), bpm_median=60.0, bpm_min=55,
                bpm_max=70, n_samples=12, active_fraction=0.0,
            ))
            s.add(HeartRateSample(ts=OLD + timedelta(seconds=5 * i), bpm=60))
            s.add(StepMinute(ts=OLD + timedelta(minutes=i), steps=30))
        s.add(DailyMetric(metric_date=OLD.date(), metric="daily-resting-heart-rate",
                          value=58.0))
        s.add(PhaseEstimate(
            target_date=OLD.date(), computed_at=OLD, model_version="0.2.0",
            dlmo_ts=OLD, cbtmin_ts=OLD + timedelta(hours=7), phase_sd_minutes=50.0,
            confidence_tier=1, confidence_q=0.5, n_nights=10,
        ))
        s.add(CalendarBlock(
            block_key="old", category="sleep", target_date=OLD.date(),
            kind="sleep_window", start_ts=OLD, end_ts=OLD + timedelta(hours=8),
            title="Biological night", model_version="0.2.0",
        ))


SURVIVORS = [
    (SleepSession, "sleep sessions"),
    (SleepStage, "sleep stages"),
    (HeartRateMinute, "1-minute heart rate"),
    (StepMinute, "step minutes"),
    (DailyMetric, "daily metrics"),
    (PhaseEstimate, "phase estimates"),
    (CalendarBlock, "calendar blocks"),
]


@pytest.mark.parametrize("model,label", SURVIVORS)
def test_retention_never_deletes_what_the_models_read(db, model, label):
    from circa.ingest.retention import run_retention

    _seed_old_everything(db)
    with db() as s:
        before = s.scalar(select(func.count()).select_from(model))
    run_retention()
    with db() as s:
        after = s.scalar(select(func.count()).select_from(model))
    assert after == before, f"retention deleted {before - after} {label}"


def test_retention_does_prune_the_things_it_is_supposed_to(db):
    """Otherwise the storage budget the whole deployment rests on is fiction."""
    from circa.ingest.retention import run_retention

    _seed_old_everything(db)
    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateSample)) == 50
    run_retention()
    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateSample)) == 0


def test_retention_on_an_empty_database_is_a_no_op(db):
    from circa.ingest.retention import run_retention

    report = run_retention()
    assert report.raw_deleted == 0
    assert report.hr_samples_deleted == 0


def test_retention_keeps_recent_heart_rate_samples(db):
    from circa.ingest.retention import run_retention

    with db() as s:
        for i in range(20):
            s.add(HeartRateSample(ts=NOW - timedelta(minutes=i), bpm=61))
    run_retention()
    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateSample)) == 20


def test_retention_is_idempotent(db):
    from circa.ingest.retention import run_retention

    _seed_old_everything(db)
    first = run_retention()
    second = run_retention()
    assert second.hr_samples_deleted == 0
    assert second.raw_deleted == 0
    assert first.hr_samples_deleted == 50


# --- alertness --------------------------------------------------------------

from circa.alertness.model import compute, grid  # noqa: E402
from circa.alertness.process_s import SleepWakeHistory, simulate  # noqa: E402

BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _curve(episodes, hours=48):
    history = SleepWakeHistory(episodes, BASE, BASE + timedelta(hours=hours))
    times = grid(BASE, BASE + timedelta(hours=hours), minutes=10)
    return compute(times, -5 * 3600, history,
                   np.random.default_rng(0).normal(4.0, 0.4, 100))


DEGENERATE = [
    pytest.param([], id="no sleep at all"),
    pytest.param([(BASE, BASE + timedelta(hours=48))], id="asleep the whole window"),
    pytest.param([(BASE, BASE + timedelta(seconds=1))], id="one-second episode"),
    pytest.param([(BASE + timedelta(hours=2), BASE + timedelta(hours=1))], id="ends before it starts"),
    pytest.param(
        [(BASE, BASE + timedelta(hours=8)), (BASE + timedelta(hours=4),
                                             BASE + timedelta(hours=12))],
        id="overlapping episodes",
    ),
    pytest.param(
        [(BASE + timedelta(hours=i), BASE + timedelta(hours=i, minutes=20))
         for i in range(0, 48, 2)],
        id="24 fragmented naps",
    ),
    pytest.param([(BASE - timedelta(days=400), BASE - timedelta(days=399))],
                 id="entirely outside the window"),
]


@pytest.mark.parametrize("episodes", DEGENERATE)
def test_alertness_stays_finite_on_degenerate_histories(episodes):
    """A NaN here propagates into every block boundary downstream."""
    curve = _curve(episodes)
    assert np.all(np.isfinite(curve.alertness)), "non-finite alertness"
    assert len(curve.times) == len(curve.alertness) == len(curve.asleep)
    assert np.all(np.abs(curve.alertness) < 100), "alertness left any sane range"


@pytest.mark.parametrize("episodes", DEGENERATE)
def test_process_s_stays_within_its_bounds(episodes):
    history = SleepWakeHistory(episodes, BASE, BASE + timedelta(hours=48))
    times = grid(BASE, BASE + timedelta(hours=48), minutes=10)
    s_values = simulate(history, times)
    assert np.all(np.isfinite(s_values))
    assert np.all(s_values >= -1e-6) and np.all(s_values <= 1.0 + 1e-6), (
        f"process S left [0, 1]: min={s_values.min()}, max={s_values.max()}"
    )


def test_sleep_pressure_falls_during_sleep_and_rises_awake():
    """The sign of the two-process model must not be inverted."""
    episodes = [(BASE + timedelta(hours=24), BASE + timedelta(hours=32))]
    history = SleepWakeHistory(episodes, BASE, BASE + timedelta(hours=48))
    times = grid(BASE, BASE + timedelta(hours=48), minutes=10)
    s_values = simulate(history, times)
    awake_idx = [i for i, t in enumerate(times) if not history.is_asleep(t)]
    asleep_idx = [i for i, t in enumerate(times) if history.is_asleep(t)]
    assert s_values[asleep_idx[-1]] < s_values[asleep_idx[0]], "S rose during sleep"
    early_wake = [i for i in awake_idx if i < asleep_idx[0]]
    assert s_values[early_wake[-1]] > s_values[early_wake[0]], "S fell while awake"


# --- oscillator -------------------------------------------------------------


def test_a_collapsed_oscillator_falls_back_instead_of_reporting_a_phase():
    """Constant dim light drives the limit cycle toward zero amplitude.

    A collapsed oscillator has no phase, so continuing to read a DLMO off it
    would produce a confident-looking number with nothing behind it.
    """
    from circa.phase.oscillator import (
        MIN_EQUILIBRATED_AMPLITUDE,
        default_initial_condition,
        equilibrate,
    )

    hours = np.arange(0, 24, 0.1)
    state = equilibrate(np.full_like(hours, 0.0), np.ones_like(hours), hours, taux=24.2)
    assert np.all(np.isfinite(state))
    amplitude = float(np.hypot(state[0], state[1]))
    assert amplitude >= MIN_EQUILIBRATED_AMPLITUDE or np.allclose(
        state, default_initial_condition()
    )


def test_equilibration_is_quiet_on_a_realistic_schedule(recwarn):
    """The upstream convergence warning fired on every single run."""
    from circa.phase.oscillator import equilibrate

    hours = np.arange(0, 24, 0.1)
    light = np.where((hours >= 8) & (hours < 22), 400.0, 2.0)
    wake = np.where((hours >= 7.5) & (hours < 23.5), 1.0, 0.0)
    for taux in (24.0, 24.2, 24.4, 24.6):
        equilibrate(light, wake, hours, taux=taux)
    assert not [w for w in recwarn if "equilibrate" in str(w.message)]


def test_equilibrated_state_is_a_healthy_limit_cycle():
    from circa.phase.oscillator import equilibrate

    hours = np.arange(0, 24, 0.1)
    light = np.where((hours >= 8) & (hours < 22), 400.0, 2.0)
    wake = np.where((hours >= 7.5) & (hours < 23.5), 1.0, 0.0)
    state = equilibrate(light, wake, hours, taux=24.4)
    assert 0.5 < float(np.hypot(state[0], state[1])) < 3.0


def test_stale_calendar_tombstones_are_pruned(db):
    """Block keys carry a date, so anything that moves a block to a different
    day leaves the old row behind - and every one is re-read on every sync."""
    from circa.ingest.retention import run_retention

    old = NOW - timedelta(days=60)
    with db() as s:
        s.add(CalendarBlock(
            block_key="light:morning_light:old", category="light", kind="morning_light",
            target_date=old.date(), start_ts=old, end_ts=old + timedelta(hours=2),
            title="Bright light window", model_version="0.2.0", deleted=True,
        ))
        s.add(CalendarBlock(
            block_key="light:morning_light:recent", category="light",
            kind="morning_light", target_date=NOW.date(),
            start_ts=NOW - timedelta(days=1), end_ts=NOW - timedelta(hours=22),
            title="Bright light window", model_version="0.2.0", deleted=True,
        ))
        s.add(CalendarBlock(
            block_key="light:morning_light:live", category="light",
            kind="morning_light", target_date=NOW.date(),
            start_ts=old, end_ts=old + timedelta(hours=2),
            title="Bright light window", model_version="0.2.0", deleted=False,
        ))

    report = run_retention()
    assert report.tombstones_deleted == 1

    with db() as s:
        keys = {b.block_key for b in s.scalars(select(CalendarBlock))}
    assert "light:morning_light:old" not in keys
    assert "light:morning_light:recent" in keys, "a recent tombstone was dropped too early"
    assert "light:morning_light:live" in keys, "retention deleted a live block"

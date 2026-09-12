"""Heart-rate channel identifiability.

A failure the earlier synthetic tests missed. For anyone with a regular sleep
schedule, a binary `asleep` regressor is nearly collinear with the 24-hour
harmonic, so the fit assigns the whole day/night swing to the indicator and
leaves the harmonic chasing a few bpm of wobble. It then reports a concentration
comparable to the sleep channel and drags the fused estimate hours early, with
an acrophase in the middle of the night -- heart rate supposedly peaking
mid-sleep.

The channel must abstain when it cannot identify the rhythm, not vote quietly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from circa.db.models import HeartRateMinute, SleepSession, StepMinute
from circa.db.session import session_scope
from circa.phase import hr_phase
from circa.settings_store import RuntimeSettings

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def _seed(days=4, amplitude=8.0, noise=2.0, sleep_start=0.0, sleep_hours=8.0, seed=0):
    """Write synthetic HR/steps/sleep with a known circadian acrophase.

    The generated rhythm peaks at 16:00 local, so a working channel should
    recover an acrophase near there.
    """
    rng = np.random.default_rng(seed)
    with session_scope() as s:
        # Anchor sleep to midnight, not to NOW. Getting this wrong put the
        # synthetic sleep window on top of the synthetic acrophase, so the
        # channel correctly refused to report a peak it could not observe -
        # a broken fixture, not a broken channel.
        midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        for d in range(days):
            start = midnight - timedelta(days=days - d) + timedelta(hours=sleep_start)
            end = start + timedelta(hours=sleep_hours)
            s.add(SleepSession(
                external_id=f"s{d}", start_ts=start, end_ts=end, tz_name="UTC",
                utc_offset_seconds=0, sleep_date=end.date(), is_main_sleep=True,
                tst_minutes=sleep_hours * 60, time_in_bed_minutes=sleep_hours * 60,
                waso_minutes=0, latency_minutes=5, efficiency=0.95,
                midpoint_ts=start + timedelta(hours=sleep_hours / 2), excluded=False))
        for m in range(days * 24 * 60):
            ts = NOW - timedelta(minutes=m)
            h = ts.hour + ts.minute / 60
            asleep = sleep_start <= h < sleep_start + sleep_hours
            circadian = amplitude * np.cos(2 * np.pi * (h - 16.0) / 24)
            steps = 0.0 if asleep else max(0.0, rng.normal(10, 15))
            bpm = 62 + circadian + (0 if asleep else 8) + 0.25 * steps + rng.normal(0, noise)
            s.add(HeartRateMinute(ts=ts, bpm_median=float(np.clip(bpm, 40, 180)),
                                  bpm_min=int(bpm - 3), bpm_max=int(bpm + 3),
                                  n_samples=12, active_fraction=0.0 if asleep else 0.2))
            if steps > 0:
                s.add(StepMinute(ts=ts, steps=float(steps)))


def test_recovers_a_strong_rhythm():
    """With a clear signal the channel should find the right acrophase."""
    _seed(amplitude=10.0, noise=1.5)
    with session_scope() as s:
        obs = hr_phase.estimate(s, RuntimeSettings(), as_of=NOW, tz_offset_seconds=0)
    assert obs is not None, "a strong clean rhythm must be detected"
    error = abs(((obs.acrophase_hours - 16.0 + 12) % 24) - 12)
    assert error < 4.0, f"acrophase {obs.acrophase_hours:.1f}h, expected ~16h"
    assert obs.detail["sleep_excluded"] is True


def test_abstains_when_the_rhythm_is_buried_in_noise():
    """A 1 bpm oscillation under 10 bpm of scatter carries no phase information."""
    _seed(amplitude=0.5, noise=10.0)
    with session_scope() as s:
        obs = hr_phase.estimate(s, RuntimeSettings(), as_of=NOW, tz_offset_seconds=0)
    assert obs is None


def test_sleep_is_excluded_rather_than_regressed_out():
    """The collinearity that caused the original failure must be gone."""
    _seed(amplitude=8.0)
    with session_scope() as s:
        obs = hr_phase.estimate(s, RuntimeSettings(), as_of=NOW, tz_offset_seconds=0)
    assert obs is not None
    assert "asleep" not in obs.detail["regressors"], (
        "a binary sleep indicator is collinear with the 24h harmonic on a "
        "regular schedule and must not be a regressor"
    )
    assert obs.detail["awake_fraction_of_valid"] < 1.0


def test_weak_rhythm_gets_lower_concentration_than_strong_one():
    """kappa must track signal quality, not just fit tidiness."""
    _seed(amplitude=12.0, noise=1.5, seed=1)
    with session_scope() as s:
        strong = hr_phase.estimate(s, RuntimeSettings(), as_of=NOW, tz_offset_seconds=0)

    with session_scope() as s:
        for model in (HeartRateMinute, StepMinute, SleepSession):
            s.query(model).delete()
    _seed(amplitude=3.0, noise=6.0, seed=2)
    with session_scope() as s:
        weak = hr_phase.estimate(s, RuntimeSettings(), as_of=NOW, tz_offset_seconds=0)

    assert strong is not None
    if weak is not None:
        assert weak.kappa < strong.kappa
        assert weak.detail["amplitude_snr"] < strong.detail["amplitude_snr"]


def test_channel_can_be_disabled_entirely():
    _seed()
    with session_scope() as s:
        assert hr_phase.estimate(
            s, RuntimeSettings(hr_window_hours=0), as_of=NOW, tz_offset_seconds=0
        ) is None

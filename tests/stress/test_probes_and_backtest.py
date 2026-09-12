"""The passive validation layer.

These are the only check on whether the phase estimate is any good, since there
is no DLMO measurement to compare against. A bug here does not make the model
wrong - it makes a wrong model look right.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from circa.db.models import SleepSession
from circa.validate.metrics import interval_width_minutes, phase_metrics
from circa.validate.probes import (
    Probe,
    free_wake_phase_estimate,
    free_wake_probe,
    latency_consistency,
    sleep_latency_probe,
    waso_probe,
)

TZ = ZoneInfo("America/Chicago")
NOW = datetime.now(UTC)


def _night(db, days_ago, wake_local=8.0, hours=8.0, latency=12.0, waso=20.0,
           forced=False, tib=None):
    day = (NOW - timedelta(days=days_ago)).astimezone(TZ).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = (day + timedelta(hours=wake_local)).astimezone(UTC)
    start = end - timedelta(hours=hours)
    with db() as s:
        s.add(SleepSession(
            external_id=f"p/{days_ago}", start_ts=start, end_ts=end,
            tz_name="America/Chicago", utc_offset_seconds=-18000,
            sleep_date=end.astimezone(TZ).date(), is_main_sleep=True,
            tst_minutes=hours * 60, time_in_bed_minutes=tib if tib is not None else hours * 60,
            latency_minutes=latency, waso_minutes=waso,
            midpoint_ts=start + (end - start) / 2, wake_forced=forced,
        ))


# --- probes -----------------------------------------------------------------


def test_probes_on_an_empty_database_are_empty_not_fatal(db):
    with db() as s:
        assert sleep_latency_probe(s) == []
        assert free_wake_probe(s) == []
        assert waso_probe(s) == []
        assert free_wake_phase_estimate(s) is None


def test_forced_wakes_are_excluded_from_the_free_wake_probe(db):
    """The whole value of this probe is that the wake was not scheduled."""
    for d in range(1, 5):
        _night(db, d, forced=(d % 2 == 0))
    with db() as s:
        probes = free_wake_probe(s)
    assert probes
    assert len(probes) == 2


def test_a_night_missing_its_metrics_is_skipped(db):
    with db() as s:
        s.add(SleepSession(
            external_id="bare", start_ts=NOW - timedelta(hours=10),
            end_ts=NOW - timedelta(hours=2), tz_name="America/Chicago",
            utc_offset_seconds=-18000, sleep_date=NOW.date(), is_main_sleep=True,
        ))
    with db() as s:
        assert sleep_latency_probe(s) == []
        assert waso_probe(s) == []
        # NULL means "not yet classified", which is deliberately not the same
        # as "known to be free" - the probe's whole value is that the wake was
        # not scheduled, so an unclassified night cannot count.
        assert free_wake_probe(s) == []


def test_zero_time_in_bed_does_not_divide_by_zero(db):
    _night(db, 1, tib=0.0)
    with db() as s:
        assert waso_probe(s) == []


def test_waso_fraction_is_a_fraction(db):
    for d in range(1, 6):
        _night(db, d, waso=30.0, hours=8.0)
    with db() as s:
        for p in waso_probe(s):
            assert 0.0 <= p.value <= 1.0


def test_free_wake_phase_needs_three_nights(db):
    for d in (1, 2):
        _night(db, d)
    with db() as s:
        assert free_wake_phase_estimate(s) is None
    _night(db, 3)
    with db() as s:
        assert free_wake_phase_estimate(s) is not None


def test_free_wake_phase_lands_in_the_evening_for_a_morning_waker(db):
    """Wake ~07:00 implies CBTmin ~04:30 and DLMO ~21:30."""
    for d in range(1, 8):
        _night(db, d, wake_local=7.0)
    with db() as s:
        dlmo = free_wake_phase_estimate(s)
    assert 20.0 <= dlmo <= 23.0, f"DLMO from free wakes came out at {dlmo:.1f}h"


def test_free_wake_phase_wraps_correctly_for_a_very_early_waker(db):
    """Wake at 05:00 implies DLMO the previous evening - the subtraction wraps
    past midnight and must not come back as a negative or a 20-hour error."""
    for d in range(1, 8):
        _night(db, d, wake_local=5.0)
    with db() as s:
        dlmo = free_wake_phase_estimate(s)
    assert 0.0 <= dlmo < 24.0
    assert 18.5 <= dlmo <= 21.0, f"{dlmo:.1f}h"


# --- latency consistency ----------------------------------------------------


def _latency_probes(pairs):
    return [
        Probe(name="sleep_latency", day=date(2026, 9, 1) + timedelta(days=i),
              value=latency, detail={"attempt_hour": hour})
        for i, (hour, latency) in enumerate(pairs)
    ]


def test_latency_consistency_needs_enough_nights():
    out = latency_consistency(_latency_probes([(23.0, 10)] * 4), 23.0)
    assert out["correlation"] is None


def test_latency_consistency_finds_the_expected_relationship():
    """Attempting sleep further from the predicted gate should take longer."""
    probes = _latency_probes([
        (23.0, 8), (23.5, 10), (0.5, 20), (1.5, 35), (2.5, 50), (22.0, 15),
    ])
    out = latency_consistency(probes, 23.0)
    assert out["correlation"] is not None
    assert out["correlation"] > 0.5, out


def test_latency_consistency_survives_no_variation():
    out = latency_consistency(_latency_probes([(23.0, 10)] * 6), 23.0)
    assert out["correlation"] is None
    assert "no variation" in out["note"]


def test_latency_distance_wraps_around_midnight():
    """An attempt at 00:30 against a gate at 23:30 is one hour away, not 23."""
    near = _latency_probes([(0.5, 9), (23.5, 8), (23.0, 10), (0.0, 9), (1.0, 12), (22.5, 11)])
    out = latency_consistency(near, 23.5)
    assert out["mean_latency_minutes"] == pytest.approx(9.83, abs=0.1)
    assert out["correlation"] is not None


# --- metrics ----------------------------------------------------------------


def test_phase_metrics_treat_error_as_circular():
    predicted = np.array([0.5])
    actual = np.array([23.5])
    m = phase_metrics(predicted, actual)
    assert m.mae_minutes == pytest.approx(60.0)
    assert m.bias_minutes == pytest.approx(60.0)


def test_phase_metrics_on_a_perfect_model():
    hours = np.array([21.0, 22.0, 23.0, 0.0, 1.0])
    m = phase_metrics(hours, hours)
    assert m.mae_minutes == 0 and m.p30 == 1.0 and m.worst_decile_minutes == 0


def test_coverage_counts_an_interval_that_wraps_midnight():
    actual = np.array([23.9, 0.2, 12.0])
    intervals = [(23.0, 1.0), (23.0, 1.0), (23.0, 1.0)]
    m = phase_metrics(np.array([0.0, 0.0, 0.0]), actual, ci80=intervals)
    assert m.coverage_80 == pytest.approx(2 / 3)


def test_interval_width_is_the_forward_arc():
    assert interval_width_minutes([(23.0, 1.0)]) == pytest.approx(120.0)
    assert interval_width_minutes([(1.0, 23.0)]) == pytest.approx(22 * 60)


def test_phase_metrics_rejects_an_empty_prediction():
    with pytest.raises(ValueError):
        phase_metrics(np.array([]), np.array([]))

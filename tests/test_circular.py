"""Circular arithmetic. Phase is an angle; treating it as a number is the
single easiest way to produce a confidently wrong answer."""

from __future__ import annotations

import numpy as np
import pytest

from circa.phase.circular import (
    circular_difference,
    circular_mean,
    circular_quantiles,
    circular_sd_hours,
    combine_von_mises,
    kappa_from_sd_hours,
    sd_hours_from_kappa,
    von_mises_logpdf,
    wrap_hours,
)


def test_mean_across_midnight():
    """23:30 and 00:30 average to midnight, not to noon."""
    assert circular_mean(np.array([23.5, 0.5])) == pytest.approx(0.0, abs=1e-6)
    # The naive arithmetic mean would be 12.0 - the exact opposite.
    assert np.mean([23.5, 0.5]) == 12.0


def test_difference_takes_the_short_way_round():
    assert circular_difference(0.5, 23.5) == pytest.approx(1.0)
    assert circular_difference(23.5, 0.5) == pytest.approx(-1.0)
    assert circular_difference(1.0, 1.0) == pytest.approx(0.0)
    # Never reports more than 12 hours of error.
    assert abs(circular_difference(0.0, 13.0)) <= 12.0


def test_weighted_mean_follows_the_weights():
    hours = np.array([22.0, 23.0, 0.0])
    heavy_late = circular_mean(hours, np.array([0.1, 0.1, 5.0]))
    assert abs(circular_difference(heavy_late, 0.0)) < 0.4


def test_sd_is_small_for_concentrated_and_large_for_scattered():
    tight = circular_sd_hours(np.array([22.0, 22.1, 21.9, 22.05]))
    scattered = circular_sd_hours(np.array([0.0, 6.0, 12.0, 18.0]))
    assert tight < 0.3
    assert scattered > 3.0


def test_kappa_and_sd_round_trip():
    for sd in (0.25, 0.5, 1.0, 2.0):
        assert sd_hours_from_kappa(kappa_from_sd_hours(sd)) == pytest.approx(sd, rel=0.1)


def test_combining_two_channels_lands_between_them_and_sharpens():
    """Agreeing channels should produce a tighter estimate than either alone."""
    mu, kappa = combine_von_mises([(22.0, 4.0), (23.0, 4.0)])
    assert 22.0 < mu < 23.0
    assert kappa > 4.0


def test_combining_wraps_correctly():
    mu, _ = combine_von_mises([(23.5, 10.0), (0.5, 10.0)])
    assert abs(circular_difference(mu, 0.0)) < 0.05


def test_disagreeing_channels_produce_a_weak_combination():
    _, kappa_agree = combine_von_mises([(22.0, 5.0), (22.1, 5.0)])
    _, kappa_disagree = combine_von_mises([(22.0, 5.0), (10.0, 5.0)])
    assert kappa_disagree < kappa_agree


def test_von_mises_logpdf_peaks_at_the_mean_and_wraps():
    at_mean = von_mises_logpdf(np.array([22.0]), 22.0, 5.0)
    nearby = von_mises_logpdf(np.array([22.5]), 22.0, 5.0)
    far = von_mises_logpdf(np.array([10.0]), 22.0, 5.0)
    assert at_mean > nearby > far
    # 23.5 and 0.5 are equidistant from 0.0 either side of midnight.
    a = von_mises_logpdf(np.array([23.5]), 0.0, 5.0)
    b = von_mises_logpdf(np.array([0.5]), 0.0, 5.0)
    assert a == pytest.approx(b)


def test_von_mises_logpdf_survives_huge_kappa():
    """Large concentrations must not overflow the Bessel normaliser."""
    value = von_mises_logpdf(np.array([22.0]), 22.0, 5000.0)
    assert np.isfinite(value).all()


def test_quantiles_span_the_midnight_wrap():
    rng = np.random.default_rng(0)
    samples = wrap_hours(rng.normal(0.0, 0.5, 4000))  # straddles midnight
    q = circular_quantiles(samples, quantiles=(0.1, 0.9))
    assert abs(circular_difference(q[0.1], -0.64)) < 0.25
    assert abs(circular_difference(q[0.9], 0.64)) < 0.25

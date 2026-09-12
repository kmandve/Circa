"""Property-based sweep of the circular statistics that everything else rests on.

Phase is an angle. Every bug in this module is a bug in every number the app
shows, so these tests assert the mathematical invariants directly rather than
checking hand-picked examples.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from circa.phase.circular import (
    HOURS,
    circular_difference,
    circular_mean,
    circular_quantiles,
    circular_sd_hours,
    combine_von_mises,
    kappa_from_resultant,
    kappa_from_sd_hours,
    resultant_length,
    sd_hours_from_kappa,
    von_mises_logpdf,
    wrap_hours,
)

hours = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
clock = st.floats(min_value=0.0, max_value=24.0, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


# --- wrapping ---------------------------------------------------------------


@given(hours)
def test_wrap_is_always_in_the_half_open_day(h):
    """[0, 24) is a contract: 24.0 renders as '24:00' and breaks date maths."""
    w = wrap_hours(h)
    assert 0.0 <= w < HOURS


@given(hours, st.integers(min_value=-500, max_value=500))
def test_wrap_is_invariant_to_whole_days(h, days):
    assert wrap_hours(h) == pytest.approx(wrap_hours(h + days * HOURS), abs=1e-6)


@given(hours)
def test_wrap_is_idempotent(h):
    assert wrap_hours(wrap_hours(h)) == pytest.approx(wrap_hours(h), abs=1e-12)


# --- differences ------------------------------------------------------------


@given(hours, hours)
def test_difference_is_within_half_a_day(a, b):
    d = float(circular_difference(a, b))
    assert -HOURS / 2 < d <= HOURS / 2


@given(hours, hours)
def test_difference_is_antisymmetric(a, b):
    forward = float(circular_difference(a, b))
    back = float(circular_difference(b, a))
    # +12 is its own negation on a circle, so the boundary is exempt.
    assume(abs(abs(forward) - HOURS / 2) > 1e-6)
    assert forward == pytest.approx(-back, abs=1e-6)


@given(hours, hours)
def test_difference_reconstructs_the_second_operand(a, b):
    d = float(circular_difference(a, b))
    assert wrap_hours(b + d) == pytest.approx(wrap_hours(a), abs=1e-6)


# --- means ------------------------------------------------------------------


@given(st.lists(clock, min_size=1, max_size=60))
def test_mean_is_in_range(vals):
    assert 0.0 <= circular_mean(np.array(vals)) < HOURS


@given(st.lists(clock, min_size=1, max_size=40), hours)
def test_mean_is_equivariant_under_rotation(vals, shift):
    """Rotating every input must rotate the mean by the same amount.

    This is the property that a naive arithmetic mean fails, and the reason
    this module exists at all.
    """
    arr = np.array(vals)
    r = resultant_length(arr)
    # A mean is only well defined when the sample is not near-uniform.
    assume(r > 0.05)
    direct = circular_mean(wrap_hours(arr + shift))
    rotated = wrap_hours(circular_mean(arr) + shift)
    assert float(circular_difference(direct, rotated)) == pytest.approx(0.0, abs=1e-4)


@given(clock, st.integers(min_value=1, max_value=30))
def test_mean_of_identical_values_is_that_value(v, n):
    assert float(circular_difference(circular_mean(np.full(n, v)), v)) == pytest.approx(0, abs=1e-6)


@given(st.lists(clock, min_size=1, max_size=30))
def test_mean_ignores_weight_scale(vals):
    arr = np.array(vals)
    assume(resultant_length(arr) > 0.05)
    a = circular_mean(arr, np.ones_like(arr))
    b = circular_mean(arr, np.full_like(arr, 7.5))
    assert float(circular_difference(a, b)) == pytest.approx(0.0, abs=1e-6)


# --- spread -----------------------------------------------------------------


@given(st.lists(clock, min_size=1, max_size=50))
def test_resultant_length_is_a_fraction(vals):
    assert 0.0 <= resultant_length(np.array(vals)) <= 1.0 + 1e-12


@given(st.lists(clock, min_size=1, max_size=50))
def test_circular_sd_is_finite_and_bounded(vals):
    sd = circular_sd_hours(np.array(vals))
    assert np.isfinite(sd)
    assert 0.0 <= sd <= HOURS / 4 + 1e-9


@given(st.floats(min_value=0.02, max_value=0.999, allow_nan=False))
def test_kappa_from_resultant_is_positive_and_finite(r):
    k = kappa_from_resultant(r)
    assert np.isfinite(k) and k > 0


@given(st.floats(min_value=0.05, max_value=5.0, allow_nan=False))
def test_kappa_and_sd_are_inverses(sd):
    """A prior expressed as a spread must survive the round trip to kappa."""
    back = sd_hours_from_kappa(kappa_from_sd_hours(sd))
    assert back == pytest.approx(sd, rel=0.12, abs=0.06)


@given(positive)
def test_sd_from_kappa_is_finite(k):
    sd = sd_hours_from_kappa(k)
    assert np.isfinite(sd) and sd >= 0


# --- likelihood -------------------------------------------------------------


@given(clock, clock, positive)
def test_von_mises_logpdf_is_finite_and_peaks_at_mu(x, mu, kappa):
    at_x = float(von_mises_logpdf(np.array([x]), mu, kappa)[0])
    at_mu = float(von_mises_logpdf(np.array([mu]), mu, kappa)[0])
    assert np.isfinite(at_x) and np.isfinite(at_mu)
    assert at_mu >= at_x - 1e-9


@given(positive)
def test_von_mises_integrates_to_one(kappa):
    """An unnormalised likelihood would silently reweight the channels."""
    grid = np.linspace(0, HOURS, 4001)[:-1]
    density = np.exp(von_mises_logpdf(grid, 12.0, kappa))
    # density is per radian; convert the hour-spaced grid accordingly.
    integral = np.sum(density) * (grid[1] - grid[0]) * (2 * np.pi / HOURS)
    assert integral == pytest.approx(1.0, rel=1e-3)


# --- fusion -----------------------------------------------------------------


@given(clock, positive, positive)
def test_combining_a_channel_with_itself_only_sharpens_it(mu, k1, k2):
    combined_mu, combined_kappa = combine_von_mises([(mu, k1), (mu, k2)])
    assert float(circular_difference(combined_mu, mu)) == pytest.approx(0, abs=1e-6)
    assert combined_kappa == pytest.approx(k1 + k2, rel=1e-6)


@given(clock, clock, positive, positive)
def test_fusion_lands_between_its_inputs(a, b, ka, kb):
    """The fused estimate may not sit outside the arc spanned by the inputs."""
    mu, _ = combine_von_mises([(a, ka), (b, kb)])
    span = float(circular_difference(b, a))
    assume(abs(abs(span) - HOURS / 2) > 1e-3)
    offset = float(circular_difference(mu, a))
    assert min(0.0, span) - 1e-6 <= offset <= max(0.0, span) + 1e-6


# --- quantiles --------------------------------------------------------------


@given(st.lists(clock, min_size=2, max_size=80))
def test_quantiles_are_in_range(vals):
    for v in circular_quantiles(np.array(vals), quantiles=(0.1, 0.5, 0.9)).values():
        assert 0.0 <= v < HOURS


@given(st.lists(clock, min_size=2, max_size=80))
def test_quantiles_are_ordered_around_the_circular_mean(vals):
    """Ordering on a circle only means anything relative to an origin.

    Quantiles are taken in the mean-centred frame, so that is the frame the
    ordering holds in - unconditionally, for any sample. Asserting it on the
    wrapped values instead needs the sample to be concentrated, which made the
    test depend on a threshold and go intermittently red.
    """
    arr = np.array(vals)
    centre = circular_mean(arr)
    q = circular_quantiles(arr, quantiles=(0.1, 0.5, 0.9))
    offsets = [float(circular_difference(q[p], centre)) for p in (0.1, 0.5, 0.9)]
    assert offsets == sorted(offsets), f"{offsets} not ordered around {centre}"


@given(
    centre=clock,
    spread=st.floats(min_value=0.01, max_value=2.0),
    n=st.integers(min_value=2, max_value=60),
    rng_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_quantiles_are_ordered_on_the_clock_when_concentrated(centre, spread, n, rng_seed):
    """For a concentrated posterior - the only case the app ever displays -
    the wrapped bounds come out in order on the clock too.

    The cluster is constructed rather than filtered for: `assume(R > 0.9)` on
    uniform draws rejects almost every sample, which trips Hypothesis's
    filter_too_much health check at random and makes the test flaky rather
    than wrong.
    """
    rng = np.random.default_rng(rng_seed)
    arr = wrap_hours(centre + rng.normal(0.0, spread, n))
    assume(resultant_length(arr) > 0.9)
    q = circular_quantiles(arr, quantiles=(0.1, 0.5, 0.9))
    lo = float(circular_difference(q[0.1], q[0.5]))
    hi = float(circular_difference(q[0.9], q[0.5]))
    assert lo <= 1e-9 and hi >= -1e-9


@settings(max_examples=40)
@given(st.floats(min_value=0.2, max_value=3.0), clock)
def test_credible_interval_covers_its_nominal_share(sd, centre):
    """An 80% interval that does not contain 80% of the mass is a lie."""
    rng = np.random.default_rng(0)
    samples = wrap_hours(centre + rng.normal(0.0, sd, 20000))
    q = circular_quantiles(samples, quantiles=(0.1, 0.9))
    inside = np.abs(circular_difference(samples, q[0.1])) + np.abs(
        circular_difference(samples, q[0.9])
    )
    width = abs(float(circular_difference(q[0.9], q[0.1])))
    covered = float(np.mean(inside <= width + 1e-9))
    assert covered == pytest.approx(0.8, abs=0.05)


# --- interval width ---------------------------------------------------------


@given(clock, st.floats(min_value=0.0, max_value=23.999, allow_nan=False))
def test_interval_width_is_the_forward_arc(lo, span):
    from circa.phase.circular import interval_width_hours

    assert interval_width_hours(lo, wrap_hours(lo + span)) == pytest.approx(span, abs=1e-6)


def test_interval_width_grows_monotonically_with_dispersion():
    """A posterior that knows less must never report a narrower interval.

    `abs(circular_difference(hi, lo))` folds at 12 h, so widths climbed to ~11 h
    and then came back *down* as the posterior spread further - a uniform
    posterior, carrying no information whatsoever, reported a 4.8 h credible
    interval and scored as moderately confident.
    """
    from circa.phase.circular import interval_width_hours
    from circa.phase.confidence import assess

    rng = np.random.default_rng(2)
    widths, qs = [], []
    for sd in (1, 2, 3, 4, 5, 6, 8, 10, 15, 30):
        s = wrap_hours(rng.normal(21.0, sd, 40000))
        q = circular_quantiles(s, np.ones_like(s), quantiles=(0.1, 0.9))
        w = interval_width_hours(q[0.1], q[0.9]) * 60
        widths.append(w)
        qs.append(assess(30, w, 0.95, 10).q)

    assert widths == sorted(widths), f"width not monotone in dispersion: {widths}"
    assert qs == sorted(qs, reverse=True), f"confidence not monotone: {qs}"
    # An 80% interval over a uniform posterior covers 80% of the day.
    assert widths[-1] == pytest.approx(0.8 * 24 * 60, rel=0.05)


def test_uninformative_posterior_cannot_score_as_confident():
    from circa.phase.circular import interval_width_hours
    from circa.phase.confidence import assess

    rng = np.random.default_rng(3)
    s = rng.uniform(0, 24, 40000)
    q = circular_quantiles(s, np.ones_like(s), quantiles=(0.1, 0.9))
    width = interval_width_hours(q[0.1], q[0.9]) * 60
    conf = assess(n_nights=60, ci80_width_minutes=width, data_coverage=1.0,
                  channel_agreement_minutes=0.0)
    assert conf.tier == 0
    assert conf.q < 0.01

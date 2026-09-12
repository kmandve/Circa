"""Circular statistics for phase on a 24-hour clock.

Phase is an angle, not a number. Averaging 23:30 and 00:30 arithmetically gives
12:00 — the exact opposite of the right answer. Every aggregation in the phase
layer goes through this module so that mistake cannot be made locally.

Angles are handled internally in radians; the public surface speaks hours in
[0, 24).
"""

from __future__ import annotations

import numpy as np

HOURS = 24.0
TWO_PI = 2.0 * np.pi


def hours_to_radians(hours: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(hours) * TWO_PI / HOURS


# np.mod of a tiny negative number returns a value just under 24, which then
# rounds to exactly 24.0 - outside the [0, 24) contract, and displayed as
# "24:00". Snap anything within this tolerance of a full turn back to zero.
_WRAP_EPS = 1e-9


def _snap(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values - HOURS) < _WRAP_EPS, 0.0, values)


def radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    wrapped = np.mod(np.asarray(radians, dtype=float) * HOURS / TWO_PI, HOURS)
    result = _snap(wrapped)
    return float(result) if np.isscalar(radians) or result.ndim == 0 else result


def wrap_hours(hours: np.ndarray | float) -> np.ndarray | float:
    """Map to [0, 24), never to 24.0."""
    wrapped = np.mod(np.asarray(hours, dtype=float), HOURS)
    result = _snap(wrapped)
    return float(result) if np.isscalar(hours) or result.ndim == 0 else result


def circular_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Signed shortest difference a - b, in (-12, +12] hours.

    This is what "the model was 40 minutes late" has to mean when the truth is
    23:50 and the estimate is 00:30.
    """
    diff = np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), HOURS)
    return np.where(diff > HOURS / 2, diff - HOURS, diff)


def circular_mean(hours: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted circular mean, in hours."""
    hours = np.asarray(hours, dtype=float)
    if hours.size == 0:
        raise ValueError("circular_mean of an empty array")
    if weights is None:
        weights = np.ones_like(hours)
    weights = np.asarray(weights, dtype=float)
    angles = hours_to_radians(hours)
    vector = np.sum(weights * np.exp(1j * angles))
    return float(radians_to_hours(np.angle(vector)))


def resultant_length(hours: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean resultant length R in [0, 1]: 1 = perfectly concentrated, 0 = uniform."""
    hours = np.asarray(hours, dtype=float)
    if hours.size == 0:
        return 0.0
    if weights is None:
        weights = np.ones_like(hours)
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        return 0.0
    vector = np.sum(weights * np.exp(1j * hours_to_radians(hours))) / total
    return float(np.abs(vector))


def circular_sd_hours(hours: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Circular standard deviation in hours.

    sqrt(-2 ln R) is the standard definition; it diverges as R -> 0, so a fully
    dispersed sample is reported as 6 h (the SD of a uniform distribution on a
    circle) rather than infinity.
    """
    r = resultant_length(hours, weights)
    if r <= 1e-9:
        return HOURS / 4
    sd_rad = np.sqrt(-2.0 * np.log(min(r, 1.0)))
    return float(min(sd_rad * HOURS / TWO_PI, HOURS / 4))


def kappa_from_resultant(r: float) -> float:
    """Estimate von Mises concentration kappa from the resultant length.

    Uses the standard Fisher (1993) piecewise approximation. kappa is the
    natural way to express "how confident is this channel" when combining
    observations, since von Mises densities multiply cleanly.
    """
    r = float(np.clip(r, 1e-9, 1 - 1e-9))
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1.0 / (r**3 - 4 * r**2 + 3 * r)


def kappa_from_sd_hours(sd_hours: float) -> float:
    """Inverse of `sd_hours_from_kappa`, for expressing a prior as a spread."""
    sd_hours = max(float(sd_hours), 1e-6)
    sd_rad = sd_hours * TWO_PI / HOURS
    r = float(np.exp(-0.5 * sd_rad**2))
    return kappa_from_resultant(r)


def sd_hours_from_kappa(kappa: float) -> float:
    """Approximate circular SD in hours for a given concentration."""
    from scipy.special import i0, i1

    kappa = max(float(kappa), 1e-9)
    r = float(i1(kappa) / i0(kappa)) if kappa < 500 else 1.0 - 1.0 / (2 * kappa)
    if r >= 1 - 1e-12:
        return 0.0
    return float(np.sqrt(-2.0 * np.log(r)) * HOURS / TWO_PI)


def von_mises_logpdf(x_hours: np.ndarray, mu_hours: float, kappa: float) -> np.ndarray:
    """log von Mises density, evaluated in hours.

    Used as the observation likelihood in the particle filter. Returned
    unnormalised in kappa-independent terms is *not* safe here, because
    different channels carry different kappa and the normaliser matters when
    they are summed — so the Bessel term is kept.
    """
    from scipy.special import i0e

    kappa = max(float(kappa), 1e-9)
    delta = hours_to_radians(circular_difference(x_hours, mu_hours))
    # i0e is exp(-kappa) * i0(kappa), which keeps large kappa from overflowing.
    return kappa * (np.cos(delta) - 1.0) - np.log(TWO_PI * i0e(kappa))


def combine_von_mises(observations: list[tuple[float, float]]) -> tuple[float, float]:
    """Multiply von Mises densities: returns the combined (mu_hours, kappa).

    Each observation is (mu_hours, kappa). The product of von Mises densities
    is itself von Mises, with parameters given by the vector sum — which is why
    the phase channels can be fused this cheaply when dynamics are ignored.
    """
    if not observations:
        raise ValueError("no observations to combine")
    vector = sum(
        kappa * np.exp(1j * hours_to_radians(mu)) for mu, kappa in observations
    )
    kappa = float(np.abs(vector))
    mu = float(radians_to_hours(np.angle(vector)))
    return mu, kappa


def interval_width_hours(lower_hours: float, upper_hours: float) -> float:
    """Width of a directed circular interval [lower -> upper], in [0, 24).

    Must not be written as `abs(circular_difference(upper, lower))`. That folds
    at 12 h, so it *shrinks* as the interval grows past half a day: a posterior
    spread over the whole circle - one that knows nothing at all - reports a
    ~4.8 h credible interval and scores as moderately confident. Credible
    intervals produced by `circular_quantiles` are directed (lower quantile
    first), so the forward arc is the honest width and grows monotonically to a
    full day.
    """
    return float(np.mod(float(upper_hours) - float(lower_hours), HOURS))


def circular_quantiles(
    samples_hours: np.ndarray,
    weights: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, float]:
    """Credible interval bounds for a circular sample.

    Quantiles are undefined on a circle without an origin, so the sample is
    first rotated so its circular mean sits at 12:00, quantiles are taken
    linearly there, then rotated back. That is well behaved as long as the
    posterior is unimodal and not spread over most of the circle — which is
    exactly when a credible interval is meaningful anyway.
    """
    samples = np.asarray(samples_hours, dtype=float)
    if samples.size == 0:
        raise ValueError("no samples")
    if weights is None:
        weights = np.ones_like(samples)
    weights = np.asarray(weights, dtype=float)

    centre = circular_mean(samples, weights)
    centred = circular_difference(samples, centre)  # (-12, 12]

    order = np.argsort(centred)
    sorted_vals = centred[order]
    sorted_w = weights[order]
    cumulative = np.cumsum(sorted_w) / np.sum(sorted_w)

    out: dict[float, float] = {}
    for q in quantiles:
        idx = int(np.searchsorted(cumulative, q, side="left"))
        idx = min(idx, sorted_vals.size - 1)
        out[q] = float(wrap_hours(centre + sorted_vals[idx]))
    return out

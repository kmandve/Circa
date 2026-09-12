"""Bootstrap particle filter over the circadian oscillator.

Why a particle filter rather than the UKF the original plan called for: the
light proxy already requires sampling an ensemble of plausible light histories,
so the two unify for free. It also avoids forcing a Gaussian onto a circular,
potentially multimodal state — which matters exactly when the estimate is hard,
such as after travel or a disrupted night.

**Fixed-window re-estimation, not persistent online filtering.** Each run
re-derives the posterior from scratch over the trailing window rather than
carrying particles between invocations. The whole window costs a couple of
seconds, and it buys three things worth more than the saving: results are
reproducible, there is no serialised state to corrupt or migrate, and
re-running after a parser fix retroactively corrects history.

Each particle carries an oscillator state plus its own intrinsic period tau and
its own channel offsets, so those are inferred rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import structlog

from circa.phase.circular import (
    circular_difference,
    circular_mean,
    circular_quantiles,
    resultant_length,
    von_mises_logpdf,
    wrap_hours,
)
from circa.phase.light_proxy import LightEnsemble
from circa.phase.oscillator import equilibrate, integrate

log = structlog.get_logger(__name__)

DT_HOURS = 0.1

# Resample when the effective sample size falls below this fraction of N.
ESS_THRESHOLD = 0.5
# Roughening after resampling: without it, repeated resampling collapses the
# particle set onto a handful of distinct values and the posterior looks far
# more confident than the evidence warrants.
ROUGHEN_STATE_SD = 0.01
ROUGHEN_TAU_SD = 0.01

# Prior spread on intrinsic period. Human tau is tightly clustered near 24.2 h.
TAU_PRIOR_SD = 0.18
TAU_BOUNDS = (23.3, 25.2)

# Daily phase process noise. Represents zeitgebers the model cannot see - meals,
# social timing, stress, temperature. Without it the oscillator is an attractor:
# every particle entrains to the same phase under a shared light schedule, the
# spread collapses, and the filter reports minutes of uncertainty while being
# an hour wrong.
PHASE_PROCESS_NOISE_HOURS = 0.12

# Per-person offset between the model's phase and what a channel reports, over
# and above the population prior each channel already applies. These are the
# systematic term that more nights cannot remove: with only behavioural
# observations, phase and offset are not separately identifiable, so the offset
# prior is what stops the filter claiming false precision.
DELTA_PRIOR_SD_HOURS = {"sleep": 1.0, "hr": 1.4, "probe": 1.2}
DEFAULT_DELTA_PRIOR_SD = 1.2
# The offset is a stable personal trait, so it drifts only slowly.
DELTA_ROUGHEN_SD = 0.05

# Model-adequacy standard deviation: how far a *perfectly fitted* oscillator
# still sits from a real person's measured DLMO.
#
# This is the error floor the literature describes, and it is structural, not
# statistical. Huang et al. 2021 report ~0.6 h MAE for activity-driven models
# against laboratory DLMO in regular day workers, rising to 2.5-2.8 h under
# shift work; other wearable approaches land in 0.4-1.1 h. No number of extra
# nights removes it, because it comes from the mapping between an oscillator
# and human physiology, not from sampling noise.
#
# The filter's internal spread reflects only what the model believes about
# itself, which after entrainment is often 15-25 minutes. Reporting that
# directly would be false precision. The posterior handed to the calendar is
# therefore convolved with this term.
MODEL_ADEQUACY_SD_HOURS = 0.55


@dataclass
class ChannelObservation:
    """One channel's view of DLMO on one day, as a von Mises."""

    day: datetime
    channel: str
    mu_hours: float
    kappa: float


@dataclass
class PosteriorResult:
    dlmo_hours: float
    dlmo_ci80: tuple[float, float]
    dlmo_ci95: tuple[float, float]
    cbtmin_hours: float
    sd_minutes: float
    samples_hours: np.ndarray
    weights: np.ndarray
    tau_mean: float
    tau_sd: float
    amplitude_mean: float
    ess: float
    n_updates: int
    channel_agreement_minutes: float | None
    detail: dict = field(default_factory=dict)


# Channels that are independent enough for their disagreement to be evidence.
_CORROBORATING_CHANNELS = frozenset({"sleep", "hr"})


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: lower variance than multinomial, O(N)."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).clip(0, n - 1)


def _circular_sd(hours: np.ndarray, weights: np.ndarray) -> float:
    from circa.phase.circular import circular_sd_hours

    return circular_sd_hours(hours, weights)


def _rotate_phase(state: np.ndarray, delta_hours: np.ndarray) -> np.ndarray:
    """Advance/retard each particle's oscillator phase by a small angle.

    Phase is the argument of z = x + i(-xc), so a phase perturbation is a
    rotation of that complex number. Jittering x and xc independently would
    perturb amplitude as well and is not the same thing.
    """
    theta = delta_hours * (2.0 * np.pi / 24.0)
    x, xc = state[0], state[1]
    z = x + 1j * (-xc)
    z = z * np.exp(1j * theta)
    out = state.copy()
    out[0] = np.real(z)
    out[1] = -np.imag(z)
    return out


def _local_hours(ts: datetime, offset_seconds: int) -> float:
    local = ts + timedelta(seconds=offset_seconds)
    return local.hour + local.minute / 60 + local.second / 3600


def channel_agreement_minutes(
    observations: list[ChannelObservation],
) -> float | None:
    """Mean pairwise disagreement between independent channels, in minutes.

    Agreement is a property of *channels*, so each channel collapses to one
    angle before the comparison. This matters because the sleep channel is
    applied once per filter day: comparing raw observations would pit it
    against twenty-one identical copies of itself, and a mean spread near zero
    reads as perfect corroboration at exactly the moment there is nothing to
    corroborate.

    Returns None when fewer than two channels reported. That is not agreement
    and not disagreement, and `confidence.assess` scores it separately.
    """
    by_channel: dict[str, list[float]] = {}
    for obs in observations:
        if obs.channel in _CORROBORATING_CHANNELS:
            by_channel.setdefault(obs.channel, []).append(obs.mu_hours)
    if len(by_channel) < 2:
        return None
    mus = [circular_mean(np.asarray(v)) for v in by_channel.values()]
    spread = [
        abs(float(circular_difference(a, b))) * 60
        for i, a in enumerate(mus)
        for b in mus[i + 1 :]
    ]
    return float(np.mean(spread)) if spread else None


def run(
    ensemble: LightEnsemble,
    observations: list[ChannelObservation],
    n_particles: int,
    tau_prior_mean: float,
    utc_offset_seconds: int,
    seed: int | None = None,
) -> PosteriorResult:
    """Filter over the light ensemble, updating on each day's observations."""
    rng = np.random.default_rng(seed)
    n_groups = ensemble.light.shape[0]
    if n_particles < n_groups:
        n_particles = n_groups
    # Particles are split across light histories; every history gets an equal
    # share, so the ensemble's spread propagates into the posterior.
    group_of = np.arange(n_particles) % n_groups

    start_ts = ensemble.times[0]
    total_hours = ensemble.hours[-1] if len(ensemble.hours) > 1 else 24.0

    # --- initialise --------------------------------------------------------
    tau = np.clip(
        rng.normal(tau_prior_mean, TAU_PRIOR_SD, n_particles), *TAU_BOUNDS
    )
    seed_hours = np.arange(0, 24, DT_HOURS)
    seed_light = np.interp(seed_hours, ensemble.hours, ensemble.light.mean(axis=0))
    seed_wake = np.interp(seed_hours, ensemble.hours, ensemble.wake)
    base_state = equilibrate(seed_light, seed_wake, seed_hours, taux=tau_prior_mean)
    state = np.tile(np.asarray(base_state).reshape(3, 1), (1, n_particles))
    # Spread the starting phase: with no history the initial phase is barely
    # known, and starting every particle identically would fake certainty.
    state += rng.normal(0.0, 0.25, state.shape)

    # Per-particle channel offsets, sampled from their priors and then learned.
    channels = sorted({o.channel for o in observations}) or ["sleep"]
    delta = {
        ch: rng.normal(
            0.0, DELTA_PRIOR_SD_HOURS.get(ch, DEFAULT_DELTA_PRIOR_SD), n_particles
        )
        for ch in channels
    }

    log_weights = np.zeros(n_particles)
    by_day: dict[int, list[ChannelObservation]] = {}
    for obs in observations:
        day_index = int((obs.day - start_ts).total_seconds() // 86400)
        by_day.setdefault(day_index, []).append(obs)

    n_days = max(int(np.ceil(total_hours / 24.0)), 1)
    n_updates = 0
    ess = float(n_particles)

    last_result = None
    for day in range(n_days):
        t0, t1 = day * 24.0, min((day + 1) * 24.0, total_hours)
        if t1 - t0 < 1.0:
            break
        hours = np.arange(t0, t1, DT_HOURS)
        if hours.size < 2:
            continue

        # --- propagate: each light history advances its own particle group ---
        new_state = np.empty_like(state)
        dlmo_abs = np.empty(n_particles)
        cbt_abs = np.empty(n_particles)
        amplitude = np.empty(n_particles)

        for g in range(n_groups):
            members = np.flatnonzero(group_of == g)
            if members.size == 0:
                continue
            light = np.interp(hours, ensemble.hours, ensemble.light[g])
            wake = np.interp(hours, ensemble.hours, ensemble.wake)
            result = integrate(
                state[:, members], light, wake, hours, taux=tau[members]
            )
            new_state[:, members] = result.states[-1]
            dlmo_abs[members] = result.dlmo_hours
            cbt_abs[members] = result.cbtmin_hours
            amplitude[members] = result.amplitude

        # Process noise: unmodelled zeitgebers nudge the clock every day.
        state = _rotate_phase(
            new_state, rng.normal(0.0, PHASE_PROCESS_NOISE_HOURS, n_particles)
        )

        # --- update on this day's observations -------------------------------
        day_obs = by_day.get(day, [])
        if day_obs:
            # Convert each particle's predicted DLMO into local clock hours so
            # it is directly comparable with the channel observations.
            predicted = wrap_hours(
                _local_hours(start_ts, utc_offset_seconds) + dlmo_abs
            )
            for obs in day_obs:
                # The channel reports phase plus this person's own offset, so
                # the comparison is against (model phase + delta), not phase.
                offset = delta.get(obs.channel)
                shifted = predicted if offset is None else wrap_hours(predicted + offset)
                log_weights += von_mises_logpdf(shifted, obs.mu_hours, obs.kappa)
                n_updates += 1

            log_weights -= log_weights.max()
            weights = np.exp(log_weights)
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                log.warning("particle_filter.degenerate_weights", day=day)
                log_weights = np.zeros(n_particles)
                weights = np.full(n_particles, 1.0 / n_particles)
            else:
                weights /= total
            ess = float(1.0 / np.sum(weights**2))

            if ess < ESS_THRESHOLD * n_particles:
                idx = _systematic_resample(weights, rng)
                state = state[:, idx]
                tau = tau[idx]
                dlmo_abs = dlmo_abs[idx]
                cbt_abs = cbt_abs[idx]
                amplitude = amplitude[idx]
                # Roughen, or the set collapses to a few duplicated particles.
                state += rng.normal(0.0, ROUGHEN_STATE_SD, state.shape)
                tau = np.clip(tau + rng.normal(0.0, ROUGHEN_TAU_SD, n_particles), *TAU_BOUNDS)
                for ch in delta:
                    delta[ch] = delta[ch][idx] + rng.normal(
                        0.0, DELTA_ROUGHEN_SD, n_particles
                    )
                log_weights = np.zeros(n_particles)

        last_result = (dlmo_abs, cbt_abs, amplitude)

    if last_result is None:
        raise RuntimeError("particle filter produced no propagation steps")

    dlmo_abs, cbt_abs, amplitude = last_result
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    offset_hours = _local_hours(start_ts, utc_offset_seconds)
    dlmo_local = wrap_hours(offset_hours + dlmo_abs)
    cbt_local = wrap_hours(offset_hours + cbt_abs)

    filter_sd_hours = _circular_sd(dlmo_local, weights)

    # Convolve the filter posterior with the model-adequacy distribution before
    # reporting. Everything downstream - credible intervals, block widths,
    # confidence tiers - reads the inflated version, because that is the honest
    # statement of what is known about the person rather than about the model.
    reported = wrap_hours(
        dlmo_local + rng.normal(0.0, MODEL_ADEQUACY_SD_HOURS, dlmo_local.shape)
    )

    mean_hours = circular_mean(reported, weights)
    quantiles = circular_quantiles(
        reported, weights, quantiles=(0.025, 0.1, 0.9, 0.975)
    )
    r = resultant_length(reported, weights)
    sd_hours = _circular_sd(reported, weights)

    agreement = channel_agreement_minutes(observations)

    return PosteriorResult(
        dlmo_hours=float(mean_hours),
        dlmo_ci80=(quantiles[0.1], quantiles[0.9]),
        dlmo_ci95=(quantiles[0.025], quantiles[0.975]),
        cbtmin_hours=float(circular_mean(cbt_local, weights)),
        sd_minutes=float(sd_hours * 60),
        samples_hours=reported,
        weights=weights,
        tau_mean=float(np.sum(weights * tau)),
        tau_sd=float(np.sqrt(np.sum(weights * (tau - np.sum(weights * tau)) ** 2))),
        amplitude_mean=float(np.sum(weights * amplitude)),
        ess=ess,
        n_updates=n_updates,
        channel_agreement_minutes=agreement,
        detail={
            "n_particles": n_particles,
            "n_light_histories": n_groups,
            "resultant_length": round(r, 4),
            "days_propagated": n_days,
            "filter_sd_minutes": round(filter_sd_hours * 60, 1),
            "model_adequacy_sd_minutes": round(MODEL_ADEQUACY_SD_HOURS * 60, 1),
            # Learned personal offsets. A large |mean| means this channel reads
            # systematically early or late for this person versus the
            # population prior - which is exactly the term a one-off DLMO test
            # would pin down.
            "channel_offsets_hours": {
                ch: {
                    "mean": round(float(np.sum(weights * v)), 3),
                    "sd": round(
                        float(np.sqrt(np.sum(weights * (v - np.sum(weights * v)) ** 2))), 3
                    ),
                }
                for ch, v in delta.items()
            },
        },
    )

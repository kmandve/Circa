"""Predicted alertness = circadian drive - sleep pressure - sleep inertia.

Clean-room implementation. The Three Process Model of Alertness (Akerstedt &
Folkard 1997) and Jewett & Kronauer's alertness/throughput model are the
reference formulations; their equations are published, so this is written from
them rather than adapted from code. The only maintained open-source
implementation, FIPS, is AGPL-3.0 — whose network clause would reach a hosted
service — so it is deliberately not linked, and is useful only as an offline
oracle to check numerics against. The SAFTE name and architecture are avoided
entirely.

The curve is computed **per particle** and the ensemble gives the confidence
band, so phase uncertainty propagates into alertness uncertainty automatically
rather than being asserted separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import structlog

from circa.alertness import process_c, process_s
from circa.alertness.process_s import SleepWakeHistory

log = structlog.get_logger(__name__)

# Relative weights of the three terms. Circadian drive and homeostatic pressure
# are comparable over a normal day; inertia is large but brief.
#
# W_HOMEOSTATIC trades the depth of the afternoon dip against the size of the
# evening wake-maintenance rise - raising it deepens the dip and flattens the
# evening recovery. 1.1 keeps both visible (dip ~0.5 z, evening rise ~0.4 z),
# which matters because they are the two features the calendar acts on.
W_CIRCADIAN = 1.0
W_HOMEOSTATIC = 1.1
W_INERTIA = 1.0

# Fixed bounds for the absolute energy scale, from the reachable range of the
# three components (C in about [-1, +0.62], S in [0, 1], I in [0, 1]).
#
# `alertness` is z-scored against its own window, which is right for *timing* -
# finding today's peak means comparing today's hours to each other. But it also
# means a day that is uniformly worse looks identical to a day that is
# uniformly better: subtracting the window mean removes exactly the signal that
# says "you slept four hours and today has a lower ceiling". So the raw curve is
# also published on a stable scale that is comparable across days.
_ENERGY_MAX = W_CIRCADIAN * 0.62
_ENERGY_MIN = -W_CIRCADIAN * 1.0 - W_HOMEOSTATIC * 1.0 - W_INERTIA * 1.0


def _to_energy(raw: np.ndarray) -> np.ndarray:
    return np.clip(
        100.0 * (raw - _ENERGY_MIN) / (_ENERGY_MAX - _ENERGY_MIN), 0.0, 100.0
    )


@dataclass(slots=True)
class AlertnessCurve:
    times: np.ndarray          # aware datetimes, UTC
    local_hours: np.ndarray    # local clock hour of each point
    alertness: np.ndarray      # z-scored within this window - for *timing*
    # 0-100 on a fixed scale, so two different days are directly comparable.
    # `alertness` cannot do this: z-scoring subtracts the window mean, which is
    # precisely the quantity that a short night moves.
    energy: np.ndarray
    energy_lower: np.ndarray   # same band as `lower`/`upper`, on the 0-100 scale
    energy_upper: np.ndarray
    lower: np.ndarray          # 10th percentile across particles
    upper: np.ndarray          # 90th percentile across particles
    circadian: np.ndarray
    homeostatic: np.ndarray
    inertia: np.ndarray
    asleep: np.ndarray         # bool mask
    cbtmin_hours: float
    detail: dict


def compute(
    times: np.ndarray,
    utc_offset_seconds: int,
    history: SleepWakeHistory,
    cbtmin_samples_hours: np.ndarray,
    weights: np.ndarray | None = None,
    s_initial: float | None = None,
) -> AlertnessCurve:
    """Build the alertness curve and its uncertainty band.

    `cbtmin_samples_hours` is the posterior sample of CBTmin in local clock
    hours. Each sample yields its own curve; the spread of those curves is the
    band. That is the mechanism by which "we are unsure of your phase" becomes
    "we are unsure when your peak is" rather than a separate hand-set width.
    """
    if len(times) == 0:
        raise ValueError("no time points")

    local_hours = np.array(
        [
            (t + timedelta(seconds=utc_offset_seconds)).hour
            + (t + timedelta(seconds=utc_offset_seconds)).minute / 60.0
            for t in times
        ]
    )

    if s_initial is None:
        s_initial = process_s.equilibrate_initial(history, times)
    s = process_s.simulate(history, times, s_initial=s_initial)
    i_term = process_s.inertia(times, history)
    asleep = np.array([history.is_asleep(t) for t in times])

    samples = np.atleast_1d(np.asarray(cbtmin_samples_hours, dtype=float))
    if weights is None:
        weights = np.ones(len(samples))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    # Subsample: a few hundred distinct phases fully describe the band, and the
    # filter may carry thousands of particles.
    if len(samples) > 400:
        idx = np.random.default_rng(0).choice(len(samples), 400, p=weights, replace=True)
        samples = samples[idx]
        weights = np.ones(len(samples)) / len(samples)

    curves = np.empty((len(samples), len(times)))
    for k, cbtmin in enumerate(samples):
        c = process_c.drive(process_c.hours_since(cbtmin, local_hours))
        curves[k] = W_CIRCADIAN * c - W_HOMEOSTATIC * s - W_INERTIA * i_term

    mean_curve = np.average(curves, axis=0, weights=weights)

    # Z-score against waking hours only. Including sleep would drag the mean
    # down and make ordinary daytime alertness look exceptional.
    awake = ~asleep
    reference = mean_curve[awake] if awake.sum() > 10 else mean_curve
    mu, sigma = float(np.mean(reference)), float(np.std(reference))
    sigma = sigma if sigma > 1e-6 else 1.0

    raw_lower = np.percentile(curves, 10, axis=0)
    raw_upper = np.percentile(curves, 90, axis=0)

    z = (mean_curve - mu) / sigma
    lower = (raw_lower - mu) / sigma
    upper = (raw_upper - mu) / sigma

    mean_cbtmin = float(
        np.mod(
            np.angle(np.sum(weights * np.exp(1j * samples * 2 * np.pi / 24)))
            * 24
            / (2 * np.pi),
            24,
        )
    )

    energy = _to_energy(mean_curve)
    energy_lower = _to_energy(raw_lower)
    energy_upper = _to_energy(raw_upper)

    return AlertnessCurve(
        times=times,
        local_hours=local_hours,
        alertness=z,
        energy=energy,
        energy_lower=energy_lower,
        energy_upper=energy_upper,
        lower=lower,
        upper=upper,
        circadian=process_c.drive(process_c.hours_since(mean_cbtmin, local_hours)),
        homeostatic=s,
        inertia=i_term,
        asleep=asleep,
        cbtmin_hours=mean_cbtmin,
        detail={
            "s_initial": round(float(s_initial), 4),
            "n_phase_samples": len(samples),
            "raw_mean": round(mu, 4),
            "raw_sd": round(sigma, 4),
            "energy_peak": round(float(energy[~asleep].max()) if (~asleep).any() else 0.0, 1),
            "energy_mean_awake": round(
                float(energy[~asleep].mean()) if (~asleep).any() else 0.0, 1
            ),
        },
    )


def grid(start: datetime, end: datetime, minutes: int = 10) -> np.ndarray:
    n = max(int((end - start).total_seconds() // (minutes * 60)), 1)
    return np.array([start + timedelta(minutes=minutes * i) for i in range(n)])

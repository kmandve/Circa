"""Wrapper around the St. Hilaire 2007 circadian oscillator.

Uses `circadian` (Arcascope, MIT) for the dynamics. Hilaire07 is chosen over
Jewett99/Forger99 because it is the only model in the package with a **nonphotic
drive** — it takes wake state as a second input alongside light. That matters
here for two reasons: light is *inferred* rather than measured, so a second
independent input adds real information; and exercise/activity is a
demonstrated nonphotic zeitgeber with its own human phase-response curve.

Two things this module adds on top of the package:

* **Batched integration**, using the `(n_states, n_particles)` convention. The
  package's checker only validates the leading dimension, so a `(3, B)` initial
  condition integrates all B particles at once — ~43 us/particle versus ~7 ms
  solving them one at a time.
* **A vectorised CBTmin finder.** The package's `cbt()` calls `scipy.find_peaks`,
  which is 1-D only and cannot see a batched trajectory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import structlog
from circadian.models import Hilaire07

log = structlog.get_logger(__name__)

N_STATES = 3  # x, xc, n

# Hours from CBTmin back to DLMO. This is the package default and a population
# average; individual variation in it is part of the irreducible error floor.
CBT_TO_DLMO_HOURS = 7.0


@dataclass(slots=True)
class OscillatorResult:
    times: np.ndarray        # hours since start, shape (n_time,)
    states: np.ndarray       # (n_time, 3, n_particles)
    cbtmin_hours: np.ndarray  # (n_particles,) hours-since-start of last CBTmin
    dlmo_hours: np.ndarray   # (n_particles,) hours-since-start of last DLMO
    amplitude: np.ndarray    # (n_particles,) oscillator amplitude


def default_initial_condition(n_particles: int = 1) -> np.ndarray:
    """Package default state, tiled to `(3, n_particles)`."""
    base = np.array([-0.0480751, -1.22504441, 0.51854818])
    if n_particles == 1:
        return base
    return np.tile(base.reshape(N_STATES, 1), (1, n_particles))


def make_model(taux: np.ndarray | float = 24.2, **overrides) -> Hilaire07:
    """Build a model, optionally with a per-particle intrinsic period.

    `taux` may be a scalar or an array of shape `(n_particles,)` — it enters the
    derivative only through a term that broadcasts, so each particle can carry
    its own intrinsic period. That is what lets the filter learn tau rather than
    assume the population mean.
    """
    model = Hilaire07()
    params = dict(model.parameters)
    params["taux"] = taux
    params.update(overrides)
    return Hilaire07(params=params)


# Below this the oscillator has no meaningful phase left: every DLMO derived
# from a collapsed limit cycle is noise wearing a number. A healthy state sits
# near 1.1-1.2.
MIN_EQUILIBRATED_AMPLITUDE = 0.5


def equilibrate(
    light: np.ndarray, wake: np.ndarray, hours: np.ndarray, taux: float = 24.2, loops: int = 25
) -> np.ndarray:
    """Run the schedule repeatedly to settle onto the limit cycle.

    Used for cold start, when there is no previous state to carry forward.
    """
    model = make_model(taux=taux)
    inputs = np.column_stack([light, wake])
    try:
        with warnings.catch_warnings():
            # The package warns whenever its DLMO convergence check has not
            # settled within `loops`. That is expected here rather than
            # diagnostic: a light schedule inferred from step counts is a much
            # weaker zeitgeber than a laboratory protocol, and with taux away
            # from 24 h the oscillator need not reach a fixed point at all. The
            # returned state is still a point on the limit cycle. What actually
            # matters is whether the cycle collapsed, which is checked below -
            # so the vague warning is replaced by a specific one.
            warnings.simplefilter("ignore", UserWarning)
            state = np.asarray(model.equilibrate(hours, inputs, num_loops=loops))
    except Exception as exc:  # noqa: BLE001
        log.warning("oscillator.equilibrate_failed", error=str(exc))
        return default_initial_condition()

    amplitude = float(np.hypot(state[0], state[1]))
    if not np.all(np.isfinite(state)) or amplitude < MIN_EQUILIBRATED_AMPLITUDE:
        log.warning("oscillator.amplitude_collapsed", amplitude=round(amplitude, 4))
        return default_initial_condition()
    return state


def integrate(
    initial_state: np.ndarray,
    light: np.ndarray,
    wake: np.ndarray,
    hours: np.ndarray,
    taux: np.ndarray | float = 24.2,
) -> OscillatorResult:
    """Integrate a batch of particles through one light history.

    `initial_state` is `(3,)` or `(3, n_particles)`; `light` and `wake` are
    shared across the batch.
    """
    state = np.asarray(initial_state, dtype=float)
    if state.ndim == 1:
        state = state.reshape(N_STATES, 1)

    model = make_model(taux=taux)
    inputs = np.column_stack([np.asarray(light, dtype=float), np.asarray(wake, dtype=float)])
    trajectory = model.integrate(hours, initial_condition=state, input=inputs)

    states = np.asarray(trajectory.states)
    if states.ndim == 2:  # single particle
        states = states[:, :, None]

    cbtmin = last_cbtmin(hours, states, phi_ref=float(model.parameters["phi_ref"]))
    x = states[-1, 0, :]
    xc = states[-1, 1, :]
    return OscillatorResult(
        times=hours,
        states=states,
        cbtmin_hours=cbtmin,
        dlmo_hours=cbtmin - CBT_TO_DLMO_HOURS,
        amplitude=np.hypot(x, -xc),
    )


def last_cbtmin(hours: np.ndarray, states: np.ndarray, phi_ref: float = 0.97) -> np.ndarray:
    """Hours-since-start of the most recent core-temperature minimum, per particle.

    The package's `cbt()` uses `scipy.signal.find_peaks`, which cannot operate on
    a batched trajectory, so this searches each particle's `x` for its trough
    within the final cycle and refines it by parabolic interpolation on the
    three surrounding samples — sub-timestep precision without shrinking dt.
    """
    x = states[:, 0, :]  # (n_time, n_particles)
    n_time, n_particles = x.shape

    dt = float(np.mean(np.diff(hours))) if n_time > 1 else 0.1
    window = min(int(round(26.0 / dt)), n_time)
    offset = n_time - window
    segment = x[offset:, :]

    idx = np.argmin(segment, axis=0) + offset
    refined = np.empty(n_particles, dtype=float)

    for p in range(n_particles):
        i = int(idx[p])
        if 0 < i < n_time - 1:
            y0, y1, y2 = x[i - 1, p], x[i, p], x[i + 1, p]
            denom = y0 - 2 * y1 + y2
            # A near-zero denominator means a flat trough; the sample index is
            # already the best available estimate.
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            shift = float(np.clip(shift, -1.0, 1.0))
        else:
            shift = 0.0
        refined[p] = hours[i] + shift * dt

    return refined + phi_ref

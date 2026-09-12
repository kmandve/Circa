"""Process S — homeostatic sleep pressure.

Classic two-process formulation (Borbely 1982; Daan, Beersma & Borbely 1984):
pressure rises saturatingly during wake and decays exponentially during sleep.

This is why phase and alertness are kept in separate modules. Two days with an
identical circadian phase but 8.5 h versus 4.5 h of sleep have nearly identical
C(t) and very different S(t), and therefore very different performance. Folding
them together would turn an estimated energy curve into something that looks
like a physiological measurement.

Driven entirely by *observed* sleep and wake from the Fitbit, so it needs no
assumptions about intent — only about what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.db.models import SleepSession

log = structlog.get_logger(__name__)

# Time constants from Daan, Beersma & Borbely (1984). Wake accumulation is slow
# (~18 h) and sleep recovery is fast (~4 h), which is what makes one good night
# largely restorative while one bad night is not catastrophic.
TAU_WAKE_HOURS = 18.2
TAU_SLEEP_HOURS = 4.2

# S is normalised to [0, 1]: 0 = fully rested, 1 = maximum pressure.
S_UPPER = 1.0
S_LOWER = 0.0

# Sleep inertia: grogginess decays over roughly the first hour of wake. Modelled
# separately from S because it is a distinct transient, not part of the
# homeostat - see Jewett & Kronauer's treatment of the three interacting terms.
INERTIA_INITIAL = 0.55
TAU_INERTIA_HOURS = 0.5

# Where S is assumed to sit when there is no history to integrate from.
COLD_START_S = 0.35


@dataclass(slots=True)
class SleepWakeHistory:
    """Observed sleep episodes, as (start, end) UTC pairs, oldest first."""

    episodes: list[tuple[datetime, datetime]]
    window_start: datetime
    window_end: datetime

    def is_asleep(self, ts: datetime) -> bool:
        return any(start <= ts < end for start, end in self.episodes)

    def last_wake_before(self, ts: datetime) -> datetime | None:
        """When the most recent sleep episode ended, for the inertia term."""
        ends = [end for _, end in self.episodes if end <= ts]
        return max(ends) if ends else None


def load_history(
    session: Session, start: datetime, end: datetime, include_naps: bool = True
) -> SleepWakeHistory:
    stmt = (
        select(SleepSession)
        .where(
            SleepSession.end_ts > start,
            SleepSession.start_ts < end,
            SleepSession.excluded.is_(False),
        )
        .order_by(SleepSession.start_ts)
    )
    episodes = [
        (row.start_ts, row.end_ts)
        for row in session.scalars(stmt)
        if include_naps or row.is_main_sleep
    ]
    return SleepWakeHistory(episodes=episodes, window_start=start, window_end=end)


def simulate(
    history: SleepWakeHistory,
    times: np.ndarray,
    s_initial: float = COLD_START_S,
) -> np.ndarray:
    """Integrate S over `times` (array of aware datetimes).

    Uses the closed-form exponential solution per step rather than a numerical
    integrator: the equations are exactly solvable, so there is no reason to
    accept discretisation error.
    """
    if len(times) == 0:
        return np.array([])

    s = np.empty(len(times))
    current = float(np.clip(s_initial, S_LOWER, S_UPPER))
    s[0] = current

    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]).total_seconds() / 3600.0
        if dt <= 0:
            s[i] = current
            continue
        # Classify the step by its midpoint, so a boundary crossing lands in
        # whichever state occupied most of the interval.
        midpoint = times[i - 1] + (times[i] - times[i - 1]) / 2
        if history.is_asleep(midpoint):
            current = S_LOWER + (current - S_LOWER) * np.exp(-dt / TAU_SLEEP_HOURS)
        else:
            current = S_UPPER - (S_UPPER - current) * np.exp(-dt / TAU_WAKE_HOURS)
        s[i] = current

    return s


def _warmup_grid(start: datetime, end: datetime, minutes: int) -> np.ndarray:
    n = max(int((end - start).total_seconds() // (minutes * 60)), 1)
    return np.array([start + timedelta(minutes=minutes * i) for i in range(n + 1)])


def equilibrate_initial(
    history: SleepWakeHistory,
    times: np.ndarray,
    loops: int = 3,
    step_minutes: int = 10,
) -> float:
    """Sleep pressure at the first forecast point, carried across real history.

    This used to loop `simulate` over `times` alone - the forecast window - and
    feed its own last value back in. That is a fixed point on the *future's*
    schedule, and it discarded everything that happened before the window
    started. A four-hour night and a nine-hour night therefore produced exactly
    the same starting pressure and exactly the same predicted day, which defeats
    the entire purpose of driving the homeostat from recorded sleep.

    Now the arbitrary initial condition is settled by looping only the earliest
    day of history, and the observed schedule is then integrated forward from
    there to the start of the forecast. What you actually slept last night is
    what sets today's pressure.
    """
    if len(times) == 0:
        return COLD_START_S

    start = history.window_start
    if start is None or start >= times[0]:
        # Nothing recorded before the window; the best available answer is a
        # fixed point on the schedule we do have.
        s0 = COLD_START_S
        for _ in range(loops):
            trace = simulate(history, times, s_initial=s0)
            if trace.size == 0:
                break
            s0 = float(trace[-1])
        return s0

    warmup = _warmup_grid(start, times[0], step_minutes)
    per_day = max(int(24 * 60 / step_minutes), 2)
    head = warmup[:per_day] if len(warmup) > per_day else warmup

    s0 = COLD_START_S
    for _ in range(loops):
        trace = simulate(history, head, s_initial=s0)
        if trace.size == 0:
            break
        s0 = float(trace[-1])

    trace = simulate(history, warmup, s_initial=s0)
    return float(trace[-1]) if trace.size else s0


def inertia(times: np.ndarray, history: SleepWakeHistory) -> np.ndarray:
    """Sleep-inertia term I(t) = I0 * exp(-t_awake / tau_I)."""
    out = np.zeros(len(times))
    for i, ts in enumerate(times):
        if history.is_asleep(ts):
            # Inertia is about the transition out of sleep; during sleep the
            # term is irrelevant and is reported as its full value so the
            # alertness curve does not imply usable wakefulness mid-night.
            out[i] = INERTIA_INITIAL
            continue
        woke = history.last_wake_before(ts)
        if woke is None:
            continue
        hours_awake = (ts - woke).total_seconds() / 3600.0
        out[i] = INERTIA_INITIAL * np.exp(-hours_awake / TAU_INERTIA_HOURS)
    return out


def hours_awake(times: np.ndarray, history: SleepWakeHistory) -> np.ndarray:
    out = np.zeros(len(times))
    for i, ts in enumerate(times):
        woke = history.last_wake_before(ts)
        out[i] = (ts - woke).total_seconds() / 3600.0 if woke else np.nan
    return out


@dataclass(slots=True)
class SleepDebt:
    """Cumulative shortfall against the target, over a rolling window."""

    hours: float             # positive = short of target
    nights: int              # nights with usable data
    window_days: int
    last_night_hours: float | None
    last_night_delta: float | None   # vs target; negative = short

    @property
    def is_meaningful(self) -> bool:
        """Debt from one or two nights is noise, not a trend."""
        return self.nights >= 3


def recent_sleep_debt(
    session: Session,
    as_of: datetime,
    target_hours: float,
    days: int = 14,
) -> SleepDebt:
    """Sleep debt over the trailing window.

    Not fed into Process S - the homeostat already integrates the actual sleep
    that happened. This is the human-legible version, and the thing that adjusts
    how much sleep tonight is aimed at.

    Nights are counted once each: a night Fitbit split at a long awakening has
    its segments summed rather than treated as two short nights, which would
    otherwise manufacture debt that was never owed.
    """
    start = as_of - timedelta(days=days)
    rows = list(session.scalars(
        select(SleepSession).where(
            SleepSession.end_ts >= start,
            SleepSession.end_ts <= as_of,
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
        ).order_by(SleepSession.end_ts)
    ))

    by_night: dict[object, float] = {}
    latest_key = None
    for row in rows:
        if row.tst_minutes is None:
            continue
        key = row.sleep_date or row.end_ts.date()
        by_night[key] = by_night.get(key, 0.0) + row.tst_minutes / 60.0
        latest_key = key

    debt = sum(target_hours - slept for slept in by_night.values())
    last = by_night.get(latest_key) if latest_key is not None else None
    return SleepDebt(
        hours=round(debt, 2),
        nights=len(by_night),
        window_days=days,
        last_night_hours=round(last, 2) if last is not None else None,
        last_night_delta=round(last - target_hours, 2) if last is not None else None,
    )


def target_sleep_tonight(
    base_target_hours: float,
    debt: SleepDebt,
    payback_fraction: float,
    max_payback_hours: float,
) -> float:
    """How much sleep tonight should be aimed at, given the debt carried in.

    Debt is repaid in slices: a fortnight of short nights is not recoverable in
    one, and a recommendation that says so is one nobody follows. A surplus is
    not used to shorten the window - sleeping less than target is never the
    advice.
    """
    if not debt.is_meaningful or debt.hours <= 0:
        return base_target_hours
    payback = min(debt.hours * payback_fraction, max_payback_hours)
    return base_target_hours + max(payback, 0.0)


def utc_now() -> datetime:
    return datetime.now(UTC)

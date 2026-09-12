"""Passive behavioural probes — validation without asking the user anything.

The user opted for a fully passive system, so there are no subjective alertness
ratings to validate against. Behaviour, however, is observable, and several
behaviours are genuine probes of sleep propensity and circadian phase. They cost
nothing and they are the closest available substitute for ground truth.

These serve double duty: as extra observation channels for the filter, and as
the backtest target in `backtest.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.db.models import SleepSession
from circa.phase.circular import circular_mean, wrap_hours
from circa.phase.sleep_phase import local_hour_of_day

log = structlog.get_logger(__name__)


@dataclass
class Probe:
    name: str
    day: date
    value: float
    detail: dict


def sleep_latency_probe(session: Session, days: int = 60) -> list[Probe]:
    """Sleep latency against the clock hour at which sleep was attempted.

    The strongest passive signal available. If the model says the circadian gate
    is open at 23:00 and sleep arrives in 8 minutes, the phase estimate is
    consistent. If it takes 45 minutes, the estimate is probably too early.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    probes: list[Probe] = []
    for row in session.scalars(
        select(SleepSession).where(
            SleepSession.start_ts >= since,
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
        )
    ):
        if row.latency_minutes is None or row.sleep_date is None:
            continue
        probes.append(
            Probe(
                name="sleep_latency",
                day=row.sleep_date,
                value=float(row.latency_minutes),
                detail={
                    "attempt_hour": local_hour_of_day(
                        row.start_ts, row.tz_name or "UTC", row.utc_offset_seconds
                    )
                },
            )
        )
    return probes


def free_wake_probe(session: Session, days: int = 60) -> list[Probe]:
    """Wake time on days with no forced wake — the best endogenous phase anchor.

    On an alarm-free day, sleep offset is a biological event rather than a
    scheduling one.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    probes: list[Probe] = []
    for row in session.scalars(
        select(SleepSession).where(
            SleepSession.end_ts >= since,
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
            SleepSession.wake_forced.is_(False),
        )
    ):
        if row.sleep_date is None:
            continue
        probes.append(
            Probe(
                name="free_wake",
                day=row.sleep_date,
                value=local_hour_of_day(
                    row.end_ts, row.tz_name or "UTC", row.utc_offset_seconds
                ),
                detail={"tst_minutes": row.tst_minutes},
            )
        )
    return probes


def waso_probe(session: Session, days: int = 60) -> list[Probe]:
    """Wake-after-sleep-onset as a fraction of the night."""
    since = datetime.now(UTC) - timedelta(days=days)
    probes: list[Probe] = []
    for row in session.scalars(
        select(SleepSession).where(
            SleepSession.end_ts >= since,
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
        )
    ):
        if row.waso_minutes is None or not row.time_in_bed_minutes or row.sleep_date is None:
            continue
        probes.append(
            Probe(
                name="waso_fraction",
                day=row.sleep_date,
                value=float(row.waso_minutes / row.time_in_bed_minutes),
                detail={},
            )
        )
    return probes


def free_wake_phase_estimate(session: Session, days: int = 60) -> float | None:
    """Phase implied by alarm-free wake times alone.

    An independent, behaviour-only estimate to compare the full model against.
    If the model cannot beat this, the extra machinery is not earning its place.
    """
    probes = free_wake_probe(session, days)
    if len(probes) < 3:
        return None
    wake_hours = np.array([p.value for p in probes])
    mean_wake = circular_mean(wake_hours)
    # Wake typically occurs ~2-3 h after CBTmin, and DLMO ~7 h before CBTmin.
    return float(wrap_hours(mean_wake - 2.5 - 7.0))


def latency_consistency(
    probes: list[Probe], predicted_onset_hours: float
) -> dict:
    """Do short latencies cluster near the predicted sleep gate?

    Correlating latency against distance from the predicted onset is a genuine,
    fully passive test of the phase estimate. A negative correlation means
    attempting sleep near the predicted gate really does produce faster sleep
    onset - which is what the model claims.
    """
    if len(probes) < 5:
        return {"n": len(probes), "correlation": None, "note": "insufficient nights"}

    from circa.phase.circular import circular_difference

    distances = np.array(
        [abs(float(circular_difference(p.detail["attempt_hour"], predicted_onset_hours)))
         for p in probes]
    )
    latencies = np.array([p.value for p in probes])
    if np.std(distances) < 1e-6 or np.std(latencies) < 1e-6:
        return {"n": len(probes), "correlation": None, "note": "no variation"}

    correlation = float(np.corrcoef(distances, latencies)[0, 1])
    return {
        "n": len(probes),
        "correlation": round(correlation, 3),
        "mean_latency_minutes": round(float(np.mean(latencies)), 1),
        "note": (
            "positive correlation supports the phase estimate: attempting sleep "
            "further from the predicted gate takes longer"
        ),
    }

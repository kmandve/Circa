"""Sleep-timing phase channel — the baseline every other model must beat.

Sleep midpoint is a strong but imperfect proxy for circadian phase: it is
behaviour, not biology. Published work finds it lands within about an hour of
DLMO in regular sleepers and fails badly under forced or shifted schedules,
which shapes two decisions here:

1. **Alarm-forced wakes are down-weighted.** A wake time set by a meeting says
   more about the calendar than the clock. Free days are the closest thing to an
   endogenous phase probe available passively.
2. **Recency is weighted exponentially.** Phase drifts; a night from five weeks
   ago is evidence, but weaker evidence than last night.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.db.models import SleepSession
from circa.phase.circular import (
    circular_mean,
    kappa_from_resultant,
    resultant_length,
    wrap_hours,
)
from circa.settings_store import RuntimeSettings

log = structlog.get_logger(__name__)

# Halving time for the recency weight. ~10 days means a fortnight-old night
# carries roughly a third the weight of last night - responsive to a real shift
# without letting one odd night dominate.
RECENCY_HALFLIFE_DAYS = 10.0

# A forced wake still carries information (you did fall asleep when you did),
# just less about where the endogenous clock sits.
FORCED_WAKE_WEIGHT = 0.35

# Nights this far from the running mean are treated as regime changes rather
# than noise, and are not allowed to quietly drag the estimate.
OUTLIER_HOURS = 4.0


@dataclass(slots=True)
class SleepPhaseObservation:
    """A von Mises observation of circadian phase from sleep timing."""

    mu_hours: float          # estimated DLMO, local clock hours [0, 24)
    kappa: float             # concentration
    midpoint_hours: float    # the underlying sleep midpoint
    n_nights: int
    n_free_nights: int       # nights not alarm-forced - the strongest evidence
    resultant: float         # 0-1 regularity of sleep timing
    detail: dict


def local_hour_of_day(ts: datetime, tz_name: str, offset_seconds: int | None) -> float:
    """Decimal local clock hour of an instant.

    Uses the offset recorded with the observation when present, so a night spent
    in another timezone is interpreted in the clock the user was actually living
    on rather than retro-fitted to their home zone.
    """
    if offset_seconds is not None:
        local = ts + timedelta(seconds=offset_seconds)
        return local.hour + local.minute / 60 + local.second / 3600
    local = ts.astimezone(ZoneInfo(tz_name))
    return local.hour + local.minute / 60 + local.second / 3600


@dataclass
class Night:
    """One night's sleep, with any split segments merged back together."""

    sleep_date: date
    start_ts: datetime
    end_ts: datetime
    midpoint_ts: datetime | None
    tz_name: str | None
    utc_offset_seconds: int | None
    wake_forced: bool
    tst_minutes: float | None
    n_segments: int


def load_sessions(
    session: Session,
    settings: RuntimeSettings,
    as_of: datetime | None = None,
) -> list[SleepSession]:
    """Raw main-sleep rows, one per API session."""
    as_of = as_of or datetime.now(UTC)
    since = as_of - timedelta(days=settings.sleep_history_days)
    stmt = (
        select(SleepSession)
        .where(
            SleepSession.end_ts >= since,
            SleepSession.end_ts <= as_of,
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
        )
        .order_by(SleepSession.end_ts)
    )
    return list(session.scalars(stmt))


def load_nights(
    session: Session,
    settings: RuntimeSettings,
    as_of: datetime | None = None,
) -> list[Night]:
    """Main sleep, collapsed to one entry per night.

    Fitbit splits a night at a long awakening, and both halves can clear the
    main-sleep threshold. Counting them separately meant one night arrived as
    two: `n_nights` ran ahead of the real evidence and pulled the confidence
    tier up with it, while the sleep channel averaged two fragment midpoints
    instead of one mid-sleep time.

    Segments are merged across the whole night, so the midpoint is the middle
    of first-onset to final-wake - the standard definition - and the forced-wake
    flag comes from the segment that actually ended the night.
    """
    by_date: dict[date, list[SleepSession]] = {}
    for row in load_sessions(session, settings, as_of):
        if row.sleep_date is None:
            continue
        by_date.setdefault(row.sleep_date, []).append(row)

    nights: list[Night] = []
    for sleep_date, rows in sorted(by_date.items()):
        rows.sort(key=lambda r: r.start_ts)
        start, end = rows[0].start_ts, max(r.end_ts for r in rows)
        last = max(rows, key=lambda r: r.end_ts)
        tst = [r.tst_minutes for r in rows if r.tst_minutes is not None]
        nights.append(
            Night(
                sleep_date=sleep_date,
                start_ts=start,
                end_ts=end,
                midpoint_ts=(
                    rows[0].midpoint_ts
                    if len(rows) == 1
                    else start + (end - start) / 2
                ),
                tz_name=last.tz_name,
                utc_offset_seconds=last.utc_offset_seconds,
                wake_forced=bool(last.wake_forced),
                tst_minutes=sum(tst) if tst else None,
                n_segments=len(rows),
            )
        )
    return nights


def estimate(
    session: Session,
    settings: RuntimeSettings,
    as_of: datetime | None = None,
) -> SleepPhaseObservation | None:
    """Estimate circadian phase from recent sleep timing.

    Returns None when there is no usable history at all; the caller then falls
    back to the chronotype prior alone.
    """
    as_of = as_of or datetime.now(UTC)
    nights = load_nights(session, settings, as_of)
    if not nights:
        return None

    midpoints: list[float] = []
    weights: list[float] = []
    forced: list[bool] = []

    for night in nights:
        if night.midpoint_ts is None:
            continue
        midpoint = local_hour_of_day(
            night.midpoint_ts, night.tz_name or "UTC", night.utc_offset_seconds
        )
        age_days = (as_of - night.end_ts).total_seconds() / 86400
        weight = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
        is_forced = bool(night.wake_forced)
        if is_forced:
            weight *= FORCED_WAKE_WEIGHT
        midpoints.append(midpoint)
        weights.append(weight)
        forced.append(is_forced)

    if not midpoints:
        return None

    mid_arr = np.array(midpoints)
    w_arr = np.array(weights)

    # Down-weight (never delete) nights far from the provisional centre. An
    # all-nighter is real data about a real event, but it should not be allowed
    # to silently relocate the baseline.
    provisional = circular_mean(mid_arr, w_arr)
    from circa.phase.circular import circular_difference

    deviation = np.abs(circular_difference(mid_arr, provisional))
    outlier_scale = np.where(deviation > OUTLIER_HOURS, 0.25, 1.0)
    w_arr = w_arr * outlier_scale

    midpoint_hours = circular_mean(mid_arr, w_arr)
    r = resultant_length(mid_arr, w_arr)

    # Effective sample size for exponentially-weighted data. Kappa from the
    # resultant alone would claim high confidence from three near-identical
    # nights, so it is scaled by how much independent evidence there really is.
    n_eff = float(np.sum(w_arr) ** 2 / np.sum(w_arr**2)) if np.sum(w_arr**2) > 0 else 1.0
    kappa_raw = kappa_from_resultant(r)
    kappa = kappa_raw * min(n_eff / 14.0, 1.0)

    # Sleep midpoint is not DLMO. The offset is a chronotype-dependent
    # population prior until enough personal data exists to move it.
    offset = settings.prior["dlmo_offset_hours"]
    dlmo_hours = wrap_hours(midpoint_hours - offset)

    # The prior offset is itself uncertain, and that uncertainty is part of the
    # phase estimate - not something to quietly drop.
    prior_sd = settings.prior["prior_sd_hours"]
    from circa.phase.circular import kappa_from_sd_hours

    kappa = 1.0 / (1.0 / max(kappa, 1e-6) + 1.0 / kappa_from_sd_hours(prior_sd))

    n_free = sum(1 for f in forced if not f)
    return SleepPhaseObservation(
        mu_hours=float(dlmo_hours),
        kappa=float(max(kappa, 1e-3)),
        midpoint_hours=float(midpoint_hours),
        n_nights=len(midpoints),
        n_free_nights=n_free,
        resultant=float(r),
        detail={
            "n_effective": round(n_eff, 2),
            "dlmo_offset_prior_hours": offset,
            "prior_sd_hours": prior_sd,
            "kappa_before_prior": round(kappa_raw, 3),
            "outliers_downweighted": int(np.sum(outlier_scale < 1.0)),
            "forced_wake_nights": len(forced) - n_free,
        },
    )


def nightly_midpoints(
    session: Session, settings: RuntimeSettings, as_of: datetime | None = None
) -> list[tuple[date, float, bool]]:
    """(sleep_date, local midpoint hour, was_forced) — for charts and backtests."""
    out = []
    for night in load_nights(session, settings, as_of):
        if night.midpoint_ts is None or night.sleep_date is None:
            continue
        out.append(
            (
                night.sleep_date,
                local_hour_of_day(
                    night.midpoint_ts, night.tz_name or "UTC", night.utc_offset_seconds
                ),
                bool(night.wake_forced),
            )
        )
    return out

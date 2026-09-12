"""Orchestrates one full phase estimation.

Sequence: gather each channel's observation, sample the light ensemble, run the
filter, assess confidence, persist. Everything downstream (alertness, calendar,
web) reads the persisted `PhaseEstimate` rather than recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import (
    DailyMetric,
    HeartRateMinute,
    PhaseEstimate,
    PhaseObservation,
    SleepSession,
    StepMinute,
)
from circa.phase import confidence as confidence_mod
from circa.phase import hr_phase, light_proxy, sleep_phase
from circa.phase.circular import circular_mean, interval_width_hours, wrap_hours
from circa.phase.particle_filter import ChannelObservation, PosteriorResult
from circa.phase.particle_filter import run as run_filter
from circa.settings_store import RuntimeSettings, load_settings

log = structlog.get_logger(__name__)

MODEL_VERSION = "0.2.0"

# How much history the filter integrates over. Long enough for the oscillator to
# entrain and for tau to be identifiable, short enough to stay a couple of
# seconds on a shared-core VM.
FILTER_WINDOW_DAYS = 21


@dataclass
class PhaseResult:
    target_date: date
    dlmo_local_hours: float
    dlmo_ts: datetime
    cbtmin_local_hours: float
    cbtmin_ts: datetime
    ci80: tuple[datetime, datetime]
    ci80_local_hours: tuple[float, float]
    sd_minutes: float
    confidence: confidence_mod.Confidence
    posterior: PosteriorResult
    observations: list[ChannelObservation]
    tz_name: str
    utc_offset_seconds: int


def _utc_offset(tz_name: str, at: datetime) -> int:
    return int(at.astimezone(ZoneInfo(tz_name)).utcoffset().total_seconds())


def _local_hour_after(hours: float, after: datetime, offset_seconds: int) -> datetime:
    """First instant at local clock hour `hours` strictly after `after`."""
    candidate = _local_hours_to_ts(hours, after, offset_seconds)
    while candidate <= after:
        candidate += timedelta(days=1)
    while candidate - timedelta(days=1) > after:
        candidate -= timedelta(days=1)
    return candidate


def _local_hours_to_ts(hours: float, reference: datetime, offset_seconds: int) -> datetime:
    """Turn a local clock hour into the nearest actual instant around `reference`."""
    local_ref = reference + timedelta(seconds=offset_seconds)
    midnight = local_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    candidate = midnight + timedelta(hours=hours) - timedelta(seconds=offset_seconds)
    # Pick whichever daily repetition is closest to the reference instant.
    best = min(
        (candidate + timedelta(days=d) for d in (-1, 0, 1)),
        key=lambda c: abs((c - reference).total_seconds()),
    )
    return best


def _data_coverage(session: Session, as_of: datetime, days: int = 3) -> float:
    start = as_of - timedelta(days=days)
    expected = days * 24 * 60
    got = session.scalar(
        select(func.count())
        .select_from(HeartRateMinute)
        .where(HeartRateMinute.ts >= start, HeartRateMinute.ts < as_of)
    ) or 0
    return float(min(got / expected, 1.0))


def _illness_flag(session: Session, as_of: datetime) -> bool:
    """Elevated resting HR plus elevated nightly temperature vs baseline.

    Deliberately conservative: one metric alone is noisy, both together is the
    pattern that should widen the posterior rather than be treated as a real
    phase change.
    """
    today = as_of.date()
    temp = session.scalar(
        select(DailyMetric).where(
            DailyMetric.metric == "daily-sleep-temperature-derivations",
            DailyMetric.metric_date >= today - timedelta(days=1),
        )
    )
    temp_elevated = False
    if temp is not None and temp.extra:
        baseline = temp.extra.get("baselineTemperatureCelsius")
        if baseline and temp.value:
            temp_elevated = (temp.value - float(baseline)) > 0.6

    rhr_rows = list(
        session.scalars(
            select(DailyMetric)
            .where(
                DailyMetric.metric == "daily-resting-heart-rate",
                DailyMetric.metric_date >= today - timedelta(days=30),
            )
            .order_by(DailyMetric.metric_date)
        )
    )
    rhr_elevated = False
    if len(rhr_rows) >= 7:
        values = [r.value for r in rhr_rows if r.value is not None]
        if len(values) >= 7:
            baseline = float(np.median(values[:-1]))
            rhr_elevated = (values[-1] - baseline) > 5.0

    return bool(temp_elevated and rhr_elevated)


def _data_seed(session: Session, as_of: datetime, settings: RuntimeSettings) -> int:
    """A seed determined entirely by the data and configuration.

    The particle filter is stochastic. Left unseeded it returns a slightly
    different answer every run, which is bad for two reasons: results are not
    reproducible (the whole justification for re-estimating over a fixed window
    rather than persisting filter state), and the calendar descriptions change
    on every poll, so every event gets patched and idempotency is defeated.

    Deriving the seed from the data means identical inputs give an identical
    answer, while genuinely new data produces a genuinely new draw.
    """
    from circa.db.models import HeartRateMinute

    latest_hr = session.scalar(select(func.max(HeartRateMinute.ts)))
    latest_sleep = session.scalar(select(func.max(SleepSession.end_ts)))
    n_sleep = session.scalar(select(func.count()).select_from(SleepSession)) or 0
    n_hr = session.scalar(select(func.count()).select_from(HeartRateMinute)) or 0

    material = "|".join(
        str(x)
        for x in (
            MODEL_VERSION,
            (as_of + timedelta(seconds=0)).date().isoformat(),
            latest_hr.isoformat() if latest_hr else "-",
            latest_sleep.isoformat() if latest_sleep else "-",
            n_sleep,
            n_hr,
            settings.chronotype,
            settings.n_particles,
            settings.n_light_samples,
        )
    )
    import hashlib

    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)


def _latest_observation(session: Session) -> datetime | None:
    """Last instant backed by real device data, across every source."""
    candidates = [
        session.scalar(select(func.max(HeartRateMinute.ts))),
        session.scalar(select(func.max(StepMinute.ts))),
        session.scalar(select(func.max(SleepSession.end_ts))),
    ]
    times = [t for t in candidates if t is not None]
    return max(times) if times else None


def _earliest_observation(session: Session) -> datetime | None:
    """First instant backed by real device data, across every source."""
    candidates = [
        session.scalar(select(func.min(HeartRateMinute.ts))),
        session.scalar(select(func.min(StepMinute.ts))),
        session.scalar(select(func.min(SleepSession.start_ts))),
    ]
    times = [t for t in candidates if t is not None]
    return min(times) if times else None


def _habitual_sleep_window(
    session: Session, settings: RuntimeSettings, as_of: datetime
) -> tuple[float, float]:
    """(onset, wake) in local hours, for days with no device data.

    Derived from observed nights when there are any, otherwise from the
    chronotype prior. Only ever used to decide where darkness goes on a day the
    watch was not worn; it is an assumption, and the ensemble flags it as one.
    """
    nights = sleep_phase.load_nights(session, settings, as_of)
    onsets, wakes = [], []
    for night in nights:
        if night.start_ts is None or night.end_ts is None:
            continue
        tz_name = night.tz_name or "UTC"
        onsets.append(
            sleep_phase.local_hour_of_day(night.start_ts, tz_name, night.utc_offset_seconds)
        )
        wakes.append(
            sleep_phase.local_hour_of_day(night.end_ts, tz_name, night.utc_offset_seconds)
        )
    if onsets:
        return float(circular_mean(np.array(onsets))), float(circular_mean(np.array(wakes)))

    # No nights yet: centre a population-typical 8 h on the chronotype's
    # implied midpoint (DLMO prior + its sleep-midpoint offset).
    midpoint = wrap_hours(settings.prior["dlmo_offset_hours"] + 21.5)
    return float(wrap_hours(midpoint - 4.0)), float(wrap_hours(midpoint + 4.0))


def estimate(
    session: Session,
    as_of: datetime | None = None,
    settings: RuntimeSettings | None = None,
    persist: bool = True,
    seed: int | None = None,
) -> PhaseResult | None:
    """Run one full phase estimation. Returns None when there is no data at all."""
    as_of = as_of or datetime.now(UTC)
    settings = settings or load_settings(session)
    config = get_settings()
    tz_name = config.timezone
    offset = _utc_offset(tz_name, as_of)
    if seed is None:
        seed = _data_seed(session, as_of, settings)

    # --- channel observations ---------------------------------------------
    observations: list[ChannelObservation] = []
    channel_detail: dict[str, dict] = {}

    sleep_obs = sleep_phase.estimate(session, settings, as_of)
    hr_obs = hr_phase.estimate(session, settings, as_of, tz_offset_seconds=offset)

    if sleep_obs is None and hr_obs is None:
        log.info("engine.no_observations")
        return None

    # Never propagate the oscillator through days the device did not see. The
    # light proxy has to emit *something* for every bin, so a window that
    # reaches back past the first real data drives the model with a schedule
    # nobody lived - and light is the strongest input it has.
    filter_start = as_of - timedelta(days=FILTER_WINDOW_DAYS)
    earliest = _earliest_observation(session)
    if earliest is not None and earliest > filter_start:
        filter_start = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
    filter_days = max((as_of - filter_start).days, 1)

    if sleep_obs is not None:
        # The sleep channel is an aggregate over many nights, so it is applied
        # as a repeated daily observation rather than a single update - that is
        # what lets the dynamics and the behaviour argue with each other.
        for day in range(filter_days):
            observations.append(
                ChannelObservation(
                    day=filter_start + timedelta(days=day),
                    channel="sleep",
                    mu_hours=sleep_obs.mu_hours,
                    # Spread the evidence across days so it is not counted
                    # once per day over.
                    kappa=sleep_obs.kappa / filter_days,
                )
            )
        channel_detail["sleep"] = {
            "mu_hours": round(sleep_obs.mu_hours, 3),
            "kappa": round(sleep_obs.kappa, 3),
            "midpoint_hours": round(sleep_obs.midpoint_hours, 3),
            "n_nights": sleep_obs.n_nights,
            "n_free_nights": sleep_obs.n_free_nights,
            **sleep_obs.detail,
        }

    if hr_obs is not None:
        observations.append(
            ChannelObservation(
                day=as_of - timedelta(days=1),
                channel="hr",
                mu_hours=hr_obs.mu_hours,
                kappa=hr_obs.kappa,
            )
        )
        channel_detail["hr"] = {
            "mu_hours": round(hr_obs.mu_hours, 3),
            "kappa": round(hr_obs.kappa, 3),
            "acrophase_hours": round(hr_obs.acrophase_hours, 3),
            "amplitude_bpm": round(hr_obs.amplitude_bpm, 2),
            "coverage": round(hr_obs.coverage, 3),
            "r_squared": round(hr_obs.r_squared, 3),
            **hr_obs.detail,
        }

    # --- light ensemble ----------------------------------------------------
    # The light history ends at the last thing actually observed, not at "now".
    # Ending it at the clock meant the window gained a bin every ten minutes and
    # the posterior moved a couple of minutes with it - so the estimate drifted
    # between polls on which no new data had arrived, and the calendar was
    # rewritten for it. The estimate should be a function of the data, which is
    # the same promise `_data_seed` already makes about the random draws.
    observed_to = _latest_observation(session)
    ensemble_end = min(observed_to, as_of) if observed_to else as_of
    ensemble_end = max(ensemble_end, filter_start + timedelta(hours=1))

    ensemble = light_proxy.build_ensemble(
        session,
        filter_start,
        ensemble_end,
        n_samples=settings.n_light_samples,
        rng=np.random.default_rng(seed),
        habitual_sleep=_habitual_sleep_window(session, settings, as_of),
        utc_offset_seconds=offset,
    )

    # --- filter -------------------------------------------------------------
    posterior = run_filter(
        ensemble,
        observations,
        n_particles=settings.n_particles,
        tau_prior_mean=settings.prior["tau_hours"],
        utc_offset_seconds=offset,
        seed=seed,
    )

    # --- confidence ---------------------------------------------------------
    # Distinct nights, not sessions. Fitbit splits a night at a long awakening
    # and both halves can clear the main-sleep threshold, so counting rows made
    # the tier advance faster than the evidence justified.
    n_nights = session.scalar(
        select(func.count(func.distinct(SleepSession.sleep_date)))
        .select_from(SleepSession)
        .where(SleepSession.is_main_sleep.is_(True), SleepSession.excluded.is_(False))
    ) or 0

    last_sleep = session.scalar(
        select(SleepSession)
        .where(SleepSession.is_main_sleep.is_(True), SleepSession.excluded.is_(False))
        .order_by(SleepSession.end_ts.desc())
    )
    last_sleep_hours = (last_sleep.tst_minutes / 60.0) if last_sleep and last_sleep.tst_minutes else None

    midpoints = [m for _, m, _ in sleep_phase.nightly_midpoints(session, settings, as_of)]
    sleep_sd = None
    if len(midpoints) >= 3:
        from circa.phase.circular import circular_sd_hours

        sleep_sd = circular_sd_hours(np.array(midpoints))

    coverage = _data_coverage(session, as_of)
    ood = confidence_mod.detect_ood(
        session,
        as_of,
        sleep_sd_hours=sleep_sd,
        illness=_illness_flag(session, as_of),
        travel_mode=settings.travel_mode,
        hr_coverage=coverage,
        last_sleep_hours=last_sleep_hours,
    )

    ci_width = interval_width_hours(*posterior.dlmo_ci80) * 60
    conf = confidence_mod.assess(
        n_nights=n_nights,
        ci80_width_minutes=ci_width,
        data_coverage=coverage,
        channel_agreement_minutes=posterior.channel_agreement_minutes,
        ood_flags=ood,
    )

    # --- assemble ----------------------------------------------------------
    dlmo_ts = _local_hours_to_ts(posterior.dlmo_hours, as_of, offset)
    # CBTmin belongs to the biological night its own DLMO opened, roughly seven
    # hours later, so it is placed relative to that DLMO rather than snapped to
    # `as_of` independently. Snapping both to "nearest occurrence" let them land
    # on different nights: between 10:00 and 18:00 local the pair came out with
    # the core-temperature minimum seventeen hours *before* the melatonin onset
    # that precedes it, and the light blocks anchored to it moved a day.
    cbt_ts = _local_hour_after(posterior.cbtmin_hours, dlmo_ts, offset)
    ci_lo = _local_hours_to_ts(posterior.dlmo_ci80[0], dlmo_ts, offset)
    ci_hi = _local_hours_to_ts(posterior.dlmo_ci80[1], dlmo_ts, offset)

    result = PhaseResult(
        target_date=(as_of + timedelta(seconds=offset)).date(),
        dlmo_local_hours=posterior.dlmo_hours,
        dlmo_ts=dlmo_ts,
        cbtmin_local_hours=posterior.cbtmin_hours,
        cbtmin_ts=cbt_ts,
        ci80=(ci_lo, ci_hi),
        ci80_local_hours=posterior.dlmo_ci80,
        sd_minutes=posterior.sd_minutes,
        confidence=conf,
        posterior=posterior,
        observations=observations,
        tz_name=tz_name,
        utc_offset_seconds=offset,
    )

    if persist:
        _persist(session, result, channel_detail)

    log.info(
        "engine.estimated",
        dlmo=round(posterior.dlmo_hours, 2),
        sd_min=round(posterior.sd_minutes),
        tier=conf.tier,
        q=conf.q,
        ood=ood,
    )
    return result


def _persist(session: Session, result: PhaseResult, channel_detail: dict) -> None:
    existing = session.scalar(
        select(PhaseEstimate).where(
            PhaseEstimate.target_date == result.target_date,
            PhaseEstimate.model_version == MODEL_VERSION,
        )
    )
    if existing is None:
        existing = PhaseEstimate(
            target_date=result.target_date, model_version=MODEL_VERSION
        )
        session.add(existing)

    existing.computed_at = datetime.now(UTC)
    existing.dlmo_ts = result.dlmo_ts
    existing.dlmo_ci80_low = result.ci80[0]
    existing.dlmo_ci80_high = result.ci80[1]
    existing.cbtmin_ts = result.cbtmin_ts
    existing.phase_sd_minutes = result.sd_minutes
    existing.confidence_tier = result.confidence.tier
    existing.confidence_q = result.confidence.q
    existing.n_nights = result.confidence.n_nights
    existing.channel_agreement_minutes = result.posterior.channel_agreement_minutes
    existing.detail = {
        "dlmo_local_hours": round(result.dlmo_local_hours, 4),
        "cbtmin_local_hours": round(result.cbtmin_local_hours, 4),
        "ci80_local_hours": [round(h, 4) for h in result.ci80_local_hours],
        "ci95_local_hours": [round(h, 4) for h in result.posterior.dlmo_ci95],
        "tau_mean": round(result.posterior.tau_mean, 4),
        "tau_sd": round(result.posterior.tau_sd, 4),
        "amplitude": round(result.posterior.amplitude_mean, 4),
        "ess": round(result.posterior.ess, 1),
        "confidence": {
            "q_posterior": result.confidence.q_posterior,
            "q_coverage": result.confidence.q_coverage,
            "q_agreement": result.confidence.q_agreement,
            "q_ood": result.confidence.q_ood,
            "ood_flags": result.confidence.ood_flags,
        },
        "channels": channel_detail,
        "filter": result.posterior.detail,
    }

    for channel, detail in channel_detail.items():
        obs_row = session.scalar(
            select(PhaseObservation).where(
                PhaseObservation.obs_date == result.target_date,
                PhaseObservation.channel == channel,
                PhaseObservation.model_version == MODEL_VERSION,
            )
        )
        if obs_row is None:
            obs_row = PhaseObservation(
                obs_date=result.target_date,
                channel=channel,
                model_version=MODEL_VERSION,
            )
            session.add(obs_row)
        obs_row.mu_hours = detail["mu_hours"]
        obs_row.kappa = detail["kappa"]
        obs_row.computed_at = datetime.now(UTC)
        obs_row.detail = detail


def latest(session: Session) -> PhaseEstimate | None:
    return session.scalar(
        select(PhaseEstimate)
        .where(PhaseEstimate.model_version == MODEL_VERSION)
        .order_by(PhaseEstimate.computed_at.desc())
    )

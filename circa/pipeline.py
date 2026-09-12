"""The full run: data -> phase -> alertness -> calendar.

Called by the scheduler after each successful sync, and by `circa run`.
Deliberately a single linear function: every stage is independently testable,
and when something goes wrong the report says which stage it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.orm import Session

from circa import forecast_store
from circa.alertness.model import AlertnessCurve, compute, grid
from circa.alertness.process_s import (
    SleepDebt,
    load_history,
    recent_sleep_debt,
    target_sleep_tonight,
)
from circa.gcal import sync as gcal_sync
from circa.gcal.blocks import Block, build_all
from circa.phase import engine, sleep_phase
from circa.phase.engine import MODEL_VERSION, PhaseResult
from circa.settings_store import RuntimeSettings, load_settings

log = structlog.get_logger(__name__)

# How far behind the current moment the alertness curve is computed. Enough
# to cover the whole of the current local day from any hour of it.
CURVE_LOOKBACK_HOURS = 24

# How far past the forecast horizon the curve is computed, so that a phase
# anchor landing near the end of the horizon still has a full day of curve
# underneath it to be validated against.
CURVE_OVERRUN_HOURS = 22

# Resolution of the alertness curve, and the lattice its samples sit on.
CURVE_STEP_MINUTES = 10


@dataclass
class PipelineReport:
    ran_at: datetime
    phase: PhaseResult | None = None
    curve: AlertnessCurve | None = None
    blocks: list[Block] = field(default_factory=list)
    calendar: gcal_sync.SyncReport | None = None
    sleep_debt: SleepDebt | None = None
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)


def run(
    session: Session,
    as_of: datetime | None = None,
    push_calendar: bool = True,
    settings: RuntimeSettings | None = None,
    seed: int | None = None,
) -> PipelineReport:
    """Run every stage.

    Each stage commits before the next begins. Stages that make network calls
    must not hold a write transaction across them: SQLite permits a single
    writer, and the OAuth layer refreshes access tokens on its own connection,
    so a long-held transaction deadlocks against it with "database is locked".
    """
    as_of = as_of or datetime.now(UTC)
    settings = settings or load_settings(session)
    report = PipelineReport(ran_at=as_of)

    if settings.paused:
        report.skipped_reason = "paused in settings"
        return report

    # --- 1. forced-wake marking -------------------------------------------
    # The local heuristic runs every time; the calendar only refines it. Gating
    # the whole step on calendar access left `wake_forced` NULL on every
    # read-only run, and the free-wake probe excludes NULL - so the strongest
    # passive phase anchor available was simply never populated.
    try:
        from circa.gcal.schedule import mark_forced_wakes

        mark_forced_wakes(session, settings, use_calendar=push_calendar)
        session.commit()
    except Exception as exc:  # noqa: BLE001 - optional enrichment
        log.warning("pipeline.forced_wake_failed", error=str(exc))
        report.errors.append(f"forced-wake detection: {exc}")

    # --- 2. phase -----------------------------------------------------------
    try:
        phase = engine.estimate(session, as_of=as_of, settings=settings, seed=seed)
        # Persist the estimate before any network work begins.
        session.commit()
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline.phase_failed", error=str(exc), exc_info=True)
        report.errors.append(f"phase estimation: {exc}")
        return report

    if phase is None:
        report.skipped_reason = "no usable sleep or heart-rate data yet"
        return report
    report.phase = phase

    # --- 3. alertness --------------------------------------------------------
    try:
        horizon_end = as_of + timedelta(hours=settings.forecast_horizon_hours)
        # Reach back a full day, not twelve hours. At 23:00 a twelve-hour window
        # starts at 11:00 and the morning is simply gone from the chart - so the
        # page could not show you how the day you just lived actually went.
        # Snapped to a day boundary, not to "now". `grid` lays its points out
        # from the start, so anchoring it to the current instant moved every
        # sample by however many seconds had passed since the last run. The
        # extremum shifted by up to a step, the sampled sleep/wake edges moved
        # with it, and block boundaries flipped across the rounding quantum -
        # so two runs minutes apart rewrote calendar entries that had not
        # changed. Snapped this way the whole forecast is identical for every
        # run within a day.
        curve_start = _snap_to_grid(
            as_of - timedelta(hours=CURVE_LOOKBACK_HOURS), CURVE_STEP_MINUTES
        )
        # The curve has to run past the forecast horizon, not stop at it. Sleep,
        # light and body blocks are anchored to a phase recurrence and only used
        # when the whole 24 h they span is inside the curve - so a curve that
        # ended exactly at the horizon left room for one night's worth of them,
        # while the energy blocks (which read the curve directly) covered three
        # days. Extending it is cheap: the expensive part is the oscillator,
        # which has already run by this point.
        curve_end = horizon_end + timedelta(hours=CURVE_OVERRUN_HOURS)
        times = grid(curve_start, curve_end, minutes=CURVE_STEP_MINUTES)
        # Snapped to a day boundary, not to "now". Process S is warmed up by
        # integrating from the start of this window, so moving it by a few
        # minutes changed the whole trajectory by a hair - enough, at a knife
        # edge, to flip a block boundary across the rounding quantum. Anchored
        # this way the forecast is identical for every run within a day.
        history_start = _snap_to_grid(as_of - timedelta(days=10), 24 * 60)
        history = load_history(session, history_start, curve_end)

        # Project the observed schedule forward, otherwise Process S has no
        # sleep to recover from and the forecast drifts monotonically upward.
        history.episodes = _project_sleep(
            history.episodes, phase, settings, curve_end,
            target_hours=target_sleep_tonight(
                settings.target_sleep_hours,
                recent_sleep_debt(
                    session, as_of, settings.target_sleep_hours,
                    settings.sleep_debt_window_days,
                ),
                settings.sleep_debt_payback_fraction,
                settings.max_debt_payback_hours,
            ),
        )

        cbt_samples = _cbtmin_samples(phase)
        curve = compute(
            times,
            phase.utc_offset_seconds,
            history,
            cbt_samples,
            weights=phase.posterior.weights,
        )
        report.curve = curve
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline.alertness_failed", error=str(exc), exc_info=True)
        report.errors.append(f"alertness: {exc}")
        return report

    # --- 4. blocks -----------------------------------------------------------
    try:
        # What actually happened, alongside what is predicted to. Recorded
        # nights reach back over the history window so the calendar reads as a
        # record behind you and a plan ahead of you.
        recorded = sleep_phase.load_nights(
            session,
            settings.model_copy(update={"sleep_history_days": max(settings.history_days, 1)}),
            as_of,
        )
        debt = recent_sleep_debt(
            session, as_of, settings.target_sleep_hours, settings.sleep_debt_window_days
        )
        report.sleep_debt = debt
        report.blocks = build_all(
            curve=curve,
            dlmo_ts=phase.dlmo_ts,
            cbtmin_ts=phase.cbtmin_ts,
            ci=phase.ci80,
            conf=phase.confidence,
            settings=settings,
            offset=phase.utc_offset_seconds,
            posterior_detail=phase.posterior.detail,
            from_ts=as_of,
            to_ts=horizon_end,
            observed_nights=recorded,
            debt=debt,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline.blocks_failed", error=str(exc), exc_info=True)
        report.errors.append(f"block generation: {exc}")
        return report

    # --- 5. calendar ----------------------------------------------------------
    if push_calendar:
        try:
            report.calendar = gcal_sync.push(
                session, report.blocks, settings, model_version=MODEL_VERSION
            )
        except Exception as exc:  # noqa: BLE001
            log.error("pipeline.calendar_failed", error=str(exc), exc_info=True)
            report.errors.append(f"calendar sync: {exc}")

    # --- 6. publish for the UI ------------------------------------------------
    # Written last, so the stored forecast only ever reflects a run that got all
    # the way through.
    try:
        forecast_store.save_curve(
            session, forecast_store.serialise_curve(report), MODEL_VERSION
        )
    except Exception as exc:  # noqa: BLE001 - the run itself already succeeded
        log.warning("pipeline.forecast_store_failed", error=str(exc))

    log.info(
        "pipeline.done",
        blocks=len(report.blocks),
        tier=phase.confidence.tier,
        errors=len(report.errors),
    )
    return report


def _snap_to_grid(ts: datetime, minutes: int) -> datetime:
    """Floor an instant onto a fixed lattice, so every run shares one."""
    step = max(minutes, 1) * 60
    return datetime.fromtimestamp((int(ts.timestamp()) // step) * step, tz=UTC)


def _cbtmin_samples(phase: PhaseResult):
    """CBTmin posterior samples in local clock hours.

    Derived from the DLMO samples by the model's fixed CBT->DLMO offset, so the
    alertness band inherits the phase posterior rather than a separate guess.
    """
    import numpy as np

    from circa.phase.oscillator import CBT_TO_DLMO_HOURS

    return np.mod(phase.posterior.samples_hours + CBT_TO_DLMO_HOURS, 24.0)


def _project_sleep(
    episodes: list[tuple[datetime, datetime]],
    phase: PhaseResult,
    settings: RuntimeSettings,
    until: datetime,
    target_hours: float | None = None,
) -> list[tuple[datetime, datetime]]:
    """Extend observed sleep forward with the predicted sleep window.

    Without this the homeostat sees an unbroken wake period across the forecast
    horizon and predicts a collapse that will not happen, because the person is
    in fact going to sleep tonight.
    """
    # The same debt-adjusted target the Sleep calendar recommends. Projecting a
    # plain 8 h while the calendar advises 8h45 would have the energy forecast
    # quietly disagreeing with the advice sitting next to it.
    hours = settings.target_sleep_hours if target_hours is None else target_hours
    projected = list(episodes)
    last_end = max((e for _, e in episodes), default=phase.dlmo_ts)
    onset = phase.dlmo_ts + timedelta(hours=2)

    while onset < until:
        if onset > last_end:
            projected.append((onset, onset + timedelta(hours=hours)))
        onset += timedelta(hours=24)
    return sorted(projected)

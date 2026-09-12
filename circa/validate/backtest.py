"""Held-out-DAY backtesting and the model ablation ladder.

**Days are held out whole, never randomly sampled.** A random split puts 10:00
and 10:10 from the same day in train and test, leaking near-identical state
across the boundary. That makes a bad model look excellent, and it is the single
easiest way to fool yourself in time-series work.

The comparison target is the passive behavioural probes in `probes.py`, since a
fully passive system has no laboratory DLMO to score against. That means results
here measure *self-consistency*, not absolute accuracy — a distinction the
report is explicit about rather than glossing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from sqlalchemy.orm import Session

from circa.phase import engine, sleep_phase
from circa.phase.circular import circular_difference, circular_mean
from circa.settings_store import RuntimeSettings, load_settings
from circa.validate import probes as probe_mod
from circa.validate.metrics import PhaseMetrics, interval_width_minutes, phase_metrics

log = structlog.get_logger(__name__)


@dataclass
class AblationResult:
    name: str
    description: str
    metrics: PhaseMetrics | None
    error: str | None = None


@dataclass
class BacktestReport:
    n_days: int
    generated_at: datetime
    ablations: list[AblationResult] = field(default_factory=list)
    probe_consistency: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_days": self.n_days,
            "generated_at": self.generated_at.isoformat(),
            "ablations": [
                {
                    "name": a.name,
                    "description": a.description,
                    "metrics": a.metrics.as_dict() if a.metrics else None,
                    "error": a.error,
                }
                for a in self.ablations
            ],
            "probe_consistency": self.probe_consistency,
            "notes": self.notes,
        }


def _held_out_days(session: Session, settings: RuntimeSettings, min_days: int = 10) -> list[datetime]:
    """Evaluation instants: one per night with enough history behind it."""
    nights = sleep_phase.load_nights(session, settings)
    if len(nights) < min_days:
        return []
    # Skip the earliest nights - they have no history to estimate from.
    warmup = max(int(len(nights) * 0.4), 5)
    return [n.end_ts + timedelta(hours=12) for n in nights[warmup:]]


def run(
    session: Session,
    settings: RuntimeSettings | None = None,
    max_days: int = 14,
    seed: int = 0,
) -> BacktestReport:
    """Run the ablation ladder over held-out days."""
    settings = settings or load_settings(session)
    report = BacktestReport(n_days=0, generated_at=datetime.now(UTC))

    days = _held_out_days(session, settings)
    if not days:
        report.notes.append(
            "Not enough nights to backtest yet. This becomes meaningful at "
            "roughly 14 nights and informative at 30."
        )
        return report

    days = days[-max_days:]
    report.n_days = len(days)

    # Reference: what the free-wake behavioural probe implies, which is the
    # closest thing to an independent anchor a passive system has.
    reference = probe_mod.free_wake_phase_estimate(session)
    if reference is None:
        report.notes.append(
            "No alarm-free nights yet, so there is no independent anchor to "
            "score against. Model agreement is reported instead of accuracy."
        )

    ladder = [
        ("M0", "sleep timing only", {"hr": False, "light": False}),
        ("M1", "+ activity-corrected heart rate", {"hr": True, "light": False}),
        ("M2", "+ probabilistic light and oscillator dynamics", {"hr": True, "light": True}),
    ]

    for name, description, config in ladder:
        try:
            metrics = _evaluate(session, settings, days, reference, config, seed)
            report.ablations.append(AblationResult(name, description, metrics))
        except Exception as exc:  # noqa: BLE001
            log.warning("backtest.ablation_failed", name=name, error=str(exc))
            report.ablations.append(AblationResult(name, description, None, str(exc)))

    # --- passive probe consistency -----------------------------------------
    latest = engine.latest(session)
    if latest is not None and latest.detail:
        onset = float(latest.detail.get("dlmo_local_hours", 0.0)) + 2.0
        report.probe_consistency = probe_mod.latency_consistency(
            probe_mod.sleep_latency_probe(session), onset % 24
        )

    report.notes.append(
        "Scored against passive behavioural probes, not laboratory DLMO. These "
        "measure self-consistency; absolute accuracy would need a melatonin "
        "assay, which this setup deliberately does not require."
    )
    return report


def _evaluate(
    session: Session,
    settings: RuntimeSettings,
    days: list[datetime],
    reference: float | None,
    config: dict,
    seed: int,
) -> PhaseMetrics:
    """Estimate phase at each held-out instant and score it."""
    tuned = settings.model_copy(deep=True)
    if not config["hr"]:
        tuned.hr_window_hours = 0  # disables the HR channel
    if not config["light"]:
        tuned.n_light_samples = 4
        tuned.n_particles = 400

    predictions: list[float] = []
    intervals80: list[tuple[float, float]] = []

    for as_of in days:
        result = engine.estimate(
            session, as_of=as_of, settings=tuned, persist=False, seed=seed
        )
        if result is None:
            continue
        predictions.append(result.dlmo_local_hours)
        intervals80.append(result.ci80_local_hours)

    if not predictions:
        raise RuntimeError("no estimates produced")

    predicted = np.array(predictions)
    if reference is not None:
        actual = np.full_like(predicted, reference)
    else:
        # With no external anchor, score against the model's own long-run mean:
        # this measures stability, not accuracy, and is labelled as such.
        actual = np.full_like(predicted, circular_mean(predicted))

    metrics = phase_metrics(predicted, actual, ci80=intervals80)
    metrics.detail["anchor"] = "free_wake_probe" if reference is not None else "self_mean"
    metrics.detail["mean_ci80_width_minutes"] = round(
        interval_width_minutes(intervals80), 1
    )
    return metrics


def stability(session: Session, settings: RuntimeSettings | None = None) -> dict:
    """Night-to-night movement of the estimate.

    Large day-to-day swings with no schedule change mean the filter is chasing
    noise, which no accuracy number would reveal on its own.
    """
    from sqlalchemy import select

    from circa.db.models import PhaseEstimate

    rows = list(
        session.scalars(
            select(PhaseEstimate)
            .where(PhaseEstimate.model_version == engine.MODEL_VERSION)
            .order_by(PhaseEstimate.target_date)
        )
    )
    values = [
        r.detail.get("dlmo_local_hours")
        for r in rows
        if r.detail and r.detail.get("dlmo_local_hours") is not None
    ]
    if len(values) < 3:
        return {"n": len(values), "median_nightly_shift_minutes": None}

    shifts = [
        abs(float(circular_difference(values[i], values[i - 1]))) * 60
        for i in range(1, len(values))
    ]
    return {
        "n": len(values),
        "median_nightly_shift_minutes": round(float(np.median(shifts)), 1),
        "max_nightly_shift_minutes": round(float(np.max(shifts)), 1),
    }

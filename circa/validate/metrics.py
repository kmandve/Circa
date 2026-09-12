"""Metrics for circular phase prediction.

Two things are non-negotiable here:

* **Errors are circular.** A prediction of 00:30 against a truth of 23:50 is 40
  minutes late, not 23 hours 20 minutes early.
* **Calibration is reported alongside accuracy.** A model that is usually close
  but whose "80% interval" contains the truth 30% of the time is not usable for
  a calendar, because the block widths would be lies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from circa.phase.circular import circular_difference, interval_width_hours


@dataclass
class PhaseMetrics:
    n: int
    mae_minutes: float
    rmse_minutes: float
    bias_minutes: float          # signed: positive = predicts late
    p30: float                   # fraction within 30 min
    p60: float
    p90: float
    worst_decile_minutes: float
    coverage_80: float | None = None
    coverage_95: float | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mae_minutes": round(self.mae_minutes, 1),
            "rmse_minutes": round(self.rmse_minutes, 1),
            "bias_minutes": round(self.bias_minutes, 1),
            "p30": round(self.p30, 3),
            "p60": round(self.p60, 3),
            "p90": round(self.p90, 3),
            "worst_decile_minutes": round(self.worst_decile_minutes, 1),
            "coverage_80": None if self.coverage_80 is None else round(self.coverage_80, 3),
            "coverage_95": None if self.coverage_95 is None else round(self.coverage_95, 3),
            **self.detail,
        }


def phase_metrics(
    predicted_hours: np.ndarray,
    actual_hours: np.ndarray,
    ci80: list[tuple[float, float]] | None = None,
    ci95: list[tuple[float, float]] | None = None,
) -> PhaseMetrics:
    predicted = np.asarray(predicted_hours, dtype=float)
    actual = np.asarray(actual_hours, dtype=float)
    if predicted.size == 0:
        raise ValueError("no predictions")

    errors = np.asarray(circular_difference(predicted, actual), dtype=float) * 60.0
    absolute = np.abs(errors)

    return PhaseMetrics(
        n=int(predicted.size),
        mae_minutes=float(np.mean(absolute)),
        rmse_minutes=float(np.sqrt(np.mean(errors**2))),
        bias_minutes=float(np.mean(errors)),
        p30=float(np.mean(absolute <= 30)),
        p60=float(np.mean(absolute <= 60)),
        p90=float(np.mean(absolute <= 90)),
        # Guards against a model that looks fine on average but fails badly on
        # the days that matter - exactly the shift-work failure mode.
        worst_decile_minutes=float(np.percentile(absolute, 90)),
        coverage_80=_coverage(actual, ci80),
        coverage_95=_coverage(actual, ci95),
    )


def _coverage(actual_hours: np.ndarray, intervals) -> float | None:
    if not intervals:
        return None
    hits = 0
    for value, (lo, hi) in zip(actual_hours, intervals, strict=False):
        span = interval_width_hours(lo, hi)
        offset = float(np.mod(value - lo, 24.0))
        if offset <= span:
            hits += 1
    return hits / len(intervals)


def interval_width_minutes(intervals) -> float:
    return float(np.mean([interval_width_hours(lo, hi) * 60 for lo, hi in intervals]))

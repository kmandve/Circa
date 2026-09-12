"""Storage for the computed forecast, so the web app never runs the model.

One row, rewritten by the scheduler. Everything the UI draws comes from here.
The alternative - recomputing per request - cost nine seconds a page load and
produced numbers that could differ from the ones already written to the
calendar, because the filter is only deterministic given identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy.orm import Session

from circa.db.models import ForecastCache

log = structlog.get_logger(__name__)

CURVE_KIND = "curve"

# Past this the forecast is still shown, but labelled. It is not wrong, only
# old: the phase estimate moves slowly and a six-hour-old curve is still a
# better answer than a spinner.
STALE_AFTER = timedelta(hours=6)


@dataclass(frozen=True)
class StoredForecast:
    computed_at: datetime
    model_version: str
    payload: dict

    @property
    def age(self) -> timedelta:
        return datetime.now(UTC) - self.computed_at

    @property
    def is_stale(self) -> bool:
        return self.age > STALE_AFTER


def save_curve(
    session: Session, payload: dict[str, Any], model_version: str
) -> None:
    row = session.get(ForecastCache, CURVE_KIND)
    if row is None:
        row = ForecastCache(kind=CURVE_KIND)
        session.add(row)
    row.computed_at = datetime.now(UTC)
    row.model_version = model_version
    row.payload = payload
    session.flush()


def load_curve(session: Session) -> StoredForecast | None:
    row = session.get(ForecastCache, CURVE_KIND)
    if row is None or not row.payload:
        return None
    return StoredForecast(
        computed_at=row.computed_at,
        model_version=row.model_version,
        payload=row.payload,
    )


def serialise_curve(report) -> dict:
    """Everything the UI needs from a pipeline run, and nothing else.

    Deliberately a plain dict of primitives rather than the live objects: it has
    to survive a restart, and it must not tempt a caller into recomputing a
    field that was not stored.
    """
    curve, phase = report.curve, report.phase
    offset = phase.utc_offset_seconds
    return {
        "as_of": report.ran_at.isoformat(),
        # Every instant below is shifted into the user's configured local time
        # and then labelled "+00:00". That is deliberate, and the chart is drawn
        # with ECharts' useUTC so it renders them literally. The alternative -
        # sending true UTC and letting the browser localise - shows the clock of
        # wherever the page happens to be open, which is not what this app is
        # about, and disagreed with every server-rendered time on the same page.
        # The "now" marker is *not* stored: a forecast is served for up to a
        # scheduler interval after it was computed, so a baked-in marker would
        # drift behind the clock. The page builds it from `utc_offset_seconds`
        # instead, which keeps both on the same convention.
        "computed_for": (report.ran_at + timedelta(seconds=offset)).isoformat(),
        "utc_offset_seconds": offset,
        "model_version": getattr(phase, "model_version", ""),
        "points": [
            {
                "t": (t + timedelta(seconds=offset)).isoformat(),
                "e": round(float(e), 1),
                "lo": round(float(lo), 1),
                "hi": round(float(hi), 1),
                "asleep": bool(a),
            }
            for t, e, lo, hi, a in zip(
                curve.times, curve.energy, curve.energy_lower,
                curve.energy_upper, curve.asleep, strict=False,
            )
        ],
        "energy_peak": curve.detail.get("energy_peak"),
        "energy_mean_awake": curve.detail.get("energy_mean_awake"),
        "dlmo_local_hours": phase.dlmo_local_hours,
        "cbtmin_local_hours": curve.cbtmin_hours,
        "tier": phase.confidence.tier,
        "sleep_debt_hours": report.sleep_debt.hours if report.sleep_debt else None,
    }

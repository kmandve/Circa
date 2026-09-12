"""Routes raw payloads of a given data type to the right normaliser."""

from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from circa.db.models import RawDataPoint
from circa.normalize.activity import normalize_steps
from circa.normalize.heart_rate import normalize_heart_rate
from circa.normalize.sleep import normalize_sleep
from circa.normalize.vitals import normalize_daily, normalize_exercise, normalize_hrv

log = structlog.get_logger(__name__)

_HANDLERS = {
    "sleep": normalize_sleep,
    "steps": normalize_steps,
    "heart-rate": normalize_heart_rate,
    "heart-rate-variability": normalize_hrv,
    "exercise": normalize_exercise,
}


def normalize_type(session: Session, data_type: str, raw_points: list[RawDataPoint]) -> int:
    """Normalise a batch. Unknown types stay in the raw table only.

    Failing to normalise is never fatal: the raw payload is already persisted,
    so a schema surprise costs us a re-run, not the data.
    """
    if not raw_points:
        return 0

    handler = _HANDLERS.get(data_type)
    try:
        if handler is not None:
            return handler(session, raw_points)
        if data_type.startswith("daily-"):
            return normalize_daily(session, data_type, raw_points)
    except Exception as exc:  # noqa: BLE001 - raw data is safe; keep polling
        log.error("normalize.failed", data_type=data_type, error=str(exc), exc_info=True)
        return 0

    log.debug("normalize.no_handler", data_type=data_type, count=len(raw_points))
    return 0

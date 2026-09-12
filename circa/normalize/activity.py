"""Step / activity normalisation.

Steps do double duty: they are the covariate that de-masks heart rate, and they
are the *input to the light proxy*. Huang et al. 2021 found that scaled step
counts fed to a circadian model matched or beat measured wrist light — which is
why the Fitbit Air having no ambient light sensor is survivable.

The API returns variable-length bouts rather than uniform minutes, so bouts are
redistributed onto a 1-minute grid before use.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from circa.db.models import RawDataPoint, StepInterval, StepMinute
from circa.ingest.parsing import as_float, extract_interval, first

log = structlog.get_logger(__name__)

# A single bout longer than this is almost certainly a summary row, not a bout.
MAX_BOUT_HOURS = 6


def normalize_steps(session: Session, raw_points: list[RawDataPoint]) -> int:
    intervals: dict[tuple[datetime, datetime], int] = {}

    for raw in raw_points:
        payload = raw.payload or {}
        interval_obj = first(payload, "steps.interval", "interval") or {}
        start, end, _ = extract_interval(interval_obj)
        count = as_float(first(payload, "steps.count", "count", "steps.value", "value"))
        if start is None or end is None or count is None or end <= start:
            continue
        if (end - start) > timedelta(hours=MAX_BOUT_HOURS):
            log.debug("steps.bout_too_long", start=start.isoformat(), end=end.isoformat())
            continue
        intervals[(start, end)] = int(count)

    if not intervals:
        return 0

    rows = [
        {"start_ts": s, "end_ts": e, "count": c} for (s, e), c in sorted(intervals.items())
    ]
    for chunk in _chunked(rows, 500):
        session.execute(
            sqlite_insert(StepInterval)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["start_ts", "end_ts"])
        )
    session.flush()

    starts = [r["start_ts"] for r in rows]
    ends = [r["end_ts"] for r in rows]
    rebuild_step_minutes(session, min(starts), max(ends))
    return len(rows)


def rebuild_step_minutes(session: Session, start: datetime, end: datetime) -> int:
    """Spread bouts evenly across the minutes they span.

    Even distribution is an assumption, but a defensible one: the model consumes
    5-minute bins, so sub-bout structure is discarded anyway. What matters is
    that total steps are conserved and land in the right bins.
    """
    start = start.replace(second=0, microsecond=0)
    end = (end + timedelta(minutes=1)).replace(second=0, microsecond=0)

    bouts = session.execute(
        select(StepInterval.start_ts, StepInterval.end_ts, StepInterval.count)
        .where(StepInterval.end_ts > start, StepInterval.start_ts < end)
        .order_by(StepInterval.start_ts)
    ).all()
    if not bouts:
        return 0

    grid: dict[datetime, float] = {}
    for b_start, b_end, count in bouts:
        span_minutes = max((b_end - b_start).total_seconds() / 60.0, 1.0)
        per_minute = count / span_minutes
        minute = b_start.replace(second=0, microsecond=0)
        while minute < b_end:
            # Only credit the portion of this minute the bout actually covers.
            overlap_start = max(minute, b_start)
            overlap_end = min(minute + timedelta(minutes=1), b_end)
            overlap = (overlap_end - overlap_start).total_seconds() / 60.0
            if overlap > 0:
                grid[minute] = grid.get(minute, 0.0) + per_minute * overlap
            minute += timedelta(minutes=1)

    session.execute(delete(StepMinute).where(StepMinute.ts >= start, StepMinute.ts < end))
    rows = [{"ts": ts, "steps": round(value, 3)} for ts, value in sorted(grid.items())]
    for chunk in _chunked(rows, 500):
        session.execute(sqlite_insert(StepMinute).values(chunk))
    session.flush()
    return len(rows)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def utc_now() -> datetime:
    return datetime.now(UTC)

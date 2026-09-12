"""Heart-rate normalisation.

The API returns ~5-second samples (~8,700/day). The phase model only ever
consumes 5-minute bins, so we keep 1-minute robust aggregates forever and let
the raw samples age out after 90 days. That is the single decision that keeps
the whole database around 50 MB/year instead of ~500 MB.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from circa.db.models import HeartRateMinute, HeartRateSample, RawDataPoint
from circa.ingest.parsing import as_float, first, parse_ts

log = structlog.get_logger(__name__)

# Outside this band a reading is a sensor artefact, not physiology.
MIN_PLAUSIBLE_BPM = 25
MAX_PLAUSIBLE_BPM = 235


def normalize_heart_rate(session: Session, raw_points: list[RawDataPoint]) -> int:
    """Write HR samples, then rebuild the minute aggregates they touch."""
    rows: dict[datetime, dict] = {}
    for raw in raw_points:
        payload = raw.payload or {}
        ts = parse_ts(first(payload, "heartRate.sampleTime", "sampleTime", "heart_rate.sampleTime"))
        bpm = as_float(
            first(payload, "heartRate.beatsPerMinute", "beatsPerMinute", "heart_rate.beatsPerMinute", "bpm")
        )
        if ts is None or bpm is None:
            continue
        if not (MIN_PLAUSIBLE_BPM <= bpm <= MAX_PLAUSIBLE_BPM):
            continue
        context = first(
            payload,
            "heartRate.metadata.motionContext",
            "metadata.motionContext",
            "motionContext",
        )
        # Later samples win if the same instant appears twice.
        rows[ts] = {
            "ts": ts,
            "bpm": int(round(bpm)),
            "motion_context": str(context).upper() if context else None,
        }

    if not rows:
        return 0

    values = list(rows.values())
    for chunk in _chunked(values, 500):
        session.execute(
            sqlite_insert(HeartRateSample)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["ts"])
        )
    session.flush()

    timestamps = sorted(rows)
    rebuild_minutes(session, timestamps[0], timestamps[-1] + timedelta(minutes=1))
    return len(values)


def rebuild_minutes(session: Session, start: datetime, end: datetime) -> int:
    """Recompute 1-minute aggregates from samples over [start, end).

    Always derived from scratch for the window rather than updated in place, so
    a late-arriving sample produces the same result as a clean re-run.
    """
    start = start.replace(second=0, microsecond=0)
    samples = session.execute(
        select(HeartRateSample.ts, HeartRateSample.bpm, HeartRateSample.motion_context)
        .where(HeartRateSample.ts >= start, HeartRateSample.ts < end)
        .order_by(HeartRateSample.ts)
    ).all()
    if not samples:
        return 0

    buckets: dict[datetime, list[tuple[int, str | None]]] = {}
    for ts, bpm, context in samples:
        key = ts.replace(second=0, microsecond=0)
        buckets.setdefault(key, []).append((bpm, context))

    session.execute(
        delete(HeartRateMinute).where(HeartRateMinute.ts >= start, HeartRateMinute.ts < end)
    )

    rows = []
    for minute, entries in buckets.items():
        bpms = [b for b, _ in entries]
        active = sum(1 for _, c in entries if c == "ACTIVE")
        rows.append(
            {
                "ts": minute,
                # Median, not mean: one artefact spike should not move the bin.
                "bpm_median": statistics.median(bpms),
                "bpm_min": min(bpms),
                "bpm_max": max(bpms),
                "n_samples": len(bpms),
                "active_fraction": active / len(entries),
            }
        )

    for chunk in _chunked(rows, 500):
        session.execute(sqlite_insert(HeartRateMinute).values(chunk))
    session.flush()
    return len(rows)


def prune_samples(session: Session, older_than: datetime) -> int:
    """Drop raw 5-second samples past the retention window.

    Safe because `rebuild_minutes` has already distilled them into the
    1-minute aggregates the models use.
    """
    result = session.execute(delete(HeartRateSample).where(HeartRateSample.ts < older_than))
    return result.rowcount or 0


def coverage(session: Session, start: datetime, end: datetime) -> float:
    """Fraction of minutes in [start, end) that have any HR data."""
    total = max((end - start).total_seconds() / 60.0, 1.0)
    count = session.scalar(
        select(func.count())
        .select_from(HeartRateMinute)
        .where(HeartRateMinute.ts >= start, HeartRateMinute.ts < end)
    )
    return min((count or 0) / total, 1.0)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def utc_now() -> datetime:
    return datetime.now(UTC)

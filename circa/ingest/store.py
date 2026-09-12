"""Persistence of raw API payloads.

Raw points are written before any interpretation happens, and are never
mutated. Google reconciles records after the fact (a sleep session can be
revised hours later), so every poll deliberately re-requests an overlap window;
`content_hash` makes those re-fetches idempotent.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from circa.db.models import RawDataPoint
from circa.ingest.datatypes import DataType
from circa.ingest.parsing import extract_point_time

log = structlog.get_logger(__name__)

# SQLite caps a statement at 32,766 bound parameters (SQLITE_MAX_VARIABLE_NUMBER).
# A single day of heart rate is ~7-9k points, which at six columns each is well
# past that, so multi-row inserts must be chunked. Derived from the column count
# rather than hard-coded, so adding a column cannot silently reintroduce the
# limit.
_COLUMNS_PER_ROW = 6
_MAX_ROWS_PER_INSERT = 30000 // _COLUMNS_PER_ROW


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def content_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def persist_raw(
    session: Session,
    dt: DataType,
    points: list[dict],
    source: str = "api",
) -> tuple[int, int]:
    """Insert raw points, ignoring exact duplicates.

    Returns (inserted, duplicates).
    """
    if not points:
        return 0, 0

    now = datetime.now(UTC)
    rows: list[dict] = []
    seen: set[str] = set()

    for point in points:
        digest = content_hash(point)
        if digest in seen:  # duplicate inside this same batch
            continue
        seen.add(digest)
        rows.append(
            {
                "data_type": dt.name,
                "source": source,
                "point_time": extract_point_time(point, dt.payload_key),
                "fetched_at": now,
                "content_hash": digest,
                "payload": point,
            }
        )

    if not rows:
        return 0, 0

    before = session.scalar(
        select(func.count()).select_from(RawDataPoint).where(RawDataPoint.data_type == dt.name)
    )
    # ON CONFLICT DO NOTHING against uq_raw_dedupe, in chunks that respect
    # SQLite's bound-parameter ceiling.
    for chunk in _chunked(rows, _MAX_ROWS_PER_INSERT):
        session.execute(
            sqlite_insert(RawDataPoint)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["data_type", "content_hash"])
        )
    session.flush()
    after = session.scalar(
        select(func.count()).select_from(RawDataPoint).where(RawDataPoint.data_type == dt.name)
    )

    inserted = (after or 0) - (before or 0)
    return inserted, len(rows) - inserted


def load_raw(
    session: Session,
    data_type: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[RawDataPoint]:
    stmt = select(RawDataPoint).where(RawDataPoint.data_type == data_type)
    if since is not None:
        stmt = stmt.where(RawDataPoint.point_time >= since)
    if until is not None:
        stmt = stmt.where(RawDataPoint.point_time < until)
    return list(session.scalars(stmt.order_by(RawDataPoint.point_time)))


def prune_raw(session: Session, older_than: datetime, keep_types: set[str] | None = None) -> int:
    """Drop aged-out raw payloads.

    Normalised data is kept forever; this only discards the verbatim JSON, which
    exists for schema debugging and re-normalisation. Sleep and daily summaries
    are cheap and are kept regardless, since they are the records Google is most
    likely to revise.
    """
    keep = keep_types or {
        "sleep",
        "exercise",
        "daily-resting-heart-rate",
        "daily-heart-rate-variability",
        "daily-respiratory-rate",
        "daily-sleep-temperature-derivations",
        "daily-oxygen-saturation",
    }
    stmt = delete(RawDataPoint).where(
        RawDataPoint.fetched_at < older_than,
        RawDataPoint.data_type.notin_(keep),
    )
    result = session.execute(stmt)
    return result.rowcount or 0

"""Retention.

The storage budget is the reason Circa runs free. Without pruning, 5-second
heart rate plus its raw JSON would be roughly 500 MB/year; with it, the whole
database sits near 50 MB/year and fits comfortably on a GCP e2-micro's 30 GB
Always-Free disk (or any free-tier Postgres, should we ever move).

Nothing the models consume is ever deleted. Only two things age out:
raw JSON payloads (a schema-debugging convenience) and 5-second HR samples
(already distilled into the 1-minute aggregates).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import text

from circa.config import get_settings
from circa.db.session import session_scope
from circa.normalize.heart_rate import prune_samples

log = structlog.get_logger(__name__)

# Heart-rate raw JSON is by far the bulkiest and least useful to keep, since the
# 1-minute aggregates are derived immediately on ingest.
_RAW_RETENTION_OVERRIDES = {"heart-rate": 7, "oxygen-saturation": 14}

# How long a deleted calendar block is kept. Long enough that a block which
# vanishes for a day and returns is still recognised as the same one.
_TOMBSTONE_RETENTION_DAYS = 14


@dataclass
class RetentionReport:
    raw_deleted: int = 0
    tombstones_deleted: int = 0
    hr_samples_deleted: int = 0
    bytes_before: int | None = None
    bytes_after: int | None = None


def run_retention(vacuum: bool = False) -> RetentionReport:
    settings = get_settings()
    now = datetime.now(UTC)
    report = RetentionReport()

    with session_scope() as s:
        report.bytes_before = _db_bytes(s)

        from sqlalchemy import delete

        from circa.db.models import RawDataPoint

        keep_forever = {
            "sleep",
            "exercise",
            "daily-resting-heart-rate",
            "daily-heart-rate-variability",
            "daily-respiratory-rate",
            "daily-sleep-temperature-derivations",
            "daily-oxygen-saturation",
        }

        for data_type, days in _RAW_RETENTION_OVERRIDES.items():
            cutoff = now - timedelta(days=days)
            result = s.execute(
                delete(RawDataPoint).where(
                    RawDataPoint.data_type == data_type,
                    RawDataPoint.fetched_at < cutoff,
                )
            )
            report.raw_deleted += result.rowcount or 0

        cutoff = now - timedelta(days=settings.raw_payload_retention_days)
        result = s.execute(
            delete(RawDataPoint).where(
                RawDataPoint.fetched_at < cutoff,
                RawDataPoint.data_type.notin_(keep_forever | set(_RAW_RETENTION_OVERRIDES)),
            )
        )
        report.raw_deleted += result.rowcount or 0

        hr_cutoff = now - timedelta(days=settings.hr_sample_retention_days)
        report.hr_samples_deleted = prune_samples(s, hr_cutoff)

        # Tombstones for blocks that no longer exist. Block keys carry a date,
        # so any change that moves a block to a different day leaves the old row
        # behind forever - and every one of them is re-read on every sync.
        from circa.db.models import CalendarBlock

        tombstone_cutoff = now - timedelta(days=_TOMBSTONE_RETENTION_DAYS)
        result = s.execute(
            delete(CalendarBlock).where(
                CalendarBlock.deleted.is_(True),
                CalendarBlock.start_ts < tombstone_cutoff,
            )
        )
        report.tombstones_deleted = result.rowcount or 0

    if vacuum:
        # VACUUM cannot run inside a transaction, hence AUTOCOMMIT.
        try:
            from circa.db.session import get_engine

            with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("VACUUM"))
        except Exception as exc:  # noqa: BLE001
            log.warning("retention.vacuum_failed", error=str(exc))

    with session_scope() as s:
        report.bytes_after = _db_bytes(s)

    log.info(
        "retention.done",
        raw_deleted=report.raw_deleted,
        hr_deleted=report.hr_samples_deleted,
        tombstones_deleted=report.tombstones_deleted,
    )
    return report


def _db_bytes(session) -> int | None:
    try:
        page_count = session.execute(text("PRAGMA page_count")).scalar()
        page_size = session.execute(text("PRAGMA page_size")).scalar()
        if page_count and page_size:
            return int(page_count) * int(page_size)
    except Exception:  # noqa: BLE001 - not SQLite, or pragma unavailable
        return None
    return None

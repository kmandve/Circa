"""Timestamp handling.

Every instant in the database is stored as a *naive UTC* datetime and handed
back to Python as an *aware UTC* datetime. SQLite has no timezone type, so
without a decorator like this it is easy to round-trip a local-time datetime and
silently corrupt the record.

That matters more here than in most projects: a one-hour timestamp error is
indistinguishable from a real one-hour circadian phase shift, and a DST
transition would otherwise manufacture one twice a year. Local context that
genuinely matters (what wall-clock time the user experienced) is kept
separately in explicit `tz_name` / `utc_offset_seconds` columns.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator):
    """A DateTime that refuses to store ambiguous (naive) instants."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                f"Refusing to store naive datetime {value!r}. "
                "Attach a timezone (datetime.now(UTC)) before persisting."
            )
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)

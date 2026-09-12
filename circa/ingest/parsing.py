"""Tolerant parsing helpers for Google Health API payloads.

The v4 schemas are still moving, and Google's own documentation disagrees with
itself in places. Rather than hard-code one field layout and lose data when it
shifts, every extraction here tries a list of plausible paths and returns None
rather than raising. The raw payload is always kept, so a parser that guesses
wrong today can be re-run over history tomorrow.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")

# Instants outside these bounds are not representable on every platform and are
# never real health data. `datetime.fromtimestamp` raises OverflowError/OSError
# on them, and because the poller's watermark only advances after a clean pass,
# one absurd point raising here would wedge ingestion permanently rather than
# skipping a single row. Range is year 1 to year 9999.
_MIN_EPOCH_SECONDS = -62135596800
_MAX_EPOCH_SECONDS = 253402300799


def _from_epoch(seconds: Any) -> datetime | None:
    """Epoch seconds -> aware UTC datetime, or None if it cannot be one."""
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not _MIN_EPOCH_SECONDS <= value <= _MAX_EPOCH_SECONDS:
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """int() that returns None instead of raising on inf, NaN or junk."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def deep_get(obj: Any, path: str) -> Any:
    """Follow a dotted path, returning None if any hop is missing."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def first(obj: Any, *paths: str) -> Any:
    """Return the value at the first dotted path that resolves to non-None."""
    for path in paths:
        value = deep_get(obj, path)
        if value is not None:
            return value
    return None


def parse_ts(value: Any, assume_utc: bool = True) -> datetime | None:
    """Parse an API timestamp into an aware UTC datetime.

    Handles RFC3339 with offset or `Z`, offset-free "civil" timestamps, epoch
    seconds/millis, and Google's `{seconds, nanos}` protobuf form.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        if "seconds" in value:
            secs = _as_int(value["seconds"])
            if secs is None:
                return None
            return _from_epoch(secs + (_as_int(value.get("nanos", 0)) or 0) / 1e9)
        # The v4 sample-time object:
        #   {"physicalTime": "...Z", "utcOffset": "-18000s",
        #    "civilTime": {"date": {...}, "time": {...}}}
        # `physicalTime` is the unambiguous instant; civilTime is the same
        # moment expressed in the wall clock the user was living on. An earlier
        # version fell through to civilTime and returned None for every heart
        # rate sample, silently discarding the entire channel.
        if "physicalTime" in value:
            return parse_ts(value["physicalTime"])
        if "civilTime" in value:
            return _parse_civil(value["civilTime"])
        # Daily types carry {"year": .., "month": .., "day": ..}. Without this
        # every daily point stored a NULL timestamp, which then fell outside the
        # poller's window query and was never normalised - the raw arrived but
        # daily_metric stayed empty.
        if "year" in value and "month" in value:
            d = parse_date(value)
            return datetime(d.year, d.month, d.day, tzinfo=UTC) if d else None
        value = first(value, "value", "time", "dateTime")
        if value is None:
            return None

    if isinstance(value, bool):
        # `isinstance(True, int)` is True, so an unguarded bool would parse as
        # epoch second 1 and land in 1970.
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        # Heuristic: anything past ~2001 in ms would be an absurd year in s.
        return _from_epoch(value / 1000 if value > 1e11 else value)

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    # Trim sub-second precision beyond microseconds, which fromisoformat rejects.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if parsed.tzinfo is None:
        if not assume_utc:
            return None
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_civil(obj: Any) -> datetime | None:
    """Parse Google's split civil-time object into a naive-UTC-tagged instant.

    Only used when no `physicalTime` is present; the result is a wall-clock
    reading, so callers that need a true instant should prefer physicalTime.
    """
    if not isinstance(obj, dict):
        return None
    d = obj.get("date") or {}
    t = obj.get("time") or {}
    try:
        return datetime(
            int(d["year"]), int(d["month"]), int(d["day"]),
            int(t.get("hours", 0)), int(t.get("minutes", 0)),
            int(t.get("seconds", 0)), tzinfo=UTC,
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_duration_seconds(value: Any) -> int | None:
    """Parse a protobuf Duration string such as "-18000s" into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _as_int(value)
    if isinstance(value, dict) and "seconds" in value:
        return _as_int(value["seconds"])
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("s"):
            text = text[:-1]
        try:
            return _as_int(float(text))
        except (ValueError, OverflowError):
            return None
    return None


def parse_date(value: Any) -> date | None:
    """Parse a calendar date, including Google's `{year, month, day}` form."""
    if value is None:
        return None
    if isinstance(value, dict) and "year" in value:
        parts = [
            _as_int(value.get(key, default))
            for key, default in (("year", None), ("month", 1), ("day", 1))
        ]
        if any(p is None for p in parts):
            return None
        try:
            return date(*parts)
        except (ValueError, TypeError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    ts = parse_ts(value)
    return ts.date() if ts else None


def parse_local_offset_seconds(value: Any) -> int | None:
    """Extract the *user's local* UTC offset, or None if the value does not carry one.

    Deliberately distinct from `parse_offset_seconds`: a `Z`-suffixed timestamp
    is merely expressed in UTC and says nothing about what time the user's
    clocks read. Treating that as "local offset = 0" would silently place the
    user in London. Callers fall back to the configured home timezone instead,
    which is both correct for a stationary user and honest about what the
    payload actually told us.
    """
    if isinstance(value, str) and value.strip().endswith(("Z", "z")):
        return None
    return parse_offset_seconds(value)


def parse_offset_seconds(value: Any) -> int | None:
    """Extract a UTC offset in seconds from a timestamp string or offset field.

    Note `Z` yields 0, which is literally correct but is rarely what a caller
    wanting *local* time wants - see `parse_local_offset_seconds`.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Could be seconds already, or a protobuf Duration-ish dict handled above.
        return _as_int(value)
    if isinstance(value, dict):
        if "seconds" in value:
            return _as_int(value["seconds"])
        return None
    if isinstance(value, str):
        match = _TZ_SUFFIX.search(value.strip())
        if not match:
            return None
        token = match.group(1)
        if token == "Z":
            return 0
        token = token.replace(":", "")
        sign = 1 if token[0] == "+" else -1
        return sign * (int(token[1:3]) * 3600 + int(token[3:5]) * 60)
    return None


# --- domain-shaped extractors ---------------------------------------------


def extract_point_time(point: dict, field: str) -> datetime | None:
    """Best-effort "when did this happen" for a data point of any kind."""
    return parse_ts(
        first(
            point,
            f"{field}.sampleTime.physicalTime",
            f"{field}.sampleTime",
            f"{field}.interval.startTime",
            f"{field}.interval.civilStartTime",
            f"{field}.date",
            f"{field}.startTime",
            "sampleTime",
            "interval.startTime",
            "interval.civilStartTime",
            "startTime",
            "time",
            "date",
            "createTime",
        )
    )


def extract_interval(obj: dict) -> tuple[datetime | None, datetime | None, int | None]:
    """Return (start_utc, end_utc, utc_offset_seconds) from an interval object."""
    if not isinstance(obj, dict):
        return None, None, None
    raw_start = first(obj, "startTime", "civilStartTime", "start")
    raw_end = first(obj, "endTime", "civilEndTime", "end")
    start = parse_ts(raw_start)
    end = parse_ts(raw_end)

    # v4 carries the user's local offset as a Duration string alongside a
    # Z-suffixed instant, e.g. {"startTime": "...Z", "startUtcOffset": "-18000s"}.
    offset = parse_duration_seconds(
        first(obj, "startUtcOffset", "utcOffset", "endUtcOffset")
    )
    if offset is None:
        offset = parse_local_offset_seconds(raw_start)
    if offset is None:
        offset = parse_offset_seconds(first(obj, "startZoneOffset", "zoneOffset"))

    # Duration-only intervals (start + length) do occur.
    if start and not end:
        # as_float, not float(): a malformed duration on one sleep *stage* used
        # to raise out of normalisation and lose the entire night, not just the
        # stage it came from.
        value = as_float(first(obj, "duration.seconds", "durationSeconds", "durationMillis"))
        if value is not None:
            if abs(value) > 1e6:  # millis
                value /= 1000
            if 0 <= value <= 366 * 86400:
                end = start + timedelta(seconds=value)
    return start, end, offset


def as_float(value: Any) -> float | None:
    """Coerce to float, treating non-finite values as absent.

    Google emits the literal string "NaN" for derived fields it cannot compute
    yet - `baselineTemperatureCelsius` is "NaN" until ~30 nights exist. Passing
    that through as a float nan would silently poison every downstream mean,
    comparison and threshold, so it is treated as missing.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, dict):
        return as_float(first(value, "value", "magnitude", "quantity"))
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except (OverflowError, ValueError):
            return None
        return result if math.isfinite(result) else None
    return None

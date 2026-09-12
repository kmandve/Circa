"""One definition of what a clock time looks like in Circa.

Twelve-hour, everywhere a person reads it: the page, the calendar events, the
CLI. Wire formats and log timestamps are not clock times and are not touched.

`%-I` is a glibc/BSD extension rather than C89, so the hour is built by hand -
this has to work the same on every machine the app might be run on, and a
`ValueError` deep inside a calendar description is a miserable way to find out
it does not.
"""

from __future__ import annotations

from datetime import datetime

MERIDIEM = ("am", "pm")


def clock(hour: int, minute: int) -> str:
    """"14:05" -> "2:05pm". Accepts any hour; wraps at 24."""
    hour %= 24
    suffix = MERIDIEM[hour >= 12]
    display = hour % 12 or 12
    return f"{display}:{minute:02d}{suffix}"


def clock_from(ts: datetime) -> str:
    """The wall clock of an already-localised datetime."""
    return clock(ts.hour, ts.minute)


def clock_from_hours(hours: float) -> str:
    """A decimal local hour - 13.5 -> "1:30pm". Rounds to the nearest minute."""
    minutes = round((hours % 24) * 60)
    return clock(minutes // 60, minutes % 60)


def stamp(ts: datetime) -> str:
    """A date and a time, for status output. "2026-09-12 3:28pm"."""
    return f"{ts:%Y-%m-%d} {clock_from(ts)}"

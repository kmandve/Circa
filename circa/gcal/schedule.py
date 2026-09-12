"""Read the user's existing calendar to tell forced wakes from natural ones.

This is the single highest-value use of calendar read access. A 06:30 wake
before a 07:00 meeting says almost nothing about the endogenous clock; the same
wake on a day with nothing scheduled says a great deal. Weighting them equally
is one of the main reasons sleep-timing phase estimates degrade.

Only busy/free times and event start times are used — never event content.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import CalendarLink, SleepSession
from circa.gcal.client import CalendarClient, CalendarError
from circa.settings_store import RuntimeSettings

log = structlog.get_logger(__name__)

# A commitment starting within this long after wake is treated as the thing
# that caused the wake.
FORCED_WINDOW_MINUTES = 150

# How long before the wake a commitment may start and still plausibly be the
# reason for it - you can wake up slightly late for something.
COMMITMENT_GRACE_MINUTES = 60

# How far below the configured early hour a wake counts as alarm-driven on its
# own, with nothing on the calendar.
EARLY_WAKE_MARGIN_HOURS = 3


def _circa_calendar_ids(session: Session) -> set[str]:
    return {
        link.calendar_id
        for link in session.scalars(select(CalendarLink))
        if link.calendar_id
    }


def fetch_busy(
    client: CalendarClient,
    session: Session,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Busy intervals on the user's own calendars, excluding Circa's own."""
    own = _circa_calendar_ids(session)
    session.rollback()
    try:
        calendars = client.list_calendars()
    except CalendarError as exc:
        log.warning("schedule.list_failed", error=str(exc))
        return []

    ids = [
        c["id"]
        for c in calendars
        if c["id"] not in own
        # Circa's own events are transparent anyway, but skipping the
        # calendars entirely avoids any chance of self-reference.
        and c.get("selected", True)
        and not str(c.get("summary", "")).startswith("Circa ·")
    ]
    if not ids:
        return []

    busy: list[tuple[datetime, datetime]] = []
    try:
        data = client.freebusy(ids, start.isoformat(), end.isoformat())
    except CalendarError as exc:
        log.warning("schedule.freebusy_failed", error=str(exc))
        return []

    for entry in data.get("calendars", {}).values():
        for period in entry.get("busy", []):
            try:
                busy.append(
                    (
                        datetime.fromisoformat(period["start"].replace("Z", "+00:00")),
                        datetime.fromisoformat(period["end"].replace("Z", "+00:00")),
                    )
                )
            except (KeyError, ValueError):
                continue
    return sorted(busy)


def mark_forced_wakes(
    session: Session,
    settings: RuntimeSettings,
    client: CalendarClient | None = None,
    lookback_days: int = 45,
    use_calendar: bool = True,
) -> int:
    """Set `wake_forced` on recent sleep sessions. Returns rows updated.

    With `use_calendar=False` only the local early-hour heuristic is applied and
    no network call is made. That path matters: `wake_forced` starts as NULL,
    and `free_wake_probe` - the strongest passive phase anchor there is -
    excludes NULL rather than assuming a wake was free. Running this only when
    pushing to the calendar meant that any run without calendar access left
    every night permanently unclassified and the probe permanently empty.
    """
    config = get_settings()
    tz = ZoneInfo(config.timezone)
    now = datetime.now(UTC)
    start = now - timedelta(days=lookback_days)

    busy: list[tuple[datetime, datetime]] = []
    if use_calendar:
        owns = client is None
        client = client or CalendarClient()
        try:
            # Release any read lock before the HTTP calls - the OAuth layer may
            # need to write a refreshed token on its own connection, and SQLite
            # allows only one writer.
            session.rollback()
            busy = fetch_busy(client, session, start, now + timedelta(days=2))
        finally:
            if owns:
                client.close()

    updated = 0
    sessions = session.scalars(
        select(SleepSession).where(
            SleepSession.end_ts >= start,
            SleepSession.is_main_sleep.is_(True),
        )
    )
    for sleep in sessions:
        wake = sleep.end_ts
        window_start = wake - timedelta(minutes=COMMITMENT_GRACE_MINUTES)
        window_end = wake + timedelta(minutes=FORCED_WINDOW_MINUTES)
        # Test where the commitment *starts*, not merely whether it overlaps.
        # An all-day or multi-day event - "Vacation", "Out of office" - is busy
        # across the whole wake and used to mark every one of those wakes as
        # alarm-driven. Those are precisely the days when the wake is least
        # forced, so it inverted the signal on the most valuable nights the
        # sleep channel has. Something that began at midnight did not wake you
        # at 09:30.
        has_commitment = any(
            window_start <= b_start < window_end for b_start, _b_end in busy
        )

        # Even with an empty calendar, waking this far before the configured
        # early hour usually means an alarm.
        local_wake = wake.astimezone(tz)
        early_hour = max(settings.forced_wake_before_hour - EARLY_WAKE_MARGIN_HOURS, 0)
        early = local_wake.time() < time(hour=early_hour)

        forced = bool(has_commitment or early)
        if not use_calendar and sleep.wake_forced is not None:
            # The offline heuristic only fills in nights nothing has classified
            # yet. It cannot see commitments, so letting it write over a
            # conclusion that *was* drawn from the calendar silently downgrades
            # real evidence to a guess - and every alarm-driven wake after a
            # meeting would quietly become a free wake, which is the strongest
            # phase anchor there is.
            continue
        if sleep.wake_forced != forced:
            sleep.wake_forced = forced
            updated += 1

    log.info("schedule.forced_wakes_marked", updated=updated, busy_intervals=len(busy))
    return updated


def free_intervals(
    busy: list[tuple[datetime, datetime]], start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Complement of `busy` within [start, end) — where focus blocks can land."""
    free: list[tuple[datetime, datetime]] = []
    cursor = start
    for b_start, b_end in sorted(busy):
        if b_end <= cursor:
            continue
        if b_start > cursor:
            free.append((cursor, min(b_start, end)))
        cursor = max(cursor, b_end)
        if cursor >= end:
            break
    if cursor < end:
        free.append((cursor, end))
    return [(s, e) for s, e in free if e > s]

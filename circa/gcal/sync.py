"""Provision the five calendars and push blocks idempotently.

The idempotency rule matters more than it looks. Circa recomputes on every poll,
so a naive delete-and-recreate would churn event IDs and fire a notification
storm many times a day. Instead each event carries its `block_key` in
`extendedProperties.private`, and sync *patches* the existing event in place —
so an unchanged block produces no calendar activity at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import CalendarBlock, CalendarLink, SleepSession
from circa.gcal.blocks import (
    CATEGORY_EVENT_COLOUR,
    CATEGORY_META,
    CATEGORY_ORDER,
    CATEGORY_RGB,
    RETROSPECTIVE_KINDS,
    Block,
)
from circa.gcal.client import CalendarClient, CalendarError
from circa.settings_store import RuntimeSettings

log = structlog.get_logger(__name__)

# Marks every event Circa creates, so its own events can be found without
# touching anything else on the account.
APP_TAG = "circa"


@dataclass
class SyncReport:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    calendars_created: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return self.created + self.updated + self.deleted


def _enabled_categories(settings: RuntimeSettings) -> set[str]:
    from circa.gcal.blocks import BODY, DEBUG, FOCUS, LIGHT, SLEEP

    enabled = set()
    if settings.enable_focus:
        enabled.add(FOCUS)
    if settings.enable_sleep_blocks:
        enabled.add(SLEEP)
    if settings.enable_light:
        enabled.add(LIGHT)
    if any(
        (settings.enable_caffeine_cutoff, settings.enable_workout_window,
         settings.enable_last_meal)
    ):
        enabled.add(BODY)
    if settings.enable_debug_calendar:
        enabled.add(DEBUG)
    return enabled


# Scopes that permit writing the user's calendarList entry, which is where a
# calendar's own colour lives. `calendar.app.created` is not one of them.
_CALENDAR_LIST_WRITE_SCOPES = (
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.calendarlist",
)


def calendar_colours_permitted() -> bool:
    try:
        from circa.ingest.oauth import token_status

        granted = set(token_status("calendar").get("scopes") or [])
    except Exception:  # noqa: BLE001
        return False
    return any(scope in granted for scope in _CALENDAR_LIST_WRITE_SCOPES)


def provision(
    session: Session, client: CalendarClient, settings: RuntimeSettings
) -> dict[str, str]:
    """Ensure a Google Calendar exists for each enabled category.

    Returns {category: calendar_id}. Existing calendars are reused by summary
    so re-running never creates duplicates.

    Note the session is used only for short reads and writes either side of the
    HTTP calls; see the module docstring on why no transaction spans them.
    """
    config = get_settings()
    wanted = _enabled_categories(settings)

    existing_local: dict[str, str] = {}
    applied_colour: dict[str, str | None] = {}
    for link in session.scalars(select(CalendarLink)):
        existing_local[link.category] = link.calendar_id
        applied_colour[link.category] = link.color_id
    session.rollback()  # release any read lock before going to the network

    remote = {c.get("summary"): c for c in client.list_calendars()}

    mapping: dict[str, str] = {}
    created: dict[str, tuple[str, str]] = {}

    for category in CATEGORY_ORDER:
        summary, colour, _description = CATEGORY_META[category]
        if category not in wanted:
            continue
        calendar_id = existing_local.get(category)
        if calendar_id:
            mapping[category] = calendar_id
            continue
        match = remote.get(summary)
        if match is not None:
            mapping[category] = match["id"]
        else:
            new = client.create_calendar(
                summary, CATEGORY_META[category][2], config.timezone
            )
            mapping[category] = new["id"]
            created[category] = (new["id"], colour)
            log.info("gcal.calendar_created", category=category, summary=summary)

    # Recolour whenever the configured colour differs from the one last applied,
    # not only on creation. Otherwise a corrected palette never reaches the
    # calendars that already exist - which is every calendar the user has.
    recolour: dict[str, str] = {cat: CATEGORY_RGB[cat][0] for cat in created}
    for category in mapping:
        if applied_colour.get(category) != CATEGORY_RGB[category][0]:
            recolour[category] = CATEGORY_RGB[category][0]

    coloured: set[str] = set()
    rgb_ok = calendar_colours_permitted()
    if recolour and not rgb_ok:
        # Skip rather than fail once per calendar per poll. Event colours carry
        # the palette instead; this is only the sidebar swatch.
        log.debug("gcal.calendar_colour_not_permitted", categories=sorted(recolour))
        recolour = {}
    for category in recolour:
        calendar_id = mapping.get(category)
        if calendar_id is None:
            continue
        background, foreground = CATEGORY_RGB[category]
        try:
            client.set_calendar_rgb(calendar_id, background, foreground)
            coloured.add(category)
            log.info("gcal.calendar_recoloured", category=category, colour=background)
        except CalendarError as exc:
            # Google's own sentence, not just the status. "failed (400)" told us
            # nothing; "Invalid foreground color" was the whole answer.
            log.warning(
                "gcal.colour_failed", category=category,
                error=str(exc), reason=exc.reason,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("gcal.colour_failed", category=category, error=str(exc))

    # Now persist, with no network call inside the transaction.
    for category in CATEGORY_ORDER:
        summary, colour, _ = CATEGORY_META[category]
        link = session.get(CalendarLink, category)
        if link is None:
            link = CalendarLink(category=category, summary=summary, color_id=colour)
            session.add(link)
        link.enabled = category in wanted
        link.summary = summary
        colour = CATEGORY_RGB[category][0]
        if category in mapping:
            link.calendar_id = mapping[category]
        # Only record the colour once Google has actually accepted it, so a
        # failed patch is retried on the next run instead of being forgotten.
        if category in coloured:
            link.color_id = colour
        if category in created:
            link.created_at = datetime.now(UTC)
    session.flush()
    return mapping


# Google answers a delete for an event that is already gone with 404, and one
# that is already cancelled with 410. Both mean the goal state has been reached.
# Treating them as failures left the local row marked live, so every subsequent
# run retried the same delete and logged the same error - sixteen of them per
# poll, permanently.
_ALREADY_GONE = {404, 410}


def _delete_event(client, calendar_id: str, event_id: str) -> None:
    try:
        client.delete_event(calendar_id, event_id)
    except CalendarError as exc:
        if exc.status not in _ALREADY_GONE:
            raise
        log.debug("gcal.already_deleted", event_id=event_id, status=exc.status)


def _event_colour(block: Block, calendars_are_coloured: bool) -> str | None:
    """Which colour an event should carry, or None to inherit the calendar's.

    An event colorId always wins over the calendar it sits on, so leaving one in
    place would drag the palette back to Google's eleven presets even once the
    calendars themselves carry real hex.
    """
    if calendars_are_coloured:
        return None
    return CATEGORY_EVENT_COLOUR.get(block.category, "8")


def _event_body(
    block: Block, timezone: str, model_version: str, revision: int,
    use_calendar_colour: bool = False,
) -> dict:
    body: dict = {
        "summary": block.title,
        "description": block.description,
        "start": {"dateTime": block.start.isoformat(), "timeZone": timezone},
        "end": {"dateTime": block.end.isoformat(), "timeZone": timezone},
        "transparency": "transparent",  # never mark the user busy
        # Explicitly null when the calendars carry real colours, so the event
        # inherits them. An event colorId overrides the calendar it sits on and
        # would drag the palette back to Google's eleven presets.
        #
        # It has to be null rather than absent: a PATCH leaves out what it does
        # not mention, so omitting the field left every existing event on the
        # colour it already had - while the local row recorded it as cleared,
        # which made the next poll call it unchanged. Verified against the live
        # API: omitted keeps "4", null clears it.
        "colorId": (
            None
            if use_calendar_colour
            else CATEGORY_EVENT_COLOUR.get(block.category, "8")
        ),
        "extendedProperties": {
            "private": {
                "app": APP_TAG,
                "block_key": block.key,
                "kind": block.kind,
                "model_version": model_version,
                "forecast_revision": str(revision),
            }
        },
    }
    if block.notify_minutes is None:
        body["reminders"] = {"useDefault": False, "overrides": []}
    else:
        body["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": block.notify_minutes}],
        }
    return body


# How long a logged wake can be before the calendar stops trusting it as the
# start of "today". A watch left on the nightstand must not freeze the calendar
# on a day that ended two days ago.
CIRCADIAN_DAY_STALE_HOURS = 36


def circadian_day(session: Session, now: datetime, offset: int) -> date:
    """The local date of the wake that began the day you are currently in.

    A biological day runs wake to wake, not midnight to midnight - which is why
    tonight's sleep window, wind-down and melatonin marker all belong to the day
    you woke up on, not the one you fall asleep into.
    """
    last_wake = session.scalar(
        select(func.max(SleepSession.end_ts)).where(
            SleepSession.is_main_sleep.is_(True),
            SleepSession.excluded.is_(False),
        )
    )
    if last_wake is not None:
        if last_wake.tzinfo is None:
            last_wake = last_wake.replace(tzinfo=UTC)
        age = now - last_wake
        if timedelta(0) <= age <= timedelta(hours=CIRCADIAN_DAY_STALE_HOURS):
            return (last_wake + timedelta(seconds=offset)).date()
    # Nothing recent to anchor on - the watch was not worn, or is not synced
    # yet. Fall back to the local calendar day so the calendar keeps turning
    # over instead of stalling on a day that is finished.
    return (now + timedelta(seconds=offset)).date()


def push(
    session: Session,
    blocks: list[Block],
    settings: RuntimeSettings,
    client: CalendarClient | None = None,
    model_version: str = "0.2.0",
    horizon_hours: int | None = None,
) -> SyncReport:
    """Reconcile Circa's calendars with `blocks` over the forecast horizon."""
    report = SyncReport()
    config = get_settings()
    horizon = horizon_hours or settings.forecast_horizon_hours
    now = datetime.now(UTC)
    window_end = now + timedelta(hours=horizon)

    owns_client = client is None
    client = client or CalendarClient()
    try:
        mapping = provision(session, client, settings)
        rgb_ok = calendar_colours_permitted()

        # Forecasts are only written ahead of now; records of what actually
        # happened are written behind it, back to the history horizon. That
        # asymmetry is the whole point: a prediction for a night that has
        # already passed is clutter, but a record of it is the half of the
        # picture the calendar was missing entirely.
        history_start = now - timedelta(days=max(settings.history_days, 0))
        offset = int(
            now.astimezone(ZoneInfo(config.timezone)).utcoffset().total_seconds()
        )
        today = circadian_day(session, now, offset)

        # The calendar holds exactly one circadian day: the one you woke into.
        # Blocks stay put once written, whether or not they have happened yet -
        # a plan you can look back at is worth more than a tidy calendar, and a
        # block vanishing the minute it ends makes the day unreadable by the
        # evening. They keep being refreshed while they stand, so a revised
        # estimate still moves them.
        #
        # The whole set is replaced at the wake that starts the next day, which
        # is the only boundary that means anything: the previous day's plan goes
        # and the new one appears together, once there is real data to build it
        # from. Midnight is not that boundary; waking up is.
        def _in_window(block: Block) -> bool:
            if block.kind in RETROSPECTIVE_KINDS:
                return block.start >= history_start
            if block.start > window_end:
                return False
            if block.day is None:
                return True          # not day-scoped; the rule does not apply
            return block.day == today

        wanted = {b.key: b for b in blocks if _in_window(b)}

        # --- read current state, then release the lock --------------------
        stored: dict[str, dict] = {}
        for row in session.scalars(
            select(CalendarBlock).where(
                CalendarBlock.start_ts >= min(history_start, now - timedelta(days=2)),
                CalendarBlock.start_ts <= window_end,
            )
        ):
            stored[row.block_key] = {
                "category": row.category,
                "kind": row.kind,
                "google_event_id": row.google_event_id,
                "start_ts": row.start_ts,
                "end_ts": row.end_ts,
                "title": row.title,
                "description": row.description,
                "forecast_revision": row.forecast_revision,
                "deleted": row.deleted,
                "notify_minutes": row.notify_minutes,
                "event_color_id": row.event_color_id,
            }
        disabled_links = [
            (link.category, link.calendar_id)
            for link in session.scalars(select(CalendarLink))
            if link.calendar_id and link.category not in _enabled_categories(settings)
        ]
        disabled_blocks: list[tuple[str, str, str]] = []
        for category, calendar_id in disabled_links:
            for row in session.scalars(
                select(CalendarBlock).where(
                    CalendarBlock.category == category,
                    CalendarBlock.deleted.is_(False),
                    CalendarBlock.start_ts >= now,
                )
            ):
                if row.google_event_id:
                    disabled_blocks.append((row.block_key, calendar_id, row.google_event_id))

        session.commit()  # no write lock held while we talk to Google

        # --- network phase, accumulating results in memory -----------------
        writes: dict[str, dict] = {}
        deletes: list[str] = []

        for key, block in wanted.items():
            calendar_id = mapping.get(block.category)
            if calendar_id is None:
                continue
            existing = stored.get(key)

            if (
                existing
                and existing["google_event_id"]
                and _unchanged(
                    existing, block, settings.round_to_minutes,
                    _event_colour(block, rgb_ok),
                )
            ):
                report.unchanged += 1
                continue

            revision = (existing["forecast_revision"] + 1) if existing else 0
            body = _event_body(
                block, config.timezone, model_version, revision, rgb_ok
            )
            # A previously deleted event is dead, not dormant. Google accepts a
            # PATCH against a cancelled event and returns 200 without reviving
            # it, so patching one silently marks the block live locally while
            # nothing appears on the calendar - and the next poll then sees it
            # as unchanged and never sends anything again. Any block that was
            # ever dropped and later came back was gone permanently.
            event_id = (
                existing["google_event_id"]
                if existing and not existing["deleted"]
                else None
            )

            # Every failure here is recorded and skipped rather than raised.
            # `writes` is only flushed to the database after this loop, so an
            # exception escaping would strand events that Google has already
            # created - and the next poll would insert them a second time,
            # putting visible duplicates on a real calendar.
            try:
                if event_id:
                    patched = client.patch_event(calendar_id, event_id, body)
                    if isinstance(patched, dict) and patched.get("status") == "cancelled":
                        # Belt and braces: the event was cancelled behind our
                        # back, and the patch did not bring it back.
                        event_id = client.insert_event(calendar_id, body)["id"]
                        report.created += 1
                    else:
                        report.updated += 1
                else:
                    event_id = client.insert_event(calendar_id, body)["id"]
                    report.created += 1
            except Exception as exc:  # noqa: BLE001
                recreate = (
                    isinstance(exc, CalendarError) and exc.status == 404 and event_id
                )
                if not recreate:
                    report.errors.append(f"{key}: {exc}")
                    continue
                # Deleted in the UI; recreate it. This can fail in turn, and an
                # exception raised inside an except block is not caught by it.
                try:
                    event_id = client.insert_event(calendar_id, body)["id"]
                    report.created += 1
                except Exception as retry_exc:  # noqa: BLE001
                    report.errors.append(f"{key}: recreate failed: {retry_exc}")
                    continue

            writes[key] = {"block": block, "event_id": event_id, "revision": revision}

        # A recorded night replaces the forecast written for it. Matched by
        # overlap, not by key: the forecast's key is dated by the evening its
        # DLMO fell in, which is not the date the night ended on.
        superseded: set[str] = set()
        for block in wanted.values():
            if not block.supersedes_kinds:
                continue
            for key, meta in stored.items():
                if key in wanted or meta["deleted"] or not meta["google_event_id"]:
                    continue
                if meta["kind"] not in block.supersedes_kinds:
                    continue
                if meta["start_ts"] < block.end and meta["end_ts"] > block.start:
                    superseded.add(key)

        # A forecast that has already happened is spent. Keeping it left the
        # calendar silting up with predictions nobody can act on - twenty-one of
        # them after a few days - while the useful record of the past (what you
        # actually slept) is written separately and kept deliberately. So the
        # calendar reads as a record behind you and a plan ahead of you, with
        # nothing stale in between.
        #
        # Guarded on the run having produced something: if block generation
        # failed and `wanted` came back empty, this would otherwise wipe the
        # calendar rather than leave it as it was.
        prune_past = bool(wanted)

        for key, meta in stored.items():
            if key in wanted or meta["deleted"] or not meta["google_event_id"]:
                continue
            finished = meta["end_ts"] <= now
            if finished and not prune_past:
                continue
            if not finished and meta["start_ts"] < now and key not in superseded:
                # In progress and no longer forecast: deleting it out from under
                # the user mid-block is worse than letting it finish.
                continue
            calendar_id = mapping.get(meta["category"])
            if calendar_id is None:
                continue
            try:
                _delete_event(client, calendar_id, meta["google_event_id"])
                deletes.append(key)
                report.deleted += 1
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"delete {key}: {exc}")

        for key, calendar_id, event_id in disabled_blocks:
            try:
                _delete_event(client, calendar_id, event_id)
                deletes.append(key)
                report.deleted += 1
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"disable {key}: {exc}")

        # --- write results back -------------------------------------------
        for key, meta in writes.items():
            block = meta["block"]
            row = session.scalar(
                select(CalendarBlock).where(CalendarBlock.block_key == key)
            )
            if row is None:
                row = CalendarBlock(block_key=key)
                session.add(row)
            row.google_event_id = meta["event_id"]
            row.category = block.category
            row.kind = block.kind
            row.target_date = block.start.date()
            row.start_ts = block.start
            row.end_ts = block.end
            row.title = block.title
            row.description = block.description
            row.notify_minutes = block.notify_minutes
            row.event_color_id = _event_colour(block, rgb_ok)
            row.forecast_revision = meta["revision"]
            row.model_version = model_version
            row.synced_at = datetime.now(UTC)
            row.deleted = False

        for key in deletes:
            row = session.scalar(
                select(CalendarBlock).where(CalendarBlock.block_key == key)
            )
            if row is not None:
                row.deleted = True
        session.flush()
    finally:
        if owns_client:
            client.close()

    log.info(
        "gcal.pushed",
        created=report.created, updated=report.updated,
        unchanged=report.unchanged, deleted=report.deleted,
        errors=len(report.errors),
    )
    return report


def _within(a: datetime, b: datetime, tolerance: timedelta) -> bool:
    return abs(a - b) <= tolerance


def _unchanged(
    stored: dict, block: Block, tolerance_minutes: int = 0,
    expected_colour: str | None = None,
) -> bool:
    """True when the stored block is close enough that rewriting it is noise.

    Times are compared with a tolerance rather than exactly. The model moves by
    seconds between runs - the light window ends at "now", so its bin count
    shifts - and once a boundary is rounded to the display quantum, a drift of
    seconds can flip it by a whole quantum. That rewrote calendar entries on
    every poll for changes far below the precision the block itself claims
    (+/-40 minutes). A real shift still exceeds the tolerance and still lands.
    """
    tolerance = timedelta(minutes=tolerance_minutes)
    return (
        _within(stored["start_ts"], block.start, tolerance)
        and _within(stored["end_ts"], block.end, tolerance)
        and stored["title"] == block.title
        and stored["description"] == block.description
        # Reminders are part of the event body, so a change of notification
        # policy has to count as a change - otherwise the new setting is
        # computed, stored, and never sent.
        and stored.get("notify_minutes") == block.notify_minutes
        # The colour is in the event body too. Leaving it out of the comparison
        # would mean a corrected palette never reached a single existing event -
        # exactly how the reminder setting went missing.
        and stored.get("event_color_id") == expected_colour
        and not stored["deleted"]
    )

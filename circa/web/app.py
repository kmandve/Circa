"""FastAPI application + background scheduler.

Four pages: Today (what to do now), Trends (is it working), Settings (change
anything without touching code), Model (is it any good). Binds to localhost by
default — there is no authentication and this database holds detailed sleep and
cardiovascular data, so it is reached over an SSH tunnel rather than an open
port.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select

from circa import forecast_store
from circa.alertness.process_s import recent_sleep_debt
from circa.config import get_settings
from circa.db.models import (
    CalendarBlock,
    CalendarLink,
    DailyMetric,
    DeviceSync,
    HeartRateMinute,
    PhaseEstimate,
    RawDataPoint,
    SleepSession,
    SyncState,
)
from circa.db.session import session_scope
from circa.gcal.blocks import RETROSPECTIVE_KINDS
from circa.gcal.sync import calendar_colours_permitted
from circa.settings_store import (
    Chronotype,
    LowConfidencePolicy,
    NotificationPolicy,
    RuntimeSettings,
    load_settings,
    save_settings,
)
from circa.timefmt import clock_from, clock_from_hours, stamp

log = structlog.get_logger(__name__)

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
# Clock times are twelve-hour throughout. Registered as a filter so a
# template can never quietly reinvent a second format.
templates.env.filters["clock_hours"] = clock_from_hours


# ---------------------------------------------------------------------------
# scheduled jobs
# ---------------------------------------------------------------------------


def _scheduled_sync() -> None:
    from circa.ingest.poller import sync_once

    try:
        report = sync_once()
    except Exception as exc:  # noqa: BLE001 - a failed poll must not kill the scheduler
        log.error("scheduler.sync_failed", error=str(exc), exc_info=True)
        return

    # Only re-run the model when new data actually landed.
    if report.total_inserted > 0:
        _scheduled_pipeline()


# One model run at a time, process-wide. The scheduler's job and a Refresh from
# the browser are separate threads that both write, and SQLite allows a single
# writer - two at once is how "database is locked" happens.
_PIPELINE_LOCK = threading.Lock()


def _scheduled_pipeline() -> bool:
    """Run the model unless one is already running. True if this call ran it."""
    from circa.pipeline import run

    if not _PIPELINE_LOCK.acquire(blocking=False):
        log.info("scheduler.pipeline_already_running")
        return False
    try:
        with session_scope() as session:
            run(session)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("scheduler.pipeline_failed", error=str(exc), exc_info=True)
        return False
    finally:
        _PIPELINE_LOCK.release()


def _scheduled_retention() -> None:
    from circa.ingest.retention import run_retention

    try:
        run_retention(vacuum=True)
    except Exception as exc:  # noqa: BLE001
        log.error("scheduler.retention_failed", error=str(exc), exc_info=True)


def create_app(with_scheduler: bool = True) -> FastAPI:
    config = get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler: BackgroundScheduler | None = None
        if with_scheduler:
            scheduler = BackgroundScheduler(timezone="UTC")
            scheduler.add_job(
                _scheduled_sync,
                IntervalTrigger(minutes=config.poll_interval_minutes),
                id="sync", max_instances=1, coalesce=True,
                next_run_time=datetime.now(UTC) + timedelta(seconds=15),
            )
            # A daily model run even if no new data arrived, so the forecast
            # horizon keeps rolling forward.
            scheduler.add_job(
                _scheduled_pipeline,
                CronTrigger(hour="*/6", minute=7),
                id="pipeline", max_instances=1, coalesce=True,
            )
            scheduler.add_job(
                _scheduled_retention,
                CronTrigger(hour=4, minute=17),
                id="retention", max_instances=1, coalesce=True,
            )
            scheduler.start()
            log.info("scheduler.started", interval_minutes=config.poll_interval_minutes)
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(title="Circa", docs_url=None, redoc_url=None, lifespan=lifespan)
    static_dir = _HERE / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -- pages -------------------------------------------------------------

    @app.get("/")
    def today(request: Request):
        return templates.TemplateResponse(request, "today.html", _today_context())

    @app.get("/trends")
    def trends(request: Request):
        return templates.TemplateResponse(request, "trends.html", _trends_context())

    @app.get("/settings")
    def settings_page(request: Request):
        with session_scope() as s:
            current = load_settings(s)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                **_nav_context(),
                "s": current,
                "chronotypes": list(Chronotype),
                "low_conf": list(LowConfidencePolicy),
                "notify": list(NotificationPolicy),
                "config": get_settings(),
            },
        )

    @app.post("/settings")
    async def save_settings_route(request: Request):
        form = await request.form()
        with session_scope() as s:
            current = load_settings(s)
            data = current.model_dump()
            unparseable: list[str] = []
            for key, value in form.items():
                if key not in data:
                    continue
                try:
                    data[key] = _coerce(current, key, value)
                except (TypeError, ValueError):
                    unparseable.append(key)
            if unparseable:
                log.warning("settings.unparseable", fields=unparseable)
                return RedirectResponse(
                    f"/settings?invalid={quote(', '.join(unparseable))}",
                    status_code=303,
                )
            # Unchecked checkboxes are simply absent from the form body.
            for key in data:
                if isinstance(getattr(current, key, None), bool) and key not in form:
                    data[key] = False
            try:
                save_settings(s, RuntimeSettings.model_validate(data))
            except ValidationError as exc:
                # Reporting this matters: the whole submission is discarded, so
                # silently redirecting to "?saved=1" told the user their changes
                # had been applied when none of them had.
                log.warning("settings.invalid", error=str(exc))
                fields = ", ".join(
                    str(err["loc"][0]) for err in exc.errors() if err.get("loc")
                )
                return RedirectResponse(
                    f"/settings?invalid={quote(fields or 'unknown')}", status_code=303
                )
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/model")
    def model_page(request: Request):
        return templates.TemplateResponse(request, "model.html", _model_context())

    @app.post("/run")
    def run_now():
        # Fire and redirect. A full run takes about five seconds; holding the
        # request open for it is what made the Refresh button feel broken.
        _request_background_refresh()
        return RedirectResponse("/", status_code=303)

    # -- api ---------------------------------------------------------------

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(_status_payload())

    @app.get("/api/curve")
    def api_curve() -> JSONResponse:
        return JSONResponse(_curve_payload())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


def _coerce(current: RuntimeSettings, key: str, value):
    """Form string -> field type. Raises ValueError if it is not convertible.

    Falling back to the current value here looked harmless but meant a typo in
    a number field was dropped on the floor while the page still reported the
    settings as saved.
    """
    field_value = getattr(current, key)
    if isinstance(field_value, bool):
        return str(value).lower() in {"1", "true", "on", "yes"}
    if isinstance(field_value, int) and not isinstance(field_value, bool):
        return int(float(value))
    if isinstance(field_value, float):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# context builders
# ---------------------------------------------------------------------------


def _nav_context() -> dict:
    return {"now": datetime.now(UTC)}


def _latest_phase(session) -> PhaseEstimate | None:
    from circa.phase.engine import MODEL_VERSION

    return session.scalar(
        select(PhaseEstimate)
        .where(PhaseEstimate.model_version == MODEL_VERSION)
        .order_by(PhaseEstimate.computed_at.desc())
    )


def _fmt_local(ts: datetime | None, offset: int) -> str:
    if ts is None:
        return "—"
    return clock_from(ts + timedelta(seconds=offset))


def _fmt_stamp(ts: datetime | None, offset: int) -> str | None:
    """A date and a twelve-hour time, in the user's timezone."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return stamp(ts + timedelta(seconds=offset))


def _request_background_refresh() -> None:
    """Kick off a model run without making anyone wait for it."""
    global _REFRESH_THREAD
    if _REFRESH_THREAD is not None and _REFRESH_THREAD.is_alive():
        return

    def _work() -> None:
        try:
            _scheduled_pipeline()
        except Exception as exc:  # noqa: BLE001
            log.warning("web.background_refresh_failed", error=str(exc))

    _REFRESH_THREAD = threading.Thread(target=_work, name="circa-refresh", daemon=True)
    _REFRESH_THREAD.start()
    log.info("web.background_refresh_started")


_REFRESH_THREAD = None


def _refresh_in_progress() -> bool:
    return _REFRESH_THREAD is not None and _REFRESH_THREAD.is_alive()


# If there is no night ahead to stop at - the forecast is thin, or sleep blocks
# are switched off - fall back to a fixed window that still reads as "today".
TIMELINE_FALLBACK_HOURS = 18


def _until_tomorrow(rows: list[dict], now, offset: int) -> tuple[list[dict], int]:
    """Trim the timeline to everything up to the end of the night ahead."""
    cutoff = None
    for row in rows:
        if row["kind"] == "sleep_window" and row["end_raw"] > now:
            cutoff = row["end_raw"]
            break
    if cutoff is None:
        cutoff = now + timedelta(hours=TIMELINE_FALLBACK_HOURS)
    kept = [r for r in rows if r["start_raw"] <= cutoff]
    return kept, len(rows) - len(kept)


def _plain_title(title: str, kind: str) -> str:
    """Calendar title with the uncertainty suffix and the emoji removed."""
    from circa.gcal.blocks import KIND_EMOJI

    text = title.split("(")[0].strip()
    emoji = KIND_EMOJI.get(kind, "")
    if emoji:
        text = text.replace(emoji, "")
    return " ".join(text.split())


def _progress(start, end, now) -> int | None:
    """How far through a block we are, 0-100, or None if it is not running."""
    span = (end - start).total_seconds()
    if span <= 0 or not start <= now <= end:
        return None
    return int(round(100 * (now - start).total_seconds() / span))


def _duration_words(hours: float | None) -> str:
    if hours is None:
        return "—"
    total = int(round(abs(hours) * 60))
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}h {m:02d}m"
    return f"{h}h" if h else f"{m}m"


def _debt_words(debt, target_hours: float) -> dict:
    """Sleep debt as a headline, a direction and an honest caveat."""
    if not debt.is_meaningful:
        return {
            "value": "—",
            "tone": "unknown",
            "line": (
                f"Needs {3 - debt.nights} more night"
                f"{'' if 3 - debt.nights == 1 else 's'} before this means anything. "
                "One short night is a short night, not a trend."
            ),
        }
    if debt.hours >= 0.5:
        return {
            "value": _duration_words(debt.hours),
            "tone": "behind",
            "line": (
                f"Short of your {_duration_words(target_hours)} target across the "
                f"last {debt.window_days} days. Tonight's window is stretched to "
                "pay back part of it."
            ),
        }
    if debt.hours <= -0.5:
        return {
            "value": _duration_words(abs(debt.hours)),
            "tone": "ahead",
            "line": f"Ahead of your target over the last {debt.window_days} days.",
        }
    return {
        "value": "Clear",
        "tone": "clear",
        "line": f"Roughly on target across the last {debt.window_days} days.",
    }


def _today_context() -> dict:
    from circa.ingest.oauth import token_status

    config = get_settings()
    now = datetime.now(UTC)
    offset = int(now.astimezone(ZoneInfo(config.timezone)).utcoffset().total_seconds())

    with session_scope() as s:
        phase = _latest_phase(s)
        settings = load_settings(s)
        nights = s.scalar(
            select(func.count(func.distinct(SleepSession.sleep_date)))
            .select_from(SleepSession)
            .where(SleepSession.is_main_sleep.is_(True), SleepSession.excluded.is_(False))
        ) or 0
        debt = recent_sleep_debt(
            s, now, settings.target_sleep_hours, settings.sleep_debt_window_days
        )
        last_device = s.scalar(select(func.max(DeviceSync.last_sync_time)))
        blocks = list(
            s.scalars(
                select(CalendarBlock)
                .where(
                    CalendarBlock.deleted.is_(False),
                    CalendarBlock.end_ts >= now - timedelta(hours=30),
                    CalendarBlock.start_ts <= now + timedelta(hours=48),
                )
                .order_by(CalendarBlock.start_ts)
            )
        )
        block_rows = [
            {
                "category": b.category,
                "kind": b.kind,
                "title": b.title,
                # The calendar needs a glyph to survive a month grid. The page
                # has a real icon next to every row already, so carrying the
                # emoji here as well is just noise.
                "label": _plain_title(b.title, b.kind),
                "start_raw": b.start_ts,
                "end_raw": b.end_ts,
                "start": _fmt_local(b.start_ts, offset),
                "end": _fmt_local(b.end_ts, offset),
                "day": (b.start_ts + timedelta(seconds=offset)).strftime("%a"),
                "active": b.start_ts <= now <= b.end_ts,
                "progress": _progress(b.start_ts, b.end_ts, now),
                "past": b.end_ts < now,
                "recorded": b.kind in RETROSPECTIVE_KINDS,
                "description": b.description or "",
            }
            for b in blocks
        ]
        # "Your day" has to actually be a day. The calendar holds three days of
        # forecast, which is right for the calendar and far too long for a
        # glance - so the list stops at the end of the night ahead, and the
        # chart carries the longer view.
        block_rows, beyond = _until_tomorrow(block_rows, now, offset)

        links = {
            link.category: link for link in s.scalars(select(CalendarLink))
        }
        n_calendars = sum(1 for link in links.values() if link.calendar_id and link.enabled)

    detail = (phase.detail or {}) if phase else {}
    token = token_status()

    # --- what is happening right now --------------------------------------
    active = [b for b in block_rows if b["active"]]
    # The timeline reaches back over recorded nights, so "not active" is no
    # longer the same as "still to come" - without this the hero announced
    # "Coming up: slept 8h 12m" about a night that ended two days ago.
    upcoming = [b for b in block_rows if not b["active"] and not b["past"]]
    now_progress = None
    if active:
        # Prefer the most specific active block over the all-night one.
        primary = min(active, key=lambda b: 0 if b["kind"] != "sleep_window" else 1)
        headline, detail_line = _NOW_HEADLINES.get(
            primary["kind"], (primary["title"], "")
        )
        now_kicker = "Right now"
        now_until = f"until {primary['end']}"
        now_progress = primary["progress"]
    elif upcoming:
        nxt = upcoming[0]
        headline, detail_line = _NOW_HEADLINES.get(nxt["kind"], (nxt["title"], ""))
        headline = f"Next: {headline[0].lower()}{headline[1:]}"
        now_kicker = "Coming up"
        now_until = f"{nxt['day']} {nxt['start']}–{nxt['end']}"
    else:
        headline, detail_line, now_kicker, now_until = (
            "Nothing scheduled", "", "Right now", "",
        )

    debt_words = _debt_words(debt, settings.target_sleep_hours)
    with session_scope() as s:
        stored = forecast_store.load_curve(s)
    if stored is None:
        _request_background_refresh()
    peak = None
    if stored:
        peak = _today_peak(
            stored.payload.get("points", []),
            stored.payload.get("utc_offset_seconds", 0),
        )
        # Before the first wake of the day the curve may hold no awake point
        # dated today at all; the model-wide peak is the honest fallback.
        if peak is None:
            peak = stored.payload.get("energy_peak")
    energy_peak = round(peak) if peak is not None else None
    forecast_age = int(stored.age.total_seconds() // 60) if stored else None
    forecast_stale = stored.is_stale if stored else True
    nights_to_next = _nights_to_next(nights)
    next_threshold = {0: 7, 1: 14, 2: 30}.get(
        phase.confidence_tier if phase else 0
    )
    tier_progress = (
        min(nights / next_threshold, 1.0) if next_threshold else 1.0
    )

    return {
        **_nav_context(),
        "now_headline": headline,
        "now_detail": detail_line,
        "now_kicker": now_kicker,
        "now_until": now_until,
        "now_progress": now_progress,
        "nights_to_next": nights_to_next,
        "tier_progress": round(tier_progress * 100),
        "n_calendars": n_calendars,
        "authenticated": token.get("authenticated", False),
        "testing_status_suspected": token.get("testing_status_suspected", False),
        # The custom calendar palette needs a scope older tokens do not carry.
        # Without this the colours simply never change and nothing says why.
        "calendar_colours_allowed": calendar_colours_permitted(),
        "phase": phase,
        "has_phase": phase is not None,
        "dlmo": _fmt_local(phase.dlmo_ts, offset) if phase else "—",
        "cbtmin": _fmt_local(phase.cbtmin_ts, offset) if phase else "—",
        "ci_low": _fmt_local(phase.dlmo_ci80_low, offset) if phase else "—",
        "ci_high": _fmt_local(phase.dlmo_ci80_high, offset) if phase else "—",
        "sd_minutes": round(phase.phase_sd_minutes) if phase and phase.phase_sd_minutes else None,
        "tier": phase.confidence_tier if phase else 0,
        "tier_name": _tier_name(phase.confidence_tier if phase else 0),
        "quality": round(phase.confidence_q, 2) if phase and phase.confidence_q else None,
        "ood_flags": detail.get("confidence", {}).get("ood_flags", []),
        "tau": detail.get("tau_mean"),
        "nights": nights,
        "blocks": block_rows,
        "blocks_beyond": beyond,
        "debt": debt,
        "debt_words": debt_words,
        "last_night": _duration_words(debt.last_night_hours) if debt.last_night_hours else None,
        "last_night_delta": debt.last_night_delta,
        "target_sleep_words": _duration_words(settings.target_sleep_hours),
        "energy_peak": energy_peak,
        "forecast_age_minutes": forecast_age,
        "forecast_computed_at": stored.computed_at.isoformat() if stored else "",
        "forecast_stale": forecast_stale,
        "refreshing": _refresh_in_progress(),
        "calendars": links,
        "device_sync_age_minutes": (
            round((now - last_device).total_seconds() / 60) if last_device else None
        ),
        "paused": settings.paused,
        "timezone": config.timezone,
    }


# Plain-language headline for whatever block is running now. The page should
# answer "what should I be doing" before it answers "what does the model think".
# Headline and one supporting line per block kind. Names must match the calendar
# titles in `KIND_TITLES` - the page saying "Circadian dip" while the timeline
# directly below it said "Afternoon dip" made them look like two things.
_NOW_HEADLINES = {
    "grogginess":       ("Grogginess", "Sleep inertia. It clears on its own — this is not how your day will go."),
    "peak_focus":       ("Morning peak", "Your sharpest window. Protect it for demanding work."),
    "second_wind":      ("Evening peak", "Real alertness, but leaning into it pushes your clock later."),
    "circadian_dip":    ("Afternoon dip", "Routine or physical tasks land better than hard thinking."),
    "morning_light":    ("Get light", "Outdoors beats indoor lighting by a wide margin."),
    "dim_light":        ("Dim lights", "Bright light from here pushes your clock later."),
    "melatonin_window": ("Melatonin peak", "Sleep comes with the least effort from here — this is your ideal bedtime."),
    "wind_down":        ("Wind-down", "Screens down, lights low."),
    "sleep_window":     ("Sleep", "This is when sleep works best for you."),
    "sleep_actual":     ("Slept", "What your watch actually recorded."),
    "caffeine_cutoff":  ("Last coffee", "After this it is still in your system at bedtime."),
    "last_meal":        ("Last meal", "Finishing now keeps your metabolic clocks aligned."),
    "workout":          ("Workout", "Strength and body temperature peak together."),
}


def _tier_name(tier: int) -> str:
    from circa.phase.confidence import TIER_NAMES

    return TIER_NAMES.get(tier, "unknown")


def _nights_to_next(nights: int) -> int | None:
    """Nights remaining before the next confidence tier unlocks."""
    from circa.phase.confidence import TIER_NIGHTS

    for threshold in sorted(t for t in TIER_NIGHTS.values() if t > 0):
        if nights < threshold:
            return threshold - nights
    return None


def _trends_context() -> dict:
    from circa.phase.engine import MODEL_VERSION
    from circa.phase.sleep_phase import local_hour_of_day

    config = get_settings()
    now = datetime.now(UTC)

    with session_scope() as s:
        nights = list(
            s.scalars(
                select(SleepSession)
                .where(
                    SleepSession.is_main_sleep.is_(True),
                    SleepSession.excluded.is_(False),
                    SleepSession.end_ts >= now - timedelta(days=60),
                )
                .order_by(SleepSession.end_ts)
            )
        )
        sleep_rows = [
            {
                "date": n.sleep_date.isoformat() if n.sleep_date else "",
                "onset": local_hour_of_day(n.start_ts, n.tz_name or "UTC", n.utc_offset_seconds),
                "offset": local_hour_of_day(n.end_ts, n.tz_name or "UTC", n.utc_offset_seconds),
                "midpoint": (
                    local_hour_of_day(n.midpoint_ts, n.tz_name or "UTC", n.utc_offset_seconds)
                    if n.midpoint_ts else None
                ),
                "tst": round((n.tst_minutes or 0) / 60, 2),
                "forced": bool(n.wake_forced),
            }
            for n in nights
        ]

        estimates = list(
            s.scalars(
                select(PhaseEstimate)
                .where(PhaseEstimate.model_version == MODEL_VERSION)
                .order_by(PhaseEstimate.target_date)
            )
        )
        phase_rows = [
            {
                "date": e.target_date.isoformat(),
                "dlmo": (e.detail or {}).get("dlmo_local_hours"),
                "ci": (e.detail or {}).get("ci80_local_hours"),
                "tier": e.confidence_tier,
                "q": e.confidence_q,
            }
            for e in estimates
            if e.detail
        ]

        coverage = []
        for offset_days in range(29, -1, -1):
            day = (now - timedelta(days=offset_days)).date()
            start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            minutes = s.scalar(
                select(func.count()).select_from(HeartRateMinute)
                .where(HeartRateMinute.ts >= start, HeartRateMinute.ts < start + timedelta(days=1))
            ) or 0
            coverage.append({"date": day.isoformat(), "fraction": round(min(minutes / 1440, 1), 3)})

        rhr = [
            {"date": r.metric_date.isoformat(), "value": r.value}
            for r in s.scalars(
                select(DailyMetric)
                .where(
                    DailyMetric.metric == "daily-resting-heart-rate",
                    DailyMetric.metric_date >= (now - timedelta(days=60)).date(),
                )
                .order_by(DailyMetric.metric_date)
            )
            if r.value
        ]

    return {
        **_nav_context(),
        "sleep_rows": sleep_rows,
        "phase_rows": phase_rows,
        "coverage": coverage,
        "rhr": rhr,
        "timezone": config.timezone,
    }


def _model_context() -> dict:
    from circa.phase.engine import MODEL_VERSION
    from circa.validate import backtest

    config = get_settings()
    now = datetime.now(UTC)
    offset = int(now.astimezone(ZoneInfo(config.timezone)).utcoffset().total_seconds())

    with session_scope() as s:
        phase = _latest_phase(s)
        settings = load_settings(s)
        stability = backtest.stability(s)
        states = [
            {
                "data_type": st.data_type,
                # Rendered here rather than in the template: it is the user's
                # own clock like every other time on the site, not a raw UTC
                # ISO string sliced to sixteen characters.
                "watermark": _fmt_stamp(st.watermark, offset),
                "last_success": _fmt_stamp(st.last_success_at, offset),
                "ingested": st.points_ingested or 0,
                "failures": st.consecutive_failures or 0,
                "error": st.last_error,
            }
            for st in s.scalars(select(SyncState).order_by(SyncState.data_type))
        ]
        raw_count = s.scalar(select(func.count()).select_from(RawDataPoint)) or 0

    detail = (phase.detail or {}) if phase else {}
    return {
        **_nav_context(),
        "model_version": MODEL_VERSION,
        "phase": phase,
        "detail": detail,
        "channels": detail.get("channels", {}),
        "filter": detail.get("filter", {}),
        "confidence": detail.get("confidence", {}),
        "stability": stability,
        "sync_states": states,
        "raw_points": raw_count,
        "settings": settings,
    }


def _status_payload() -> dict:
    with session_scope() as s:
        phase = _latest_phase(s)
        nights = s.scalar(
            select(func.count()).select_from(SleepSession)
            .where(SleepSession.is_main_sleep.is_(True))
        ) or 0
    with session_scope() as s:
        stored = forecast_store.load_curve(s)
    return {
        # The page polls this to notice a finished recompute. Cheap on purpose:
        # it must not cost more than the thing it is waiting for.
        "forecast_computed_at": stored.computed_at.isoformat() if stored else None,
        "refreshing": _refresh_in_progress(),
        "nights": nights,
        "tier": phase.confidence_tier if phase else 0,
        "quality": phase.confidence_q if phase else None,
        "dlmo_utc": phase.dlmo_ts.isoformat() if phase and phase.dlmo_ts else None,
        "sd_minutes": phase.phase_sd_minutes if phase else None,
        "computed_at": phase.computed_at.isoformat() if phase else None,
        "detail": (phase.detail or {}) if phase else {},
    }


# How much of the curve is worth looking at. Enough history to see the night
# you just had, and far enough ahead to cover the night to come.
CURVE_VIEW_BEHIND_HOURS = 4
CURVE_VIEW_AHEAD_HOURS = 20


def _visible_window(points: list[dict], offset_seconds: int) -> list[dict]:
    """Slice the stored curve down to the span the chart actually shows.

    Sliced here rather than in the browser so the payload is small and the
    window keeps moving with the clock even if the forecast itself is hours old.
    """
    if not points:
        return points
    now_local = datetime.now(UTC) + timedelta(seconds=offset_seconds)
    lo = (now_local - timedelta(hours=CURVE_VIEW_BEHIND_HOURS)).replace(tzinfo=None)
    hi = (now_local + timedelta(hours=CURVE_VIEW_AHEAD_HOURS)).replace(tzinfo=None)

    def at(point: dict) -> datetime:
        return datetime.fromisoformat(point["t"]).replace(tzinfo=None)

    window = [p for p in points if lo <= at(p) <= hi]
    # A forecast old enough that none of it is still in view is better shown in
    # full than not at all.
    return window or points


def _today_peak(points: list[dict], offset_seconds: int) -> float | None:
    """The best energy the model expects *today*, awake.

    The curve itself spans roughly two days, so its own maximum is not an
    answer to "how good does today get" - it drifts every time the window
    slides and it can be reporting tomorrow morning. Restrict it to the local
    calendar day and to the hours you are predicted to be awake.
    """
    if not points:
        return None
    today = (datetime.now(UTC) + timedelta(seconds=offset_seconds)).date()
    awake = [
        p["e"]
        for p in points
        if not p.get("asleep")
        and datetime.fromisoformat(p["t"]).date() == today
    ]
    return max(awake) if awake else None


def _curve_payload() -> dict:
    """The stored forecast, verbatim. This endpoint never runs the model.

    It used to call the whole pipeline - oscillator, particle filter and all -
    on every request, which cost about 3.7 s per page load and produced numbers
    that could disagree with the ones already on the calendar.
    """
    with session_scope() as s:
        stored = forecast_store.load_curve(s)
    if stored is None:
        _request_background_refresh()
        return {
            "available": False,
            "reason": "The first forecast is still being computed.",
            "computing": True,
        }
    payload = dict(stored.payload)
    # The stored curve runs to the end of the calendar horizon, which is right
    # for placing blocks and far too much to read. Show a day around now.
    payload["points"] = _visible_window(
        payload.get("points", []), payload.get("utc_offset_seconds", 0)
    )
    payload["available"] = bool(payload.get("points"))
    payload["computed_at"] = stored.computed_at.isoformat()
    payload["age_minutes"] = int(stored.age.total_seconds() // 60)
    payload["stale"] = stored.is_stale
    return payload
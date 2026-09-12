"""The web app must never run the model, and the model must never be run twice
at once.

Opening Today used to cost about nine seconds: the page ran the whole pipeline
for one number, then the chart endpoint ran it again for the curve. Both
arrived at what the scheduler had already computed minutes earlier.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from circa import forecast_store
from circa.db.models import ForecastCache, HeartRateMinute, SleepSession, StepMinute
from circa.settings_store import RuntimeSettings

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)

FAST = {"n_particles": 300, "n_light_samples": 4}


@pytest.fixture
def client():
    from circa.web.app import create_app

    return TestClient(create_app(with_scheduler=False))


def _seed(db, nights=10):
    rng = np.random.default_rng(0)
    with db() as s:
        for d in range(nights, 0, -1):
            day = (NOW - timedelta(days=d)).astimezone(TZ).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start = (day + timedelta(hours=23.5)).astimezone(UTC)
            end = start + timedelta(hours=8)
            s.add(SleepSession(
                external_id=f"f/{d}", start_ts=start, end_ts=end,
                tz_name="America/Chicago", utc_offset_seconds=-18000,
                sleep_date=end.astimezone(TZ).date(), is_main_sleep=True,
                tst_minutes=460.0, time_in_bed_minutes=480.0,
                midpoint_ts=start + (end - start) / 2, wake_forced=False,
            ))
            for minute in range(0, 24 * 60, 10):
                ts = (day + timedelta(minutes=minute)).astimezone(UTC)
                s.add(HeartRateMinute(
                    ts=ts, bpm_median=62 + 8 * np.cos(2 * np.pi * (minute / 60 - 16) / 24)
                    + rng.normal(0, 2),
                    bpm_min=55, bpm_max=75, n_samples=6, active_fraction=0.1,
                ))
                if 8 * 60 <= minute < 22 * 60:
                    s.add(StepMinute(ts=ts, steps=int(max(0, rng.normal(40, 25)))))


def _compute(db):
    from circa.pipeline import run

    with db() as s:
        return run(s, as_of=NOW, push_calendar=False,
                   settings=RuntimeSettings(**FAST), seed=7)


# --- the store --------------------------------------------------------------


def test_a_run_publishes_a_forecast(db):
    _seed(db)
    _compute(db)
    with db() as s:
        stored = forecast_store.load_curve(s)
    assert stored is not None
    assert stored.payload["points"], "no curve points stored"
    assert stored.payload["energy_peak"] is not None
    assert stored.age < timedelta(minutes=1)
    assert not stored.is_stale


def test_the_stored_forecast_survives_a_restart(db):
    from circa.db.session import init_db, reset_engine

    _seed(db)
    _compute(db)
    reset_engine()
    init_db()
    with db() as s:
        assert forecast_store.load_curve(s) is not None


def test_a_second_run_replaces_rather_than_accumulating(db):
    _seed(db)
    _compute(db)
    _compute(db)
    with db() as s:
        assert len(list(s.scalars(select(ForecastCache)))) == 1


def test_an_old_forecast_is_marked_stale_not_hidden(db):
    """Six hours old is not wrong, only old - and a stale curve beats a spinner."""
    _seed(db)
    _compute(db)
    with db() as s:
        row = s.get(ForecastCache, forecast_store.CURVE_KIND)
        row.computed_at = datetime.now(UTC) - timedelta(hours=9)
    with db() as s:
        stored = forecast_store.load_curve(s)
    assert stored.is_stale
    assert stored.payload["points"]


# --- the web app does not compute -------------------------------------------


def test_the_chart_endpoint_never_runs_the_model(db, client, monkeypatch):
    _seed(db)
    _compute(db)

    import circa.pipeline as pipeline

    def _boom(*a, **k):
        raise AssertionError("the web app ran the model")

    monkeypatch.setattr(pipeline, "run", _boom)
    r = client.get("/api/curve")
    assert r.status_code == 200
    assert r.json()["available"] is True
    assert r.json()["points"]


def test_today_never_runs_the_model(db, client, monkeypatch):
    _seed(db)
    _compute(db)

    import circa.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the web app ran the model")),
    )
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("path", ["/", "/api/curve", "/trends", "/settings"])
def test_pages_are_fast_once_a_forecast_exists(db, client, path):
    """The threshold is generous; the point is that it is milliseconds, not
    seconds. Today was 5.0 s and the chart 3.7 s before the forecast was
    stored."""
    _seed(db)
    _compute(db)
    client.get(path)                      # warm any template compilation
    start = time.perf_counter()
    r = client.get(path)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 0.5, f"{path} took {elapsed * 1000:.0f} ms"


def test_with_no_forecast_the_page_still_renders_and_asks_for_one(db, client):
    r = client.get("/api/curve")
    body = r.json()
    assert r.status_code == 200
    assert body["available"] is False
    assert body.get("computing") is True
    assert client.get("/").status_code == 200


def test_refresh_returns_immediately(db, client):
    _seed(db)
    start = time.perf_counter()
    r = client.post("/run", follow_redirects=False)
    elapsed = time.perf_counter() - start
    assert r.status_code == 303
    assert elapsed < 0.5, f"refresh blocked for {elapsed * 1000:.0f} ms"


# --- one writer -------------------------------------------------------------


def test_only_one_model_run_happens_at_a_time(db):
    """The scheduler's job and a Refresh from the browser are separate threads
    that both write, and SQLite allows a single writer."""
    import circa.pipeline as pipeline
    from circa.web import app as web_app

    # A background refresh from an earlier test may still hold the lock.
    deadline = time.time() + 30
    while time.time() < deadline:
        if web_app._PIPELINE_LOCK.acquire(blocking=False):
            web_app._PIPELINE_LOCK.release()
            break
        time.sleep(0.05)
    else:
        pytest.fail("the pipeline lock was never released")

    running: list[int] = []
    peak: list[int] = []

    def _slow_run(*a, **k):
        running.append(1)
        peak.append(len(running))
        time.sleep(0.3)
        running.pop()
        return None

    original = pipeline.run
    pipeline.run = _slow_run
    try:
        threads = [threading.Thread(target=web_app._scheduled_pipeline) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
    finally:
        pipeline.run = original

    assert peak, "no model run happened at all"
    assert max(peak) == 1, f"{max(peak)} model runs overlapped"


def test_a_blocked_run_reports_that_it_did_not_run(db):
    from circa.web import app as web_app

    assert web_app._PIPELINE_LOCK.acquire(blocking=False)
    try:
        assert web_app._scheduled_pipeline() is False
    finally:
        web_app._PIPELINE_LOCK.release()


# --- the chart's clock ------------------------------------------------------


def test_the_curve_is_expressed_in_the_users_own_timezone(db):
    """Points and the "now" marker have to share one convention.

    They did not: points were shifted into the user's local time and then
    labelled "+00:00", while the marker was a true UTC instant. A browser
    parsed both as UTC and re-localised, so the curve landed five hours away
    from the marker - and away from every server-rendered time on the page.
    """
    _seed(db)
    _compute(db)
    with db() as s:
        payload = forecast_store.load_curve(s).payload

    offset = payload["utc_offset_seconds"]
    assert offset == -18000

    # The page builds the marker from this offset, so it is what must be right.
    first = datetime.fromisoformat(payload["points"][0]["t"])
    last = datetime.fromisoformat(payload["points"][-1]["t"])
    span = (last - first).total_seconds() / 3600
    assert 80 < span < 110, f"curve spans {span:.1f}h"

    # Read literally, the first point is the user's wall clock a day before the
    # run - not the same instant expressed in UTC.
    expected = (NOW - timedelta(hours=24) + timedelta(seconds=offset)).replace(tzinfo=None)
    drift = abs((first.replace(tzinfo=None) - expected).total_seconds())
    assert drift < 3600, f"first point is {drift / 60:.0f} min from the expected wall clock"


def test_the_now_marker_is_not_baked_into_the_stored_forecast(db):
    """A forecast is served for up to a scheduler interval after it is computed,
    so a stored marker would sit behind the clock by however long that was."""
    _seed(db)
    _compute(db)
    with db() as s:
        payload = forecast_store.load_curve(s).payload
    assert "now" not in payload
    assert "utc_offset_seconds" in payload

    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[2]
        / "circa" / "web" / "templates" / "today.html"
    ).read_text()
    assert "utc_offset_seconds" in template, "the page must build the marker itself"


def test_a_sleep_stretch_lands_at_night_on_the_chart(db):
    """The strongest available check that the axis is not shifted: the synthetic
    sleeper goes to bed at 23:30, so the shaded stretches must sit at night."""
    _seed(db)
    _compute(db)
    with db() as s:
        payload = forecast_store.load_curve(s).payload

    asleep_hours = [
        datetime.fromisoformat(p["t"]).hour for p in payload["points"] if p["asleep"]
    ]
    assert asleep_hours
    # Every asleep point should be between 23:00 and 08:00 local.
    stray = [h for h in asleep_hours if 9 <= h <= 22]
    assert not stray, f"sleep shading at {sorted(set(stray))} - the axis is shifted"


def test_the_chart_is_told_to_render_the_times_literally():
    """`useUTC` is what makes the pre-shifted values correct rather than wrong."""
    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[2]
        / "circa" / "web" / "templates" / "today.html"
    ).read_text()
    assert "useUTC: true" in template
    assert "utc_offset_seconds" in template, "the marker must use the stored offset"


# --- the forecast must not move unless something changed --------------------


def test_runs_minutes_apart_do_not_rewrite_the_calendar(db):
    """Polling every fifteen minutes must not rewrite entries that have not
    meaningfully changed.

    The model moves by seconds between runs - its light window ends at "now",
    so the bin count shifts - and once a boundary is rounded to the display
    quantum, a drift of seconds can flip it by a whole quantum. Five calendar
    entries were being rewritten on every poll for changes far below the
    precision the block itself claims.
    """
    from circa.gcal.sync import push
    from circa.pipeline import run
    from tests.test_calendar_sync import FakeCalendarClient

    _seed(db)
    client = FakeCalendarClient()
    settings = RuntimeSettings(**FAST)

    reports = []
    for minutes in (0, 1, 7, 13, 16):
        with db() as s:
            report = run(s, as_of=NOW + timedelta(minutes=minutes),
                         push_calendar=False, settings=settings, seed=7)
        with db() as s:
            reports.append(push(s, report.blocks, settings, client=client))

    for i, r in enumerate(reports[1:], start=1):
        assert r.updated == 0, (
            f"poll {i} rewrote {r.updated} unchanged entries"
        )


def test_a_real_shift_still_reaches_the_calendar(db):
    """The tolerance must absorb noise without swallowing a genuine change."""
    from datetime import timedelta as td

    from circa.gcal.sync import push
    from circa.pipeline import run
    from tests.test_calendar_sync import FakeCalendarClient

    _seed(db)
    client = FakeCalendarClient()
    settings = RuntimeSettings(**FAST)
    with db() as s:
        report = run(s, as_of=NOW, push_calendar=False, settings=settings, seed=7)
    with db() as s:
        push(s, report.blocks, settings, client=client)

    # Move every block an hour - well beyond the quantum.
    for b in report.blocks:
        b.start += td(hours=1)
        b.end += td(hours=1)
    with db() as s:
        moved = push(s, report.blocks, settings, client=client)
    assert moved.updated > 0, "a one-hour shift was absorbed as noise"


def test_the_curve_grid_is_anchored_to_a_fixed_lattice(db):
    """The samples themselves have to land on the same instants every run."""
    from circa.pipeline import CURVE_STEP_MINUTES, run

    _seed(db)
    stamps = []
    for minutes in (0, 3, 9):
        with db() as s:
            report = run(s, as_of=NOW + timedelta(minutes=minutes),
                         push_calendar=False, settings=RuntimeSettings(**FAST), seed=7)
        stamps.append(report.curve.times[0])

    step = CURVE_STEP_MINUTES * 60
    for ts in stamps:
        assert int(ts.timestamp()) % step == 0, f"{ts} is off the lattice"
    # And consecutive runs within one step share the same starting sample.
    assert stamps[0] == stamps[1], "the grid moved between runs three minutes apart"


def test_the_chart_shows_about_a_day_around_now(db):
    """The stored curve is two days wide because the calendar needs it. Reading
    two days off a 240px chart is not a thing anyone does, so the endpoint
    slices it - and the slice has to follow the clock, not the forecast."""
    from circa.web.app import (
        CURVE_VIEW_AHEAD_HOURS,
        CURVE_VIEW_BEHIND_HOURS,
        _visible_window,
    )

    offset = -18000
    base = datetime.now(UTC) + timedelta(seconds=offset)
    points = [
        {"t": (base - timedelta(hours=24) + timedelta(minutes=10 * i))
               .replace(tzinfo=None).isoformat(), "e": 50.0, "asleep": False}
        for i in range(6 * 46)
    ]
    window = _visible_window(points, offset)

    span = (
        datetime.fromisoformat(window[-1]["t"])
        - datetime.fromisoformat(window[0]["t"])
    ).total_seconds() / 3600
    assert span <= CURVE_VIEW_BEHIND_HOURS + CURVE_VIEW_AHEAD_HOURS
    assert span > 20, f"only {span:.1f}h of curve survived the slice"
    # And it is centred on now, not on whenever the forecast was computed.
    first = datetime.fromisoformat(window[0]["t"])
    behind = (base.replace(tzinfo=None) - first).total_seconds() / 3600
    assert abs(behind - CURVE_VIEW_BEHIND_HOURS) < 0.5


def test_a_curve_entirely_out_of_view_is_still_drawn(db):
    """Better an old curve, labelled, than an empty card."""
    from circa.web.app import _visible_window

    stale = [
        {"t": datetime(2020, 1, 1, 0, 10 * i % 60).isoformat(), "e": 50.0}
        for i in range(6)
    ]
    assert _visible_window(stale, 0) == stale


def test_todays_ceiling_is_todays_and_excludes_sleep(db):
    """`energy_peak` off the model is the maximum over the whole two-day curve,
    so it could be reporting tomorrow morning - and it moved every time the
    window slid. The card says "today", so it has to mean today."""
    from circa.web.app import _today_peak

    offset = -18000
    today = (datetime.now(UTC) + timedelta(seconds=offset)).date()
    tomorrow = today + timedelta(days=1)

    def point(day, hour, energy, asleep=False):
        return {
            "t": datetime(day.year, day.month, day.day, hour).isoformat(),
            "e": energy,
            "asleep": asleep,
        }

    points = [
        point(today, 3, 99.0, asleep=True),    # asleep: not a ceiling
        point(today, 10, 71.0),
        point(today, 15, 64.0),
        point(tomorrow, 10, 95.0),             # a different day
    ]
    assert _today_peak(points, offset) == 71.0

    # Nothing awake today yet - fall back rather than showing a dash.
    assert _today_peak([point(tomorrow, 10, 95.0)], offset) is None

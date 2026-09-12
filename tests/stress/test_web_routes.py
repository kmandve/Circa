"""Every page, against every state the database can actually be in.

The web app is the only place a person sees whether the system is working, so
a page that 500s on an empty or half-populated database is worse than useless -
it is indistinguishable from the collector being down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PAGES = ["/", "/trends", "/settings", "/model"]
JSON = ["/api/status", "/api/curve", "/healthz"]


@pytest.fixture
def client():
    from circa.web.app import create_app

    return TestClient(create_app())


# --- empty database ---------------------------------------------------------


@pytest.mark.parametrize("path", PAGES + JSON)
def test_every_route_survives_a_completely_empty_database(client, path):
    """This is the state on first boot, before OAuth has even happened."""
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:400]}"


@pytest.mark.parametrize("path", PAGES)
def test_pages_render_real_html_when_empty(client, path):
    r = client.get(path)
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()
    # A traceback leaking into the page is a 200 that is still a failure.
    assert "Traceback" not in r.text
    assert "jinja2.exceptions" not in r.text


# --- partially populated ----------------------------------------------------


def _seed(db, *, sleep=False, hr=False, phase=False, blocks=False):
    from circa.db.models import (
        CalendarBlock,
        HeartRateMinute,
        PhaseEstimate,
        SleepSession,
    )

    now = datetime.now(UTC)
    with db() as s:
        if sleep:
            for d in range(3):
                start = now - timedelta(days=d + 1, hours=2)
                s.add(SleepSession(
                    external_id=f"sleep/{d}", start_ts=start,
                    end_ts=start + timedelta(hours=8),
                    tz_name="America/Chicago", utc_offset_seconds=-18000,
                    sleep_date=(start - timedelta(hours=5)).date(),
                    is_main_sleep=True, tst_minutes=450.0,
                    midpoint_ts=start + timedelta(hours=4),
                ))
        if hr:
            for i in range(200):
                s.add(HeartRateMinute(
                    ts=now - timedelta(minutes=i), bpm_median=60.0 + (i % 7),
                    bpm_min=55, bpm_max=70, n_samples=12, active_fraction=0.1,
                ))
        if phase:
            s.add(PhaseEstimate(
                target_date=now.date(), computed_at=now, model_version="0.2.0",
                dlmo_ts=now + timedelta(hours=6),
                dlmo_ci80_low=now + timedelta(hours=5),
                dlmo_ci80_high=now + timedelta(hours=7),
                cbtmin_ts=now + timedelta(hours=13),
                phase_sd_minutes=55.0, confidence_tier=0, confidence_q=0.3,
                n_nights=3, channel_agreement_minutes=None, detail={},
            ))
        if blocks:
            s.add(CalendarBlock(
                block_key="sleep:sleep_window:x", category="sleep",
                target_date=now.date(), kind="sleep_window",
                start_ts=now + timedelta(hours=8), end_ts=now + timedelta(hours=16),
                title="Biological night (±60m)", description="why",
                model_version="0.2.0", forecast_revision=0, deleted=False,
            ))


COMBOS = [
    {"sleep": True},
    {"hr": True},
    {"phase": True},
    {"blocks": True},
    {"sleep": True, "hr": True},
    {"phase": True, "blocks": True},
    {"sleep": True, "hr": True, "phase": True, "blocks": True},
]


@pytest.mark.parametrize("combo", COMBOS)
@pytest.mark.parametrize("path", PAGES + JSON)
def test_routes_survive_partial_data(client, db, path, combo):
    """Data arrives in pieces; no ordering of arrival may break a page."""
    _seed(db, **combo)
    r = client.get(path)
    assert r.status_code == 200, f"{path} with {combo} -> {r.status_code}: {r.text[:400]}"
    assert "Traceback" not in r.text


# --- a phase estimate with holes in it --------------------------------------


def test_phase_estimate_with_null_fields_does_not_break_today(client, db):
    """Nullable columns are nullable; an older row may predate a field."""
    from circa.db.models import PhaseEstimate

    now = datetime.now(UTC)
    with db() as s:
        s.add(PhaseEstimate(
            target_date=now.date(), computed_at=now, model_version="0.1.0",
            dlmo_ts=now, dlmo_ci80_low=None, dlmo_ci80_high=None,
            cbtmin_ts=None, phase_sd_minutes=None, confidence_tier=0,
            confidence_q=0.0, n_nights=0, channel_agreement_minutes=None,
            detail=None,
        ))
    for path in PAGES + JSON:
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:300]}"


# --- settings round trip ----------------------------------------------------


def test_settings_form_round_trips_every_field(client, db):
    """Unchecked checkboxes are simply absent from a form POST.

    Reading them naively turns every toggle the user switched off into "leave
    unchanged", so settings appear to save but silently do not.
    """
    from circa.settings_store import load_settings

    page = client.get("/settings")
    assert page.status_code == 200

    payload = {
        "chronotype": "morning",
        "target_sleep_hours": "7.5",
        "wind_down_minutes": "45",
        "caffeine_cutoff_hours": "9",
        "last_meal_hours": "2.5",
        "round_to_minutes": "15",
        "min_block_minutes": "45",
        "max_blocks_per_day": "8",
        "forecast_horizon_hours": "48",
        "low_confidence_policy": "wide_blocks",
        "notifications": "none",
        "notification_minutes_before": "10",
        # every boolean deliberately omitted == switched off
    }
    r = client.post("/settings", data=payload, follow_redirects=False)
    assert r.status_code in (200, 302, 303), r.text[:300]

    with db() as s:
        saved = load_settings(s)
    assert saved.chronotype.value == "morning"
    assert saved.target_sleep_hours == pytest.approx(7.5)
    assert saved.enable_focus is False, "omitted checkbox did not switch off"
    assert saved.enable_sleep_blocks is False

    # And back on again.
    payload["enable_focus"] = "on"
    payload["enable_sleep_blocks"] = "on"
    client.post("/settings", data=payload, follow_redirects=False)
    with db() as s:
        saved = load_settings(s)
    assert saved.enable_focus is True
    assert saved.enable_sleep_blocks is True


@pytest.mark.parametrize(
    "bad",
    [
        {"target_sleep_hours": "not-a-number"},
        {"target_sleep_hours": "-5"},
        {"target_sleep_hours": "999"},
        {"chronotype": "wizard"},
        {"max_blocks_per_day": "0"},
        {"max_blocks_per_day": "-1"},
        {"round_to_minutes": "0"},
        {"forecast_horizon_hours": "100000"},
        {"notifications": "'; DROP TABLE setting; --"},
    ],
)
def test_settings_rejects_or_clamps_nonsense_without_500(client, db, bad):
    """A hand-crafted POST must not be able to wedge the app or the model."""
    from circa.settings_store import load_settings

    payload = {
        "chronotype": "night_owl", "target_sleep_hours": "8",
        "wind_down_minutes": "45", "caffeine_cutoff_hours": "9",
        "last_meal_hours": "3", "round_to_minutes": "5",
        "min_block_minutes": "45", "max_blocks_per_day": "10",
        "forecast_horizon_hours": "48", "low_confidence_policy": "wide_blocks",
        "notifications": "none", "notification_minutes_before": "10",
    }
    payload.update(bad)
    r = client.post("/settings", data=payload, follow_redirects=False)
    assert r.status_code < 500, f"{bad} -> {r.status_code}: {r.text[:300]}"
    # A rejected submission must say so rather than redirect to "saved".
    assert "saved=1" not in r.headers.get("location", ""), (
        f"{bad} was silently discarded but the page claimed it saved"
    )

    # Whatever landed must still be usable by the model.
    with db() as s:
        saved = load_settings(s)
    assert saved.target_sleep_hours > 0
    assert saved.max_blocks_per_day >= 1
    assert saved.round_to_minutes >= 1
    assert 1 <= saved.forecast_horizon_hours <= 24 * 14

    # And the settings table still exists.
    assert client.get("/settings").status_code == 200


# --- the form and the model must agree --------------------------------------


def test_form_bounds_are_all_accepted_by_the_model():
    """Every min/max offered by the settings page must be a legal value.

    These two lists of bounds live in different files and drifted apart: the
    page documented `hr_window_hours = 0` as the way to switch the heart-rate
    channel off while the model had just been given a floor of 12, so following
    the page's own instructions would have thrown away the whole submission.
    """
    import re

    from circa.settings_store import RuntimeSettings

    template = (
        Path(__file__).resolve().parents[2]
        / "circa" / "web" / "templates" / "settings.html"
    ).read_text()

    pattern = re.compile(
        r"num\(\s*'(?P<name>\w+)'\s*,\s*'[^']*'\s*,\s*"
        r"(?:'[^']*'|\"[^\"]*\")\s*,\s*'[^']*'\s*,\s*'(?P<min>[\d.]+)'\s*,\s*'(?P<max>[\d.]+)'"
    )
    found = list(pattern.finditer(template))
    assert found, "could not parse any numeric fields out of settings.html"

    for m in found:
        name, lo, hi = m["name"], float(m["min"]), float(m["max"])
        assert name in RuntimeSettings.model_fields, f"{name} is not a real setting"
        for edge in (lo, hi):
            value = int(edge) if float(edge).is_integer() else edge
            try:
                RuntimeSettings(**{name: value})
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"settings.html offers {name}={value} but the model rejects it: {exc}")


# --- "right now" must not look backwards ------------------------------------


def test_the_hero_never_announces_a_past_block_as_coming_up(db):
    """The timeline reaches back over recorded nights, so "not currently
    active" stopped meaning "still to come"."""
    from circa.db.models import CalendarBlock
    from circa.web.app import _today_context

    now = datetime.now(UTC)
    with db() as s:
        s.add(CalendarBlock(
            block_key="sleep:sleep_actual:past", category="sleep", kind="sleep_actual",
            target_date=(now - timedelta(days=1)).date(),
            start_ts=now - timedelta(hours=20), end_ts=now - timedelta(hours=12),
            title="Slept 8h 12m", description="recorded",
            model_version="0.2.0", deleted=False,
        ))
        s.add(CalendarBlock(
            block_key="sleep:sleep_window:future", category="sleep", kind="sleep_window",
            target_date=now.date(),
            start_ts=now + timedelta(hours=9), end_ts=now + timedelta(hours=17),
            title="Biological night (±40m)", description="forecast",
            model_version="0.2.0", deleted=False,
        ))

    from circa.web.app import _NOW_HEADLINES

    ctx = _today_context()
    # Checked by kind rather than by literal text, so a rename cannot break it.
    assert ctx["now_headline"] != _NOW_HEADLINES["sleep_actual"][0]
    expected = _NOW_HEADLINES["sleep_window"][0].lower()
    assert expected in ctx["now_headline"].lower(), ctx["now_headline"]


def test_past_blocks_are_marked_as_past(db):
    from circa.db.models import CalendarBlock
    from circa.web.app import _today_context

    now = datetime.now(UTC)
    with db() as s:
        s.add(CalendarBlock(
            block_key="sleep:sleep_actual:x", category="sleep", kind="sleep_actual",
            target_date=(now - timedelta(days=1)).date(),
            start_ts=now - timedelta(hours=20), end_ts=now - timedelta(hours=12),
            title="Slept 8h 12m", description="recorded",
            model_version="0.2.0", deleted=False,
        ))
    rows = _today_context()["blocks"]
    assert rows and rows[0]["past"] is True and rows[0]["recorded"] is True


# --- the page must not thrash -----------------------------------------------


def test_the_page_never_schedules_a_blind_reload(client, db):
    """A six-second reload timer ran for as long as a recompute took, which with
    a calendar push is half a minute of the page jumping under you. The page
    polls for a *changed* forecast instead and reloads once."""
    html = client.get("/").text
    assert "location.replace" not in html
    assert "setTimeout(() => location" not in html
    assert "/api/status" in html, "nothing is watching for the recompute to finish"


def test_refresh_posts_without_a_refreshing_query_param(client, db):
    r = client.post("/run", follow_redirects=False)
    assert r.status_code == 303
    assert "refreshing" not in r.headers.get("location", "")


def test_status_reports_whether_a_forecast_exists_yet(client, db):
    body = client.get("/api/status").json()
    assert "forecast_computed_at" in body
    assert "refreshing" in body


def test_the_technical_numbers_are_folded_away(client, db):
    """Tau, CBTmin and the credible interval are still available - behind a
    disclosure, not on the surface."""
    from circa.db.models import PhaseEstimate

    now = datetime.now(UTC)
    with db() as s:
        s.add(PhaseEstimate(
            target_date=now.date(), computed_at=now, model_version="0.2.0",
            dlmo_ts=now + timedelta(hours=6),
            dlmo_ci80_low=now + timedelta(hours=5),
            dlmo_ci80_high=now + timedelta(hours=7),
            cbtmin_ts=now + timedelta(hours=13), phase_sd_minutes=40.0,
            confidence_tier=0, confidence_q=0.4, n_nights=5, detail={},
        ))
    html = client.get("/").text
    assert "Under the hood" in html
    # Everything technical sits inside the <details>, not before it.
    head = html.split("Under the hood")[0]
    for term in ("day length", "Lowest point", "Melatonin rises"):
        assert term not in head, f"{term!r} is still on the surface"

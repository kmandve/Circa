"""Clock times are twelve-hour, everywhere a person reads one.

There was no test for this, which is why the site could carry four different
clock formats at once: the page one way, the calendar descriptions another, the
chart axis a third and the settings spinner a bare "14". A format is exactly the
kind of thing that drifts back one call site at a time, so it is asserted at the
source, at the calendar, and at the rendered page.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from circa.timefmt import clock, clock_from, clock_from_hours, stamp

ROOT = Path(__file__).resolve().parents[2]

# An hour of 13-23 with a colon and two minutes. Deliberately narrow: 09:30 is
# ambiguous between the two conventions, 21:30 is not.
MILITARY = re.compile(r"(?<![\d:])(1[3-9]|2[0-3]):[0-5]\d(?![\d:])")


# ── the formatter itself ────────────────────────────────────────────────

def test_every_minute_of_the_day_renders_as_a_twelve_hour_clock():
    for minutes in range(24 * 60):
        text = clock(minutes // 60, minutes % 60)
        assert text.endswith(("am", "pm")), text
        hour, rest = text.split(":")
        assert 1 <= int(hour) <= 12, text
        assert 0 <= int(rest[:2]) <= 59, text


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (0, 0, "12:00am"),    # midnight is 12am, not 0am
        (0, 5, "12:05am"),
        (11, 59, "11:59am"),
        (12, 0, "12:00pm"),   # noon is 12pm, not 0pm
        (12, 30, "12:30pm"),
        (13, 5, "1:05pm"),
        (23, 59, "11:59pm"),
        (24, 0, "12:00am"),   # a wrapped hour must not render as "24"
    ],
)
def test_the_awkward_hours(hour, minute, expected):
    assert clock(hour, minute) == expected


def test_a_decimal_hour_rounds_to_the_nearest_minute():
    assert clock_from_hours(13.5) == "1:30pm"
    assert clock_from_hours(0.0) == "12:00am"
    # 23.999h is a minute short of midnight after rounding, not "24:00".
    assert clock_from_hours(23.9999) == "12:00am"
    assert clock_from_hours(-1.0) == "11:00pm"


def test_a_status_stamp_keeps_its_date():
    assert stamp(datetime(2026, 9, 12, 15, 28)) == "2026-09-12 3:28pm"


def test_clock_from_reads_the_wall_clock_it_is_given():
    """It must not localise: callers have already shifted the instant."""
    assert clock_from(datetime(2026, 9, 12, 20, 5, tzinfo=UTC)) == "8:05pm"


# ── the call sites ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path", ["circa/web/app.py", "circa/gcal/blocks.py", "circa/cli.py"]
)
def test_no_module_a_person_reads_formats_its_own_clock(path):
    """Every one of these had its own `%H:%M`. One shared formatter or they
    drift apart again."""
    source = (ROOT / path).read_text()
    assert "%H:%M" not in source, f"{path} formats a clock by hand"


def test_the_calendar_says_am_and_pm():
    from tests.stress.test_daily_shape import _build

    blocks = _build(nights=14)
    text = "\n".join(f"{b.title}\n{b.description}" for b in blocks)
    assert "am" in text or "pm" in text, "no twelve-hour clock anywhere"
    stray = MILITARY.findall(text)
    assert not stray, f"twenty-four hour clock on the calendar: {stray}"


# ── the rendered page ───────────────────────────────────────────────────

@pytest.mark.parametrize("page", ["/", "/trends", "/model", "/settings"])
def test_no_page_renders_a_twenty_four_hour_clock(db, client, page):
    body = client.get(page).text
    # The chart's own data is ISO-8601 and stays that way - it is parsed, not
    # read. Strip the <script> blocks and check what a person actually sees.
    visible = re.sub(r"(?s)<script.*?</script>", "", body)
    stray = MILITARY.findall(visible)
    assert not stray, f"{page} shows {stray}"


def test_the_collection_table_shows_a_local_twelve_hour_stamp(db, client):
    """Found live: this column rendered the raw UTC ISO watermark, sliced to
    sixteen characters - "2026-09-12 20:17". The only time on the site that was
    neither the user's clock nor twelve-hour, and an empty test database hid it
    because the table had no rows."""
    from circa.db.models import SyncState
    from circa.db.session import session_scope
    from tests.stress.test_forecast_and_speed import _compute, _seed

    # The Collection table only renders once there is an estimate to render it
    # beside, which is exactly why the empty-database check missed this.
    _seed(db)
    _compute(db)
    with session_scope() as s:
        s.add(SyncState(
            data_type="heart-rate",
            watermark=datetime(2026, 9, 12, 20, 17, tzinfo=UTC),
            last_success_at=datetime(2026, 9, 12, 20, 17, tzinfo=UTC),
            points_ingested=181657,
        ))

    body = client.get("/model").text
    assert "20:17" not in body, "raw UTC watermark still on the page"
    # America/Chicago in September is UTC-5, so 20:17Z is 3:17pm.
    assert "3:17pm" in body, body[body.find("heart-rate") - 200:][:400]


def test_the_chart_labels_its_own_axis_in_twelve_hour_time():
    """The axis is built in the browser, so no server-side check reaches it."""
    for name in ("today.html", "trends.html"):
        source = (ROOT / "circa" / "web" / "templates" / name).read_text()
        assert "am" in source and "pm" in source, f"{name} has no meridiem"
        assert "':00'" not in source, f"{name} still builds an 'HH:00' label"


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from circa.web.app import create_app

    return TestClient(create_app(with_scheduler=False))


def test_a_seeded_day_renders_twelve_hour_times_on_today(db, client):
    """An empty database renders no times at all, so the page has to be checked
    with a real forecast behind it - that is where the clocks live."""
    from tests.stress.test_forecast_and_speed import _compute, _seed

    _seed(db)
    _compute(db)
    body = client.get("/").text
    visible = re.sub(r"(?s)<script.*?</script>", "", body)
    assert re.search(r"\d{1,2}:\d{2}(am|pm)", visible), "no clock times on Today"
    assert not MILITARY.findall(visible)

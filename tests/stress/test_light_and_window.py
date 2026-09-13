"""The predicted night, and the two light windows hanging off it.

Each of these was wrong in a way that produced perfectly well-formed output:
a window off-centre from its own midpoint estimate, a bright-light block that
silently vanished, and a threshold rendered as a four-hour appointment.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from circa.alertness.model import compute, grid
from circa.alertness.process_s import SleepDebt, SleepWakeHistory
from circa.gcal.blocks import (
    MAX_DLMO_TO_ONSET_HOURS,
    MIN_DLMO_TO_ONSET_HOURS,
    build_all,
    predicted_sleep_onset,
)
from circa.phase.confidence import assess
from circa.settings_store import Chronotype, RuntimeSettings

TZ = ZoneInfo("America/Chicago")
OFFSET = -5 * 3600
NOW = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)   # 10:00 local


def _build(dlmo_local_h=21.5, settings=None, debt=None, sleep_start_h=None,
           sleep_hours=8.0, conf=None):
    settings = settings or RuntimeSettings()
    # Sleep timing and DLMO are coupled in reality - DLMO is *derived* from the
    # midpoint. Pinning one while sweeping the other would test a person whose
    # melatonin rises two hours after they fall asleep.
    if sleep_start_h is None:
        sleep_start_h = (
            dlmo_local_h + settings.prior["dlmo_offset_hours"] - sleep_hours / 2
        ) % 24
    base = datetime(2026, 9, 1, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=sleep_start_h)).astimezone(UTC),
         (base + timedelta(days=d, hours=sleep_start_h + sleep_hours)).astimezone(UTC))
        for d in range(25)
    ]
    history = SleepWakeHistory(episodes, NOW - timedelta(days=10),
                               NOW + timedelta(hours=48))
    times = grid(NOW - timedelta(hours=24), NOW + timedelta(hours=48), minutes=10)
    curve = compute(times, OFFSET, history,
                    np.random.default_rng(0).normal((dlmo_local_h + 7) % 24, 0.4, 100))
    day = datetime(2026, 9, 10, tzinfo=TZ) + timedelta(hours=dlmo_local_h)
    dlmo = min((day.astimezone(UTC) + timedelta(days=k) for k in (-1, 0, 1)),
               key=lambda c: abs((c - NOW).total_seconds()))
    return build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=conf or assess(30, 95, 0.9, 20), settings=settings, offset=OFFSET,
        from_ts=NOW, to_ts=NOW + timedelta(hours=48), debt=debt,
    )


def _kind(blocks, kind):
    return [b for b in blocks if b.kind == kind]


# --- the window is centred on its own estimate -----------------------------


@pytest.mark.parametrize("chronotype", list(Chronotype))
@pytest.mark.parametrize("target", [6.0, 7.0, 8.0, 9.0, 10.0])
def test_the_predicted_night_is_centred_on_its_own_midpoint(chronotype, target):
    """DLMO is derived by subtracting the chronotype offset from the sleep
    midpoint, so the window has to be placed back by the inverse.

    Using a separate "onset = DLMO + 2h" constant instead left the window
    off-centre from the midpoint the model had just estimated - by 36 minutes
    for an evening type and 24 the other way for a morning type. The sign of the
    error flipped with chronotype, which is how it went unnoticed.
    """
    settings = RuntimeSettings(chronotype=chronotype, target_sleep_hours=target)
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)
    onset = predicted_sleep_onset(dlmo, settings, target)
    centre = onset + timedelta(hours=target / 2)
    implied_midpoint = dlmo + timedelta(hours=settings.prior["dlmo_offset_hours"])

    gap_hours = (onset - dlmo).total_seconds() / 3600
    if MIN_DLMO_TO_ONSET_HOURS < gap_hours < MAX_DLMO_TO_ONSET_HOURS:
        off_by = abs((centre - implied_midpoint).total_seconds()) / 60
        assert off_by < 1.0, f"{chronotype} at {target}h: window off-centre by {off_by:.0f} min"


@pytest.mark.parametrize("target", [4.0, 12.0])
def test_an_extreme_target_cannot_put_sleep_before_melatonin(target):
    settings = RuntimeSettings(target_sleep_hours=target)
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)
    gap = (predicted_sleep_onset(dlmo, settings, target) - dlmo).total_seconds() / 3600
    assert MIN_DLMO_TO_ONSET_HOURS <= gap <= MAX_DLMO_TO_ONSET_HOURS


def test_sleep_and_light_agree_on_when_the_night_ends():
    """They derived the predicted wake independently, and only one of them
    knew about the sleep-debt adjustment."""
    debt = SleepDebt(hours=6.0, nights=10, window_days=14,
                     last_night_hours=6.5, last_night_delta=-1.5)
    blocks = _build(debt=debt)
    night = _kind(blocks, "sleep_window")[0]
    light = _kind(blocks, "morning_light")
    assert light, "no bright-light window generated"
    assert light[0].start >= night.end - timedelta(minutes=1), (
        f"bright light starts {night.end - light[0].start} before the night ends"
    )


# --- the bright-light window must exist ------------------------------------


@pytest.mark.parametrize("dlmo_h", [19.0, 20.5, 21.5, 23.0, 0.5, 2.0])
def test_the_bright_light_window_is_always_produced(dlmo_h):
    """It was being dropped entirely, not shortened.

    Clipping waking advice against the predicted night took the complement over
    the span of the night itself, so the waking set came out empty and every
    block touching it was discarded rather than trimmed.
    """
    blocks = _build(dlmo_local_h=dlmo_h)
    assert _kind(blocks, "morning_light"), f"no bright-light window at DLMO {dlmo_h}"


@pytest.mark.parametrize("dlmo_h", [19.0, 21.5, 23.0, 1.0])
def test_the_bright_light_window_never_lands_inside_the_night(dlmo_h):
    blocks = _build(dlmo_local_h=dlmo_h)
    nights = _kind(blocks, "sleep_window")
    for light in _kind(blocks, "morning_light"):
        for night in nights:
            overlap = min(light.end, night.end) - max(light.start, night.start)
            assert overlap <= timedelta(0), f"bright light overlaps the night by {overlap}"


def test_the_bright_light_window_has_a_usable_length():
    blocks = _build()
    light = _kind(blocks, "morning_light")[0]
    length = (light.end - light.start).total_seconds() / 3600
    assert 0.75 <= length <= 6.0, f"bright-light window is {length:.1f}h"


# --- the dim-light threshold ------------------------------------------------


def test_dim_light_is_a_marker_not_an_evening_long_appointment():
    """It is a cutoff - "from here, keep it dim" - and a four-hour bar reads as
    something you are meant to sit through."""
    blocks = _build()
    dim = _kind(blocks, "dim_light")
    assert dim, "no dim-light marker"
    minutes = (dim[0].end - dim[0].start).total_seconds() / 60
    assert minutes <= 20, f"dim-light block is {minutes:.0f} minutes long"


def test_dim_light_still_says_how_long_it_applies_for():
    """Shrinking the block must not lose the information it carried."""
    blocks = _build()
    dim = _kind(blocks, "dim_light")[0]
    assert "until" in dim.description.lower()
    assert "later" in dim.description.lower()


def test_dim_light_lands_before_the_night_it_precedes():
    blocks = _build()
    dim = _kind(blocks, "dim_light")[0]
    night = _kind(blocks, "sleep_window")[0]
    assert dim.end <= night.start, "the dim-light marker is inside the night"


# --- the whole set stays coherent ------------------------------------------


@pytest.mark.parametrize("dlmo_h", [19.0, 21.5, 23.5, 1.5])
@pytest.mark.parametrize("target", [6.5, 8.0, 9.5])
def test_no_two_blocks_of_the_same_category_overlap(dlmo_h, target):
    blocks = _build(dlmo_local_h=dlmo_h,
                    settings=RuntimeSettings(target_sleep_hours=target))
    by_cat: dict[str, list] = {}
    for b in blocks:
        by_cat.setdefault(b.category, []).append(b)
    for category, items in by_cat.items():
        items.sort(key=lambda b: b.start)
        for a, b in zip(items, items[1:], strict=False):
            assert b.start >= a.end, (
                f"{category}: {a.kind} {a.start}-{a.end} overlaps {b.kind} {b.start}-{b.end}"
            )


# --- calendar colours -------------------------------------------------------


def test_every_calendar_gets_a_visibly_different_colour():
    """Three of the five used to be shades of green.

    The ids index Google's *calendarList* palette (1-24), which is a different
    list from the event palette (1-11) and does not share its names - so a
    plausible-looking id like "1" is Cocoa, not Blueberry.
    """
    from circa.gcal.blocks import CATEGORY_COLOUR_HEX, CATEGORY_META

    ids = [meta[1] for meta in CATEGORY_META.values()]
    assert len(set(ids)) == len(ids), f"two calendars share a colour: {ids}"
    for cid in ids:
        assert cid in CATEGORY_COLOUR_HEX, f"colour {cid} has no documented hex"

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    hexes = [CATEGORY_COLOUR_HEX[c] for c in ids]
    for i, a in enumerate(hexes):
        for b in hexes[i + 1:]:
            distance = sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b), strict=True)) ** 0.5
            assert distance > 60, f"{a} and {b} are too close to tell apart"


def test_events_carry_the_category_colour(db):
    """Setting a *calendar's* colour writes to `calendarList`, which
    `calendar.app.created` cannot touch - that PATCH returned 401 on every run
    since the app was written, and the calendars kept whatever colour Google
    auto-assigned. Event colours need no extra scope and are what is actually
    seen."""
    from circa.gcal.blocks import CATEGORY_EVENT_COLOUR
    from circa.gcal.sync import push
    from tests.test_calendar_sync import FakeCalendarClient, _block

    client = FakeCalendarClient()
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)
    body = next(iter(client.events.values()))["body"]
    assert body["colorId"] == CATEGORY_EVENT_COLOUR["focus"]


def test_a_changed_event_palette_reaches_events_that_already_exist(db, monkeypatch):
    """`_unchanged` compares the event body field by field. Omitting the colour
    would mean a corrected palette never reached a single existing event."""
    from sqlalchemy import select

    from circa.db.models import CalendarBlock
    from circa.gcal.sync import push
    from tests.test_calendar_sync import FakeCalendarClient, _block

    client = FakeCalendarClient()
    block = _block("focus:peak_focus:a")
    with db() as s:
        push(s, [block], RuntimeSettings(), client=client)

    # Pretend it was written under an older palette.
    with db() as s:
        row = s.scalar(select(CalendarBlock).where(CalendarBlock.block_key == block.key))
        row.event_color_id = "11"

    client.calls.clear()
    with db() as s:
        report = push(s, [block], RuntimeSettings(), client=client)
    assert report.updated == 1, "recoloured event was treated as unchanged"


def test_the_calendar_colour_patch_is_skipped_without_the_scope_for_it(db):
    """Attempting it four times a poll only produces four warnings."""
    from circa.gcal.sync import push
    from tests.test_calendar_sync import FakeCalendarClient, _block

    client = FakeCalendarClient()
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)
    assert "set_colour" not in client.calls, client.calls


def test_the_calendar_colour_is_applied_when_the_scope_is_present(db, monkeypatch):
    """With `calendar.calendarlist` granted, colours are real hex rather than
    one of Google's eleven presets."""
    from circa.gcal import sync as gcal_sync
    from circa.gcal.blocks import CATEGORY_RGB
    from circa.gcal.sync import push
    from tests.test_calendar_sync import FakeCalendarClient, _block

    monkeypatch.setattr(gcal_sync, "calendar_colours_permitted", lambda: True)
    client = FakeCalendarClient()
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)

    assert f"set_rgb:{CATEGORY_RGB['focus'][0]}" in client.calls, client.calls
    # And the event must not carry a colorId, or it would override the calendar.
    # Absent is not good enough: a PATCH leaves out what it does not mention, so
    # an omitted field left every existing event on the colour it already had.
    # It has to be sent, and sent as null.
    body = next(iter(client.events.values()))["body"]
    assert "colorId" in body, "an omitted colorId never clears an existing one"
    assert body["colorId"] is None


def test_events_fall_back_to_presets_without_the_scope(db):
    """Until the scope is granted there is nothing else to use."""
    from circa.gcal.blocks import CATEGORY_EVENT_COLOUR
    from circa.gcal.sync import push
    from tests.test_calendar_sync import FakeCalendarClient, _block

    client = FakeCalendarClient()
    with db() as s:
        push(s, [_block("focus:peak_focus:a")], RuntimeSettings(), client=client)
    body = next(iter(client.events.values()))["body"]
    assert body["colorId"] == CATEGORY_EVENT_COLOUR["focus"]
    assert not [c for c in client.calls if c.startswith("set_rgb")]


def test_the_custom_palette_is_muted_and_distinguishable():
    """Low chroma so the blocks sit under the page rather than over it, but
    separated by hue so four calendars are still tellable apart."""
    import colorsys

    from circa.gcal.blocks import CATEGORY_RGB

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    # Every calendar that is on by default. Debug is excluded because it is not.
    active = [
        CATEGORY_RGB[c][0] for c in ("focus", "sleep", "light", "body", "rhythm")
    ]
    for h in active:
        r, g, b = (c / 255 for c in rgb(h))
        hsv = colorsys.rgb_to_hsv(r, g, b)
        assert hsv[1] < 0.25, f"{h} is too saturated for a background ({hsv[1]:.2f})"
        assert hsv[2] > 0.70, f"{h} is too dark to read text on ({hsv[2]:.2f})"

    for i, a in enumerate(active):
        for b in active[i + 1:]:
            d = sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b), strict=True)) ** 0.5
            assert d > 25, f"{a} and {b} are too close to tell apart ({d:.0f})"


def test_the_palette_has_readable_text_on_every_background():
    """Google paints the foreground we give it, so the contrast is ours to get
    right."""
    from circa.gcal.blocks import CATEGORY_RGB

    def luminance(h):
        def channel(v):
            v /= 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (channel(int(h[i:i + 2], 16)) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for category, (bg, fg) in CATEGORY_RGB.items():
        lo, hi = sorted((luminance(bg), luminance(fg)))
        ratio = (hi + 0.05) / (lo + 0.05)
        assert ratio >= 4.5, f"{category}: {fg} on {bg} is only {ratio:.1f}:1"


def test_the_foreground_is_one_of_the_two_colours_google_accepts():
    """Found live, as four 400s a poll: `calendarList.patch` takes any hex for
    the background and exactly two for the foreground. A hand-picked dark brown
    answers "Invalid foreground color" and the calendar keeps whatever colour it
    had, silently, forever."""
    from circa.gcal.blocks import (
        CALENDAR_INK_DARK,
        CALENDAR_INK_LIGHT,
        CATEGORY_RGB,
    )

    legal = {CALENDAR_INK_DARK, CALENDAR_INK_LIGHT}
    for category, (_, fg) in CATEGORY_RGB.items():
        assert fg in legal, f"{category}: {fg} is not a colour Google will take"


def test_the_ink_is_whichever_of_the_two_can_actually_be_read():
    from circa.gcal.blocks import (
        CALENDAR_INK_DARK,
        CALENDAR_INK_LIGHT,
        calendar_ink,
    )

    assert calendar_ink("#FFFFFF") == CALENDAR_INK_DARK
    assert calendar_ink("#000000") == CALENDAR_INK_LIGHT
    assert calendar_ink("#E3C9B4") == CALENDAR_INK_DARK   # every Circa background
    # Mid-grey is the crossover; either answer is defensible, but it must be one
    # of the two and it must not throw.
    assert calendar_ink("#808080") in {CALENDAR_INK_DARK, CALENDAR_INK_LIGHT}


def test_event_colours_are_distinct():
    """Categories that share a colour are deliberate; the ones that differ have
    to differ clearly."""
    from circa.gcal.blocks import CATEGORY_EVENT_COLOUR, EVENT_COLOUR_HEX

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    hexes = sorted({EVENT_COLOUR_HEX[c] for c in CATEGORY_EVENT_COLOUR.values()})
    assert len(hexes) >= 3, f"everything collapsed to {hexes}"
    for i, a in enumerate(hexes):
        for b in hexes[i + 1:]:
            d = sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b), strict=True)) ** 0.5
            assert d > 95, f"{a} and {b} are too close to tell apart ({d:.0f})"



def test_web_and_calendar_palettes_agree():
    """A category must read as the same colour on both surfaces.

    They are defined in different files, for different grounds - a dark page and
    a white calendar - so they are legitimately different lightnesses. What has
    to match is the hue, and nothing stops that drifting except this.
    """
    import colorsys
    import re
    from pathlib import Path

    from circa.gcal.blocks import BODY, CATEGORY_RGB, FOCUS, LIGHT, SLEEP

    css = (
        Path(__file__).resolve().parents[2] / "circa" / "web" / "static" / "app.css"
    ).read_text()

    def hue(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0] * 360

    for var, category in [("focus", FOCUS), ("sleep", SLEEP),
                          ("light", LIGHT), ("body", BODY)]:
        m = re.search(rf"--{var}:\s*(#[0-9A-Fa-f]{{6}})", css)
        assert m, f"--{var} not found in app.css"
        web = hue(m.group(1))
        cal = hue(CATEGORY_RGB[category][0])
        delta = min(abs(web - cal), 360 - abs(web - cal))
        assert delta <= 30, (
            f"{category}: page {m.group(1)} ({web:.0f}°) and calendar "
            f"{CATEGORY_RGB[category][0]} ({cal:.0f}°) are {delta:.0f}° apart"
        )

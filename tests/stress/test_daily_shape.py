"""The daily energy pattern - the thing the whole app is for.

The peaks and dips were absent from the calendar entirely for the first week
because tier gating deleted them, which made the app useless in exactly the
period someone decides whether to keep it. These tests assert the shape of the
day is present and coherent from the very first nights.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from circa.alertness.model import compute, grid
from circa.alertness.process_s import SleepWakeHistory
from circa.gcal.blocks import (
    KIND_TITLES,
    SPARK_BARS,
    SPARK_SAMPLES,
    _awake_spans,
    _local,
    build_all,
)
from circa.phase.confidence import assess
from circa.settings_store import Chronotype, LowConfidencePolicy, RuntimeSettings

TZ = ZoneInfo("America/Chicago")
OFFSET = -5 * 3600
NOW = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)  # 10:00 local
HORIZON = 48
OVERRUN = 22

# What a day is supposed to contain, in the order it happens.
DAILY_SHAPE = ["grogginess", "peak_focus", "circadian_dip", "second_wind"]


def _build(nights=3, dlmo_local_h=21.5, settings=None, conf=None, sleep_hours=8.0):
    settings = settings or RuntimeSettings()
    conf = conf or assess(nights, 200 if nights < 7 else 95, 0.7, None)
    onset_h = (dlmo_local_h + settings.prior["dlmo_offset_hours"] - sleep_hours / 2) % 24
    base = datetime(2026, 8, 25, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=onset_h)).astimezone(UTC),
         (base + timedelta(days=d, hours=onset_h + sleep_hours)).astimezone(UTC))
        for d in range(30)
    ]
    history = SleepWakeHistory(episodes, NOW - timedelta(days=10),
                               NOW + timedelta(hours=HORIZON + OVERRUN))
    times = grid(NOW - timedelta(hours=24),
                 NOW + timedelta(hours=HORIZON + OVERRUN), minutes=10)
    curve = compute(times, OFFSET, history,
                    np.random.default_rng(0).normal((dlmo_local_h + 7) % 24, 0.4, 100))
    day = datetime(2026, 9, 10, tzinfo=TZ) + timedelta(hours=dlmo_local_h)
    dlmo = min((day.astimezone(UTC) + timedelta(days=k) for k in (-1, 0, 1)),
               key=lambda c: abs((c - NOW).total_seconds()))
    return build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=conf, settings=settings, offset=OFFSET,
        from_ts=NOW, to_ts=NOW + timedelta(hours=HORIZON),
    )


def _local_day(block):
    return block.start.astimezone(TZ).date()


# --- presence ---------------------------------------------------------------


@pytest.mark.parametrize("nights", [1, 2, 3, 5, 7, 14, 30])
def test_the_shape_of_the_day_is_present_at_every_amount_of_history(nights):
    """This is the regression that matters most: at three nights the calendar
    showed a sleep window and nothing else."""
    kinds = {b.kind for b in _build(nights=nights)}
    for kind in DAILY_SHAPE:
        assert kind in kinds, f"no {kind} with {nights} nights of history"


@pytest.mark.parametrize("chronotype", list(Chronotype))
def test_the_shape_survives_every_chronotype(chronotype):
    kinds = {b.kind for b in _build(settings=RuntimeSettings(chronotype=chronotype))}
    for kind in DAILY_SHAPE:
        assert kind in kinds, f"no {kind} for {chronotype}"


def test_a_full_day_has_every_category_on_it():
    blocks = _build()
    # The first full local day inside the horizon.
    days = sorted({_local_day(b) for b in blocks})
    full = days[1] if len(days) > 1 else days[0]
    on_that_day = [b for b in blocks if _local_day(b) == full]
    categories = {b.category for b in on_that_day}
    assert {"focus", "sleep", "light", "body"} <= categories, categories


def test_the_melatonin_window_is_shown():
    """It is the one biological event in the forecast rather than a derived
    recommendation, and everything else is anchored to it."""
    assert any(b.kind == "melatonin_window" for b in _build())


def test_morning_grogginess_is_shown_and_follows_the_night():
    blocks = _build()
    grog = [b for b in blocks if b.kind == "grogginess"]
    nights = [b for b in blocks if b.kind == "sleep_window"]
    assert grog
    for g in grog:
        assert timedelta(minutes=45) <= (g.end - g.start) <= timedelta(hours=2)
        for n in nights:
            assert min(g.end, n.end) - max(g.start, n.start) <= timedelta(0), (
                "grogginess overlaps the night it should follow"
            )


# --- order ------------------------------------------------------------------


def test_the_day_happens_in_the_right_order():
    """Grogginess, then the morning peak, then the afternoon dip, then the
    evening peak. If they come out shuffled the model is not describing a day."""
    blocks = _build()
    by_day: dict = {}
    for b in blocks:
        if b.kind in DAILY_SHAPE:
            by_day.setdefault(_local_day(b), {})[b.kind] = b.start

    checked = 0
    for day, kinds in by_day.items():
        present = [k for k in DAILY_SHAPE if k in kinds]
        if len(present) < 3:
            continue   # a partial day at either edge of the horizon
        starts = [kinds[k] for k in present]
        assert starts == sorted(starts), f"{day}: {present} came out shuffled"
        checked += 1
    assert checked >= 1, "no complete day to check"


def test_no_two_focus_blocks_overlap():
    focus = sorted((b for b in _build() if b.category == "focus"), key=lambda b: b.start)
    for a, b in zip(focus, focus[1:], strict=False):
        assert b.start >= a.end, f"{a.kind} overlaps {b.kind}"


# --- honesty ----------------------------------------------------------------


def test_cold_start_blocks_are_wider_than_mature_ones():
    """Showing them early is only defensible if the width says how sure we are."""
    cold = _build(nights=3, conf=assess(3, 260, 0.5, None))
    mature = _build(nights=30, conf=assess(30, 85, 0.95, 10))

    def span(blocks, kind):
        found = [b for b in blocks if b.kind == kind]
        return (found[0].end - found[0].start) if found else None

    assert span(cold, "sleep_window") > span(mature, "sleep_window")


def test_every_block_says_how_uncertain_it_is():
    for b in _build(nights=3):
        if b.kind in {"sleep_actual"}:
            continue
        has_label = "±" in b.title
        says_so = "provisional" in b.description.lower() or "tier" in b.description.lower()
        assert has_label or says_so, f"{b.kind} states no uncertainty at all"


def test_titles_are_plain_language():
    for b in _build():
        assert b.kind in KIND_TITLES
        assert "_" not in b.title, b.title


def test_the_conservative_policy_is_still_available():
    blocks = _build(
        settings=RuntimeSettings(low_confidence_policy=LowConfidencePolicy.ROBUST_ONLY),
        conf=assess(3, 260, 0.5, None),
    )
    assert {b.category for b in blocks} <= {"sleep", "light"}


def test_focus_can_still_be_switched_off():
    blocks = _build(settings=RuntimeSettings(enable_focus=False))
    assert not [b for b in blocks if b.category == "focus"]


# --- the waveform itself ----------------------------------------------------


def test_process_c_has_four_real_turning_points():
    """The afternoon dip has to be a turning point, not an inflection.

    With the previous parameters the morning peak sat at CBTmin + 8.5 h and the
    dip at + 9.9 h - 1.4 h apart, so they merged into a plateau and, once the
    homeostat was subtracted, the afternoon "dip" came out as a 2.7-point wobble
    on a 0-100 scale.
    """
    from circa.alertness import process_c

    h = np.linspace(0, 24, 24 * 60, endpoint=False)
    y = process_c.drive(h)
    d = np.diff(y)
    maxima = [h[i] for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]]
    minima = [h[i] for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]]

    assert len(maxima) == 2, f"expected a morning and an evening peak, got {maxima}"
    assert len(minima) == 2, f"expected a nadir and an afternoon dip, got {minima}"

    morning, evening = sorted(maxima)
    dip = min(minima, key=lambda t: abs(t - 10.5))
    nadir = min(minima, key=lambda t: min(t, 24 - t))

    assert 5.5 <= morning <= 8.0, f"morning peak at CBTmin +{morning:.1f}h"
    assert 9.5 <= dip <= 11.5, f"afternoon dip at CBTmin +{dip:.1f}h"
    assert 14.5 <= evening <= 17.0, f"evening peak at CBTmin +{evening:.1f}h"
    assert min(nadir, 24 - nadir) < 1.0, f"nadir at CBTmin +{nadir:.1f}h"


def test_the_declared_dip_and_evening_windows_bracket_the_real_extrema():
    """`circadian_dip()` and `wake_maintenance_zone()` are used to place blocks,
    so they have to still contain the turning points after any refit."""
    from circa.alertness import process_c

    h = np.linspace(0, 24, 24 * 60, endpoint=False)
    y = process_c.drive(h)
    d = np.diff(y)
    minima = [h[i] for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]]
    maxima = [h[i] for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]]
    dip = min(minima, key=lambda t: abs(t - 10.5))
    evening = max(maxima)

    lo, hi = process_c.circadian_dip(0.0)
    assert lo <= dip <= hi, f"dip {dip:.1f}h outside declared window {lo}-{hi}"
    lo, hi = process_c.wake_maintenance_zone(0.0)
    assert lo <= evening <= hi, f"evening peak {evening:.1f}h outside {lo}-{hi}"


def test_the_afternoon_dip_is_visible_in_the_finished_curve():
    """Process C having a dip is not enough - it has to survive the homeostat."""
    settings = RuntimeSettings()
    onset_h = (21.5 + settings.prior["dlmo_offset_hours"] - 4.0) % 24
    base = datetime(2026, 8, 25, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=onset_h)).astimezone(UTC),
         (base + timedelta(days=d, hours=onset_h + 8)).astimezone(UTC))
        for d in range(30)
    ]
    history = SleepWakeHistory(episodes, NOW - timedelta(days=10), NOW + timedelta(hours=48))
    times = grid(NOW - timedelta(hours=24), NOW + timedelta(hours=48), minutes=10)
    curve = compute(times, OFFSET, history, np.full(50, (21.5 + 7) % 24))

    local = np.array([t.astimezone(TZ).hour + t.astimezone(TZ).minute / 60 for t in curve.times])
    day = (~curve.asleep) & (local >= 9) & (local <= 22)
    energy, hours = curve.energy[day], local[day]

    morning = energy[hours <= 13].max()
    dip = energy[(hours >= 13) & (hours <= 18)].min()
    evening = energy[hours >= 18].max()

    assert morning - dip > 4.0, f"afternoon dip is only {morning - dip:.1f} points deep"
    assert evening - dip > 3.0, f"no recovery after the dip: {evening - dip:.1f} points"


def test_the_page_and_the_calendar_call_things_the_same_name():
    """The hero read "Circadian dip" while the timeline directly beneath it read
    "Afternoon dip", which makes one event look like two."""
    from circa.gcal.blocks import KIND_TITLES, NON_TIMELINE_KINDS
    from circa.web.app import _NOW_HEADLINES

    for kind in _NOW_HEADLINES:
        assert kind in KIND_TITLES, f"{kind} has a headline but no calendar title"
    for kind in KIND_TITLES:
        if kind in NON_TIMELINE_KINDS:
            continue
        assert kind in _NOW_HEADLINES, f"{kind} has a calendar title but no headline"

    for kind in ("peak_focus", "circadian_dip", "second_wind", "grogginess"):
        assert _NOW_HEADLINES[kind][0] == KIND_TITLES[kind], (
            f"{kind}: page says {_NOW_HEADLINES[kind][0]!r}, "
            f"calendar says {KIND_TITLES[kind]!r}"
        )


# --- the volume cap ---------------------------------------------------------


def test_the_cap_keeps_the_important_blocks_not_the_early_ones():
    """It used to truncate by start time, so a busy day silently lost its
    evening - the wind-down and the melatonin rise, which are the two most
    actionable things on it."""
    # Pick the day the cap actually bites on, which is the busiest day *before*
    # capping. Choosing it afterwards picks whichever day happens to hold five
    # blocks, and the partial day at the edge of the horizon holds five without
    # anything having been dropped - so the assertion below would pass without
    # ever exercising the cap.
    def grouped(blocks):
        out: dict = {}
        for b in blocks:
            out.setdefault(_local_day(b), []).append(b)
        return out

    uncapped = grouped(_build(settings=RuntimeSettings(max_blocks_per_day=50)))
    day = max(uncapped, key=lambda d: len(uncapped[d]))
    assert len(uncapped[day]) > 5, "nothing to cap - the fixture changed"

    capped = grouped(_build(settings=RuntimeSettings(max_blocks_per_day=5)))
    busiest = capped[day]
    assert len(busiest) <= 5
    kinds = {b.kind for b in busiest}
    # Whatever survives, it must not be "the first five things that happened".
    assert "sleep_window" in kinds or "melatonin_window" in kinds, kinds
    assert "last_meal" not in kinds, "a low-priority marker displaced an essential"


def test_a_normal_day_is_not_capped_at_all():
    """The default has to leave headroom for the full set."""
    settings = RuntimeSettings()
    blocks = _build(settings=settings)
    by_day: dict = {}
    for b in blocks:
        by_day.setdefault(_local_day(b), []).append(b)
    busiest = max(len(v) for v in by_day.values())
    assert busiest < settings.max_blocks_per_day, (
        f"a normal day generates {busiest} blocks against a cap of "
        f"{settings.max_blocks_per_day} - the cap is about to start biting"
    )


def test_a_stored_value_that_was_only_ever_the_old_default_moves_with_it(db):
    """Otherwise a user is pinned to a number they never chose."""
    from circa.db.models import Setting
    from circa.settings_store import SETTINGS_KEY, RuntimeSettings, load_settings

    with db() as s:
        s.add(Setting(key=SETTINGS_KEY, value={"max_blocks_per_day": 12}))
    with db() as s:
        assert load_settings(s).max_blocks_per_day == RuntimeSettings().max_blocks_per_day


def test_a_deliberately_chosen_value_is_left_alone(db):
    from circa.db.models import Setting
    from circa.settings_store import SETTINGS_KEY, load_settings

    with db() as s:
        s.add(Setting(key=SETTINGS_KEY, value={"max_blocks_per_day": 6}))
    with db() as s:
        assert load_settings(s).max_blocks_per_day == 6


# --- titles -----------------------------------------------------------------


def test_every_calendar_title_carries_its_emoji():
    from circa.gcal.blocks import KIND_EMOJI

    for b in _build():
        assert b.kind in KIND_EMOJI, f"{b.kind} has no emoji"
        assert KIND_EMOJI[b.kind] in b.title, f"{b.title!r} is missing {KIND_EMOJI[b.kind]}"


def test_titles_are_short_enough_for_a_calendar_grid():
    """A month view gives you a few characters before it truncates."""
    for b in _build():
        # The all-day sparkline is the exception: its title *is* the chart, and
        # it gets a whole row to itself rather than a slot beside a time.
        if b.all_day:
            continue
        base = b.title.split("(")[0].strip()
        assert len(base) <= 22, f"{base!r} is {len(base)} characters"


def test_the_web_headline_is_the_title_without_the_emoji():
    """The page has its own SVG icon set; glyphs belong on the calendar only."""
    from circa.gcal.blocks import KIND_EMOJI, KIND_TITLES, NON_TIMELINE_KINDS
    from circa.web.app import _NOW_HEADLINES

    for kind, title in KIND_TITLES.items():
        if kind in NON_TIMELINE_KINDS:
            continue
        assert _NOW_HEADLINES[kind][0] == title, (
            f"{kind}: page says {_NOW_HEADLINES[kind][0]!r}, calendar base is {title!r}"
        )
        assert KIND_EMOJI[kind] not in _NOW_HEADLINES[kind][0]


# --- the melatonin marker ---------------------------------------------------


def test_the_melatonin_marker_sits_at_the_ideal_bedtime():
    """DLMO is the *start* of the rise, two to three hours before sleep is
    actually available - a marker there answered a question nobody asks."""

    settings = RuntimeSettings()
    blocks = _build(settings=settings)
    marker = next(b for b in blocks if b.kind == "melatonin_window")
    night = next(b for b in blocks if b.kind == "sleep_window")

    # It ends where the night begins, give or take the low-confidence widening
    # applied to the night block itself.
    assert marker.end <= night.start + timedelta(minutes=1)
    assert abs((marker.end - night.start).total_seconds()) < 90 * 60
    assert (marker.end - marker.start) <= timedelta(minutes=20), "not a marker"


def test_the_melatonin_marker_is_hours_after_dlmo_not_at_it():
    from circa.gcal.blocks import plan_night

    settings = RuntimeSettings()
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)
    night = plan_night(dlmo, dlmo + timedelta(hours=7), settings)
    gap = (night.onset - dlmo).total_seconds() / 3600
    assert 1.5 <= gap <= 3.5, f"bedtime is {gap:.1f}h after DLMO"


def test_the_melatonin_description_does_not_overclaim():
    """It is named for a plateau, not the true concentration maximum."""
    marker = next(b for b in _build() if b.kind == "melatonin_window")
    text = marker.description.lower()
    assert "ideal bedtime" in text
    assert "temperature minimum" in text, "the real peak is not mentioned"


def test_the_marker_does_not_eat_the_wind_down():
    blocks = _build()
    wind = next(b for b in blocks if b.kind == "wind_down")
    assert (wind.end - wind.start) >= timedelta(minutes=45)


# --- the palette ------------------------------------------------------------


def test_the_calendar_palette_is_muted():
    """Quieter is achieved with fewer hues, not paler ones.

    Google's event palette has exactly one neutral, so there is no pale version
    of a colour to reach for - the only lever is how many distinct hues are in
    play at once.
    """
    import colorsys

    from circa.gcal.blocks import CATEGORY_EVENT_COLOUR, EVENT_COLOUR_HEX

    def sat(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[1]

    active = [
        EVENT_COLOUR_HEX[CATEGORY_EVENT_COLOUR[c]]
        for c in ("focus", "sleep", "light", "body")
    ]
    mean_saturation = sum(sat(h) for h in active) / len(active)
    assert mean_saturation < 0.30, f"palette is too loud ({mean_saturation:.2f})"

    # The two that do carry a hue still have to be tellable apart from each
    # other and from the neutral.
    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    distinct = sorted(set(active))
    for i, a in enumerate(distinct):
        for b in distinct[i + 1:]:
            d = sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b), strict=True)) ** 0.5
            assert d > 95, f"{a} and {b} are too close to tell apart ({d:.0f})"


def test_the_categories_sharing_a_colour_are_told_apart_some_other_way():
    """Light and body are both neutral, so the emoji has to carry the load."""
    from circa.gcal.blocks import CATEGORY_EVENT_COLOUR, KIND_EMOJI

    shared = [c for c in ("light", "body") if CATEGORY_EVENT_COLOUR[c] == "8"]
    assert shared, "nothing is sharing the neutral - this test is stale"
    light_kinds = {"morning_light", "dim_light"}
    body_kinds = {"workout", "caffeine_cutoff", "last_meal"}
    assert not ({KIND_EMOJI[k] for k in light_kinds} & {KIND_EMOJI[k] for k in body_kinds})


def test_the_palette_hexes_are_the_ones_google_paints():
    """`colors.get` returns a legacy palette the UI stopped using.

    Choosing a "muted" set against those numbers picked Tangerine - the single
    most saturated colour available - in the belief it was a soft peach. Every
    colour decision here has to be made against what actually gets drawn.
    """
    from circa.gcal.blocks import EVENT_COLOUR_HEX

    # A handful of spot values from the rendered palette. If these ever drift
    # back to the API's legacy hexes, every judgement about the palette is void.
    assert EVENT_COLOUR_HEX["3"] == "#8E24AA", "colorId 3 is Grape, not pale lilac"
    assert EVENT_COLOUR_HEX["6"] == "#F09300", "colorId 6 is Tangerine, not peach"
    assert EVENT_COLOUR_HEX["8"] == "#616161", "colorId 8 is Graphite, not near-white"


# --- blocks must not vanish as the day goes on ------------------------------


def _focus_at(poll_hour: int):
    """Today's focus blocks as seen by a poll at `poll_hour`, on a fixed day."""
    as_of = datetime(2026, 9, 10, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
    settings = RuntimeSettings()
    onset_h = (21.5 + settings.prior["dlmo_offset_hours"] - 4.0) % 24
    base = datetime(2026, 8, 25, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=onset_h)).astimezone(UTC),
         (base + timedelta(days=d, hours=onset_h + 8)).astimezone(UTC))
        for d in range(30)
    ]
    history = SleepWakeHistory(episodes, as_of - timedelta(days=10),
                               as_of + timedelta(hours=70))
    times = grid(as_of - timedelta(hours=36), as_of + timedelta(hours=70), minutes=10)
    curve = compute(times, OFFSET, history, np.full(50, 4.5))
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)
    blocks = build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=assess(30, 95, 0.9, 20), settings=settings, offset=OFFSET,
        from_ts=as_of, to_ts=as_of + timedelta(hours=48),
    )
    today = as_of.astimezone(TZ).date()
    return as_of, [b for b in blocks
                   if b.category == "focus" and _local_day(b) == today]


@pytest.mark.parametrize("poll_hour", list(range(6, 24)))
def test_the_days_focus_blocks_do_not_shrink_as_the_day_goes_on(poll_hour):
    """An evening peak could once be deleted minutes before it started: the
    extremum search only looked at what was still ahead, so the window shrank
    through the day until the six-hour minimum skipped the day entirely and took
    blocks that had not happened yet with it.

    The calendar now keeps the whole circadian day whether or not each block has
    passed, so the stronger statement holds: what a late poll sees for today is
    exactly what an early one saw. Parametrised over every waking hour because
    the failure only appeared late in the day.
    """
    _, early = _focus_at(6)
    as_of, late = _focus_at(poll_hour)

    assert {b.kind for b in late} == {b.kind for b in early}, (
        f"a {poll_hour}:00 poll sees a different day from a 06:00 one"
    )
    # And the evening peak is genuinely there while it is still ahead.
    if poll_hour < 19:
        assert any(b.kind == "second_wind" for b in late), (
            f"no evening peak at a {poll_hour}:00 poll"
        )



def test_block_times_do_not_change_as_the_day_goes_on():
    """When your peak is is a property of the day, not of the moment you ask.

    The search window used to start at "now", so the morning peak found at 08:00
    was a different window from the one found at 10:00 - and every poll rewrote
    the calendar."""
    settings = RuntimeSettings()
    onset_h = (21.5 + settings.prior["dlmo_offset_hours"] - 4.0) % 24
    base = datetime(2026, 8, 25, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=onset_h)).astimezone(UTC),
         (base + timedelta(days=d, hours=onset_h + 8)).astimezone(UTC))
        for d in range(30)
    ]
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)

    seen: dict[str, set] = {}
    for poll_hour in (7, 9, 11, 13, 15):
        as_of = datetime(2026, 9, 11, poll_hour, 0, tzinfo=TZ).astimezone(UTC)
        history = SleepWakeHistory(episodes, as_of - timedelta(days=10),
                                   as_of + timedelta(hours=70))
        times = grid(as_of - timedelta(hours=24), as_of + timedelta(hours=70),
                     minutes=10)
        curve = compute(times, OFFSET, history, np.full(50, 4.5))
        blocks = build_all(
            curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
            ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
            conf=assess(30, 95, 0.9, 20), settings=settings, offset=OFFSET,
            from_ts=as_of, to_ts=as_of + timedelta(hours=48),
        )
        for b in blocks:
            if b.category == "focus" and _local_day(b) == datetime(2026, 9, 12).date():
                seen.setdefault(b.kind, set()).add((b.start, b.end))

    for kind, spans in seen.items():
        assert len(spans) == 1, (
            f"{kind} moved between polls: {sorted(str(s) for s in spans)}"
        )


def test_every_generated_block_names_the_day_it_belongs_to():
    """The calendar holds one circadian day at a time, and decides membership
    from `block.day` - which is read off the end of the key. A block that named
    no day would quietly escape the rule and live on the calendar forever."""
    for nights in (1, 7, 21):
        for block in _build(nights=nights):
            assert block.day is not None, f"{block.key} names no day"
            assert block.key.endswith(block.day.isoformat()), block.key


def test_no_block_is_keyed_to_a_day_it_does_not_fall_in():
    """The calendar keeps one circadian day, so a block's key day decides
    whether it is written at all. A block keyed to a day it does not happen on
    is therefore either missing or a day early - and both were true of the
    workout window, which hangs off CBTmin in the small hours and so lands the
    afternoon *after* the DLMO its builder was named for.

    A day may legitimately run past midnight - that is the point of a wake-to-
    wake day - so the bound is generous. Being a whole day out is not.
    """
    from datetime import datetime, time, timedelta

    for nights in (1, 7, 21):
        for b in _build(nights=nights):
            day_start = datetime.combine(b.day, time.min)
            local_start = (b.start + timedelta(seconds=OFFSET)).replace(tzinfo=None)
            delta = local_start - day_start
            assert timedelta(0) <= delta < timedelta(hours=36), (
                f"{b.key} starts {local_start} - {delta} into the day it names"
            )


# --- the day's shape, readable without the dashboard -------------------------


def test_the_rhythm_summary_is_one_all_day_block_per_day():
    """The dashboard draws the energy curve properly, but reaching it means
    opening a tunnel. The calendar is already on every device, so the shape of
    the day can just be sitting there."""
    blocks = [b for b in _build(nights=21) if b.kind == "rhythm"]
    assert blocks, "no rhythm summary generated"
    assert len({b.day for b in blocks}) == len(blocks), "two summaries for one day"
    for b in blocks:
        assert b.all_day, "a day summary that claims a span of hours"
        assert b.category == "rhythm", "it must not compete with the plan blocks"
        bars = b.title.split()[0]
        assert len(bars) == SPARK_SAMPLES
        assert set(bars) <= set(SPARK_BARS), bars


def test_the_sparkline_is_scaled_to_the_day_not_to_zero():
    """A real waking day spans something like eight points out of a hundred, so
    a fixed 0-100 scale renders every day as the same flat line. Scaled to its
    own range, the shape is visible; the numbers live in the description."""
    blocks = [b for b in _build(nights=21) if b.kind == "rhythm"]
    bars = blocks[0].title.split()[0]
    assert SPARK_BARS[0] in bars, "nothing reaches the bottom of the scale"
    assert SPARK_BARS[-1] in bars, "nothing reaches the top of the scale"
    assert "/100" in blocks[0].description, "no absolute figures to read"


def test_the_sparkline_covers_one_waking_stretch_not_a_calendar_date():
    """For anyone who goes to bed after midnight the two differ: filtering by
    date picks up the tail of the previous evening as well, and the line then
    joins across a night's sleep."""
    from circa.alertness.model import compute, grid
    from circa.alertness.process_s import SleepWakeHistory
    from circa.gcal.blocks import rhythm_block

    # Sleeps at 01:00 local, wakes at 09:00 - so every date has waking samples
    # at both ends of it.
    base = datetime(2026, 8, 25, tzinfo=TZ)
    episodes = [
        ((base + timedelta(days=d, hours=25)).astimezone(UTC),
         (base + timedelta(days=d, hours=33)).astimezone(UTC))
        for d in range(30)
    ]
    history = SleepWakeHistory(episodes, NOW - timedelta(days=10),
                               NOW + timedelta(hours=HORIZON + OVERRUN))
    times = grid(NOW - timedelta(hours=36), NOW + timedelta(hours=HORIZON + OVERRUN),
                 minutes=10)
    curve = compute(times, OFFSET, history, np.full(50, 4.5))
    conf = assess(21, 95, 0.9, 20)

    spans = {_local(a, OFFSET).date(): (a, b) for a, b in _awake_spans(curve)}
    day = sorted(spans)[1]
    blocks = rhythm_block(curve, day, RuntimeSettings(), conf, OFFSET)
    assert blocks, f"no summary for {day}"
    b = blocks[0]

    span_start, span_end = spans[day]
    assert b.start >= span_start and b.end <= span_end
    # Nothing inside the block's own span may be marked asleep.
    asleep_inside = [
        a for t, a in zip(curve.times, curve.asleep, strict=False)
        if b.start <= t <= b.end and a
    ]
    assert not asleep_inside, "the sparkline runs straight through a night's sleep"


def test_the_summary_never_eats_the_blocks_it_describes():
    """It is all-day and sits on the same day as everything else, so an overlap
    resolver that treats it as a timed block collapses the entire day into it -
    which is exactly what happened the first time it shared the Focus
    calendar."""
    blocks = _build(nights=21)
    kinds = {b.kind for b in blocks}
    for essential in ("peak_focus", "circadian_dip", "second_wind"):
        assert essential in kinds, f"{essential} was swallowed by the summary"

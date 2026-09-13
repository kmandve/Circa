"""Turn a phase posterior and alertness curve into calendar blocks.

Two rules govern everything here:

* **Uncertainty is in the title, not buried.** `Peak Focus (±40m)`, never
  `Peak Focus: 10:17 AM`. Minute-precise titles would imply a precision the
  literature says is not available from a wrist device without light sensing.
* **Confidence gates detail.** At tier 0 only the most robust categories are
  written, blocks are widened, and descriptions say plainly that the estimate is
  provisional. Detail unlocks as history accumulates and is withdrawn again on
  travel, illness or a disrupted night.

Light guidance follows the human phase-response curve: light *after* CBTmin
advances the clock (shifts earlier), light *before* CBTmin delays it. For an
evening type that means morning light is the lever that stops further drift,
and evening light is what causes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import structlog

from circa.alertness.model import AlertnessCurve
from circa.phase.confidence import Confidence
from circa.settings_store import LowConfidencePolicy, RuntimeSettings
from circa.timefmt import clock, clock_from

log = structlog.get_logger(__name__)

# Calendar categories. Each becomes its own Google Calendar so it can be
# toggled independently from the sidebar on any device.
FOCUS = "focus"
SLEEP = "sleep"
LIGHT = "light"
BODY = "body"
DEBUG = "debug"

CATEGORY_ORDER = [FOCUS, SLEEP, LIGHT, BODY, DEBUG]

# Colour ids are Google's *calendarList* palette (1-24), which is not the event
# palette (1-11) and does not match its names. Verified against the live
# `/colors` endpoint rather than assumed: the previous mapping intended
# blue/indigo/amber/green/grey and actually produced lime, brown, orange, pale
# lime and green - three of the five nearly indistinguishable.
#
# Chosen to read like the reference palette: warm coral for energy, lavender for
# sleep, gold for light, soft mint for the body, grey for diagnostics.
CATEGORY_META = {
    FOCUS: ("Circa · Focus", "2", "Peak alertness, the afternoon dip, second wind"),
    SLEEP: ("Circa · Sleep", "18", "Melatonin onset, wind-down, biological night"),
    LIGHT: ("Circa · Light", "12", "When to seek and avoid light"),
    BODY: ("Circa · Body", "7", "Workout, caffeine cutoff, last meal"),
    DEBUG: ("Circa · Debug", "19", "Phase estimate, credible interval, model version"),
}

# Per-event colours, from Google's *event* palette (1-11), which is a different
# list again from the calendar palette above.
#
# These are what actually reach the calendar. Setting a calendar's colour writes
# to `calendarList`, a user-level resource that `calendar.app.created` does not
# grant access to - so that PATCH has returned 401 "Invalid Credentials" on
# every run since the app was written, and the calendars have simply kept
# whatever colour Google auto-assigned them. Event colours need no scope beyond
# the one we already hold, and they are the soft pastels the reference palette
# actually uses.
# The real palette: arbitrary hex, set on the calendars themselves. Muted and
# low-chroma on purpose - the calendar is a white page you read all day, and the
# blocks should sit under the text rather than shout over it. Distinguished by
# hue (warm / cool / neutral / green) rather than by intensity.
CATEGORY_BACKGROUND = {
    FOCUS: "#E3C9B4",   # warm tan
    SLEEP: "#C7C3DC",   # lilac
    LIGHT: "#DCDFE3",   # cool grey
    BODY:  "#C3D6C8",   # sage
    DEBUG: "#D9D9D9",   # neutral
}

# The background takes any hex. The foreground does not: it is a two-value enum,
# and anything else answers 400 "Invalid foreground color" - which is exactly
# what a hand-picked dark brown did. Verified against the live API rather than
# read off the documentation, which says only "the foreground colour".
# Omitting it is not a fix either: the field keeps whatever it had, so a
# calendar could end up white-on-tan. Derive it instead, so a future palette
# cannot reintroduce the same 400.
CALENDAR_INK_DARK = "#000000"
CALENDAR_INK_LIGHT = "#ffffff"


def _relative_luminance(colour: str) -> float:
    def channel(value: int) -> float:
        v = value / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(int(colour[i:i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def calendar_ink(background: str) -> str:
    """Black or white, whichever Google will accept and you can actually read."""
    lum = _relative_luminance(background)
    with_black = (lum + 0.05) / 0.05
    with_white = 1.05 / (lum + 0.05)
    return CALENDAR_INK_DARK if with_black >= with_white else CALENDAR_INK_LIGHT


CATEGORY_RGB = {
    category: (background, calendar_ink(background))
    for category, background in CATEGORY_BACKGROUND.items()
}

# Fallback for when the calendarList scope has not been granted, because then
# the only lever left is Google's eleven preset event colours.
CATEGORY_EVENT_COLOUR = {
    FOCUS: "4",   # Flamingo
    SLEEP: "1",   # Lavender
    LIGHT: "8",   # Graphite
    BODY: "8",    # Graphite
    DEBUG: "8",   # Graphite
}

# What Google Calendar *renders*, which is not what the API reports.
#
# `colors.get` returns a legacy palette the UI stopped painting years ago: it
# calls colorId 3 "#dbadff", a pale lilac, while the calendar draws it as a
# strong purple. Choosing a "muted" palette against those numbers picked
# Tangerine - saturation 1.00, the loudest colour in the set - believing it to
# be a soft peach. Confirmed against a screenshot of our own events rather than
# assumed. These are the values every colour decision here is made against.
EVENT_COLOUR_HEX = {
    "1": "#7986CB",   # Lavender
    "2": "#33B679",   # Sage
    "3": "#8E24AA",   # Grape
    "4": "#E67C73",   # Flamingo
    "5": "#F6BF26",   # Banana
    "6": "#F09300",   # Tangerine
    "7": "#039BE5",   # Peacock
    "8": "#616161",   # Graphite
    "9": "#3F51B5",   # Blueberry
    "10": "#0B8043",  # Basil
    "11": "#D50000",  # Tomato
}

# What the calendar-level ids render as, for the tests that assert they stay
# distinct and for anyone reading this without the API to hand.
CATEGORY_COLOUR_HEX = {
    "2": "#d06b64",   # flamingo - dusty coral
    "18": "#b99aff",  # wisteria - lavender
    "12": "#fad165",  # banana - warm gold
    "7": "#42d692",   # eucalyptus - mint green
    "19": "#c2c2c2",  # graphite - grey
}

# Categories that survive a low-confidence day under the ROBUST_ONLY policy.
# Sleep and light depend mostly on sleep timing, the most reliable signal.
ROBUST_CATEGORIES = {SLEEP, LIGHT}

# Blocks that get a reminder under the ACTIONABLE notification policy: timing
# genuinely matters and they are easy to miss.
ACTIONABLE_KINDS = {"morning_light", "dim_light", "caffeine_cutoff", "wind_down"}

# Blocks whose advice can only be acted on while awake. Anything here is
# trimmed to the observed/forecast waking period as a final pass.
#
# This is enforced centrally rather than inside each builder because the same
# bug kept reappearing in different guises: a bright-light block anchored to
# CBTmin landing before wake, a circadian dip whose low-confidence padding
# spilled into sleep, a caffeine cutoff computed 10 h before an onset that
# wrapped into the previous night. All of them are the same mistake, and all of
# them put two Circa blocks on the calendar contradicting each other.
#
# Deliberately excluded: sleep_window and wind_down (which are *about* sleep),
# and dim_light (which legitimately runs up to sleep onset).
WAKE_REQUIRED_KINDS = {
    "peak_focus", "second_wind", "circadian_dip", "grogginess",
    "morning_light", "workout", "caffeine_cutoff", "last_meal",
}


@dataclass
class Block:
    key: str                # stable idempotency handle
    category: str
    kind: str
    start: datetime
    end: datetime
    title: str
    description: str
    notify_minutes: int | None = None
    detail: dict = field(default_factory=dict)
    # Kinds this block replaces wherever it overlaps them. A recorded night
    # supersedes the forecast written for it, so the calendar shows what
    # happened rather than leaving yesterday's guess sitting beside it.
    # Matched by overlap rather than by key: the forecast's key is dated by the
    # evening its DLMO fell in, which is not the date the night ended on.
    supersedes_kinds: frozenset[str] = frozenset()

    # Which circadian day this block belongs to, as a local date. Every key ends
    # in that date already, so it is derived from the key rather than passed at
    # each of the dozen construction sites - one definition that cannot drift
    # from the handle the calendar is reconciled by.
    day: date | None = None

    def __post_init__(self) -> None:
        if self.day is None:
            self.day = _day_from_key(self.key)


def _day_from_key(key: str) -> date | None:
    """The local date a block key ends in, or None if it names no day.

    The key is authoritative rather than the block's own start: a night that
    begins at 1:30am belongs to the evening that led into it, so its key is
    dated a day before it starts. Every forecast Circa generates names a day -
    asserted by test - and anything that does not is simply not day-scoped.
    """
    try:
        return date.fromisoformat(key.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _round_to(ts: datetime, minutes: int) -> datetime:
    seconds = minutes * 60
    epoch = ts.timestamp()
    return datetime.fromtimestamp(round(epoch / seconds) * seconds, tz=UTC)


# Displayed values are quantised so that immaterial run-to-run wobble does not
# rewrite event text. Two reasons: churned descriptions defeat calendar
# idempotency (every poll would patch every event), and a "+/-62m" label that
# flickers to "+/-59m" implies a precision the estimate does not have.
# Blocks that describe the predicted night itself. Waking advice is clipped
# against these as well as against the alertness curve.
SLEEP_SPAN_KINDS = {"sleep_window"}

# Blocks that record something that already happened rather than predicting
# something that will. These are the only kinds allowed to be written into the
# past: a forecast behind you is clutter, but a record behind you is the point.
# They carry no uncertainty label, because there is none - the watch measured it.
RETROSPECTIVE_KINDS = {"sleep_actual"}

# Forecast kinds that a recorded night replaces once it overlaps them. The
# wind-down that led into a night is still interesting; the predicted window
# for a night that has already happened is not.
SUPERSEDED_BY_RECORD = frozenset({"sleep_window"})

DISPLAY_QUANTUM_MINUTES = 5

# Length given to point-like blocks - a cutoff or a threshold rather than a
# window you are meant to occupy. Kept at or under the 20-minute bound that
# `_clip_to_wake` treats as a marker.
MARKER_MINUTES = 15


def _quantise_minutes(minutes: float) -> int:
    q = DISPLAY_QUANTUM_MINUTES
    return int(round(minutes / q) * q)


def _round_up_to(ts: datetime, minutes: int) -> datetime:
    seconds = max(minutes, 1) * 60
    import math

    return datetime.fromtimestamp(math.ceil(ts.timestamp() / seconds) * seconds, tz=UTC)


def _round_down_to(ts: datetime, minutes: int) -> datetime:
    seconds = max(minutes, 1) * 60
    import math

    return datetime.fromtimestamp(math.floor(ts.timestamp() / seconds) * seconds, tz=UTC)


def _fmt_uncertainty(sd_minutes: float) -> str:
    return f"±{_quantise_minutes(sd_minutes)}m"


def _local(ts: datetime, offset: int) -> datetime:
    return ts + timedelta(seconds=offset)


def _hhmm(ts: datetime, offset: int) -> str:
    """Local wall-clock time, quantised to the display granularity.

    Quantisation exists so immaterial run-to-run wobble in a *forecast* does not
    rewrite event text on every poll. Use `_hhmm_exact` for anything measured:
    a recorded sleep onset never wobbles, and rounding 00:36 to "00:35" states
    something that simply is not what the watch saw.
    """
    local = _local(ts, offset)
    minute = _quantise_minutes(local.hour * 60 + local.minute + local.second / 60)
    return clock(minute // 60, minute % 60)


def _hhmm_exact(ts: datetime, offset: int) -> str:
    return clock_from(_local(ts, offset))


def _confidence_note(conf: Confidence) -> str:
    lines = [
        # Quality is quantised for the same reason as the times above.
        f"Confidence: tier {conf.tier} ({conf.tier_name}), quality {conf.q:.1f}",
        f"Based on {conf.n_nights} nights.",
    ]
    if conf.ood_flags:
        lines.append(
            "Unusual conditions detected (" + ", ".join(conf.ood_flags) +
            ") — the estimate is deliberately widened."
        )
    if conf.tier == 0:
        lines.append(
            "Cold start: this is largely a population prior, not yet a personal "
            "estimate. Treat the timing as approximate."
        )
    return "\n".join(lines)


def _phase_note(dlmo_ts: datetime, ci: tuple[datetime, datetime], offset: int) -> str:
    return (
        f"Estimated melatonin onset (DLMO): {_hhmm(dlmo_ts, offset)}\n"
        f"80% credible interval: {_hhmm(ci[0], offset)}–{_hhmm(ci[1], offset)}"
    )


def _widen(start: datetime, end: datetime, conf: Confidence) -> tuple[datetime, datetime]:
    """Widen a block when confidence is low, so its width shows the uncertainty."""
    if conf.q >= 0.6:
        return start, end
    # Capped deliberately: widening should signal uncertainty, not smear a
    # block across the day.
    pad = timedelta(minutes=20 if conf.q < 0.3 else 10)
    return start - pad, end + pad


# ---------------------------------------------------------------------------
# alertness-derived blocks
# ---------------------------------------------------------------------------


def _window_around(
    z: np.ndarray, centre: int, drop: float, is_peak: bool, max_span: int
) -> tuple[int, int]:
    """Expand from an extremum while the curve stays within `drop` of it.

    Threshold crossing was the obvious approach and it is wrong here: the
    alertness curve is smooth and spends many hours on one side of any fixed
    threshold, so it produced eight- and eleven-hour "blocks". Growing outward
    from the actual extremum instead gives a window whose width reflects how
    sharply peaked the day really is.
    """
    value = z[centre]
    limit = value - drop if is_peak else value + drop
    lo = hi = centre
    grew = True
    while grew and (hi - lo) < max_span:
        grew = False
        # Grow toward whichever side is still within the limit, alternating so
        # the window stays roughly centred on the extremum. `max_span` is the
        # TOTAL width - capping each side separately would silently allow
        # windows twice as wide as intended.
        if lo > 0 and (z[lo - 1] >= limit if is_peak else z[lo - 1] <= limit):
            lo -= 1
            grew = True
        if (hi - lo) < max_span and hi < len(z) - 1 and (
            z[hi + 1] >= limit if is_peak else z[hi + 1] <= limit
        ):
            hi += 1
            grew = True
    return lo, hi


# How far alertness may fall from a peak (or rise from a trough) and still count
# as part of the same window, in z units.
# How far the curve may fall away from an extremum before the block ends,
# measured on the absolute 0-100 energy scale.
#
# Deliberately not measured in z. The z-score is normalised against whatever
# window the curve happens to cover, so it shifts every time the forecast window
# slides forward - and a block two days out would change width by a quarter of
# an hour on every poll, rewriting a calendar entry that had not actually
# changed. Energy is window-independent, so the same day yields the same block
# whenever it is computed. 1.75 points is the equivalent of the 0.35 z this
# replaces, at a typical waking spread.
WINDOW_DROP_ENERGY = 1.75
# Hard caps, in hours. A "peak focus window" that spans a whole afternoon is not
# telling you anything you can act on.
MAX_FOCUS_HOURS = 3.0
MAX_DIP_HOURS = 2.5
# An evening rise only earns its own block if it is a genuinely separate feature.
SECOND_WIND_MIN_SEPARATION_HOURS = 4.0

# Display names. Internal kind strings are deliberately left alone - they are
# the idempotency handle in every block key, and they appear in half a dozen
# behavioural sets - but nothing user-facing should read like a variable name.
# Short, plain names. Deliberately no emoji here: this map is also what the web
# app's headlines are checked against, and the web has its own SVG icon set.
KIND_TITLES = {
    "grogginess": "Grogginess",
    "peak_focus": "Morning peak",
    "circadian_dip": "Afternoon dip",
    "second_wind": "Evening peak",
    "wind_down": "Wind-down",
    "melatonin_window": "Melatonin peak",
    "sleep_window": "Sleep",
    "sleep_actual": "Slept",
    "morning_light": "Get light",
    "dim_light": "Dim lights",
    "workout": "Workout",
    "caffeine_cutoff": "Last coffee",
    "last_meal": "Last meal",
    "debug": "Circa model",
}

# Calendar titles only. A month grid gives you a few characters and a glance, and
# a glyph survives that better than a word does.
KIND_EMOJI = {
    "grogginess": "🥱",
    "peak_focus": "📈",
    "circadian_dip": "📉",
    "second_wind": "📈",
    "wind_down": "🌆",
    "melatonin_window": "🌙",
    "sleep_window": "😴",
    "sleep_actual": "😴",
    "morning_light": "☀️",
    "dim_light": "🔅",
    "workout": "🏋️",
    "caffeine_cutoff": "☕",
    "last_meal": "🍽️",
    "debug": "🔧",
}


def label(kind: str, uncertainty_minutes: float | None = None, suffix: str = "") -> str:
    """The single place a calendar event title is assembled."""
    title = f"{KIND_TITLES[kind]}{suffix} {KIND_EMOJI[kind]}".strip()
    if uncertainty_minutes is not None:
        title += f" ({_fmt_uncertainty(uncertainty_minutes)})"
    return title


# Sleep inertia decays with a half-hour time constant, so three of them leaves
# about 5% of it - which is where "still waking up" stops being true.
GROGGINESS_HOURS = 1.5

# What survives when a day has more blocks than the volume cap allows. Ordered
# most important first. The cap used to keep whichever blocks started earliest,
# which meant that on a busy day it silently dropped the evening - wind-down and
# the melatonin rise, the two most actionable things on it.
KIND_PRIORITY = [
    "sleep_actual",
    "sleep_window",
    "melatonin_window",
    "peak_focus",
    "circadian_dip",
    "second_wind",
    "wind_down",
    "grogginess",
    "morning_light",
    "dim_light",
    "caffeine_cutoff",
    "workout",
    "last_meal",
    "debug",
]

# How long melatonin takes to climb from onset to the level that actually opens
# the sleep gate.
MELATONIN_WINDOW_HOURS = 1.0
# Local-hour window searched for the afternoon dip.
DIP_SEARCH_HOURS = (11, 19)


def alertness_blocks(
    curve: AlertnessCurve,
    settings: RuntimeSettings,
    conf: Confidence,
    offset: int,
    dlmo_ts: datetime,
    ci: tuple[datetime, datetime],
    from_ts: datetime | None = None,
) -> list[Block]:
    """One peak, one dip and (when distinct) one second-wind block per day."""
    blocks: list[Block] = []
    z = curve.alertness.copy()
    z[curve.asleep] = np.nan
    # Same curve, absolute scale. Extrema are identical either way (both are
    # linear in the underlying curve); only the window extent needs the stable
    # one.
    energy = curve.energy.copy().astype(float)
    energy[curve.asleep] = np.nan

    step_minutes = max(
        int((curve.times[1] - curve.times[0]).total_seconds() // 60), 1
    ) if len(curve.times) > 1 else 10
    max_focus = int(MAX_FOCUS_HOURS * 60 / step_minutes)
    max_dip = int(MAX_DIP_HOURS * 60 / step_minutes)

    from_ts = from_ts or curve.times[0]

    local_dates = np.array([_local(t, offset).date() for t in curve.times])
    for day in sorted(set(local_dates)):
        # Extrema are found over the *whole* local day, including the part that
        # has already happened. Restricting the search to what is still ahead
        # did two bad things. It shrank the window as the day went on, so once
        # fewer than six waking hours were left the day was skipped entirely -
        # taking blocks that had not happened yet with it, which is how an
        # 18:00 evening peak disappeared from the calendar at 17:53. And it made
        # the answer depend on when the model happened to run: the morning peak
        # found at 08:00 was a different window from the one found at 10:00, so
        # every poll rewrote the day.
        #
        # When your peak is is a property of the day, not of the moment you ask.
        mask = (local_dates == day) & ~np.isnan(z)
        idx = np.flatnonzero(mask)
        # A day the curve only partly covers cannot support a "best window of
        # the day" claim. This is about coverage, not about how much is left.
        if idx.size * step_minutes < 6 * 60:
            continue
        day_z = z[idx]
        day_e = energy[idx]

        # --- daily peak -------------------------------------------------
        peak_rel = int(np.argmax(day_z))
        lo, hi = _window_around(day_e, peak_rel, WINDOW_DROP_ENERGY, True, max_focus)
        # A peak that lands in the evening is the wake-maintenance zone, not a
        # work window. Naming it correctly matters because the advice inverts:
        # leaning into it pushes the clock later.
        peak_hour = _local(curve.times[idx[peak_rel]], offset).hour
        peak_kind = "second_wind" if peak_hour >= 16 else "peak_focus"
        candidates: list[tuple[str, int, int, int, float]] = [
            (peak_kind, idx[lo], idx[hi], idx[peak_rel], float(day_z[peak_rel]))
        ]

        # --- daily trough ------------------------------------------------
        # The "circadian dip" specifically means the afternoon trough. Taking
        # the global daily minimum instead picks up either post-wake sleep
        # inertia or the pre-sleep decline - both real, neither actionable, and
        # both already covered by other calendars.
        afternoon = np.array(
            [DIP_SEARCH_HOURS[0] <= _local(curve.times[i], offset).hour < DIP_SEARCH_HOURS[1]
             for i in idx]
        )
        if afternoon.any():
            aft_idx = np.flatnonzero(afternoon)
            trough_rel = int(aft_idx[np.argmin(day_z[aft_idx])])
            if abs(trough_rel - peak_rel) * step_minutes >= 60:
                lo_d, hi_d = _window_around(
                    day_e, trough_rel, WINDOW_DROP_ENERGY, False, max_dip
                )
                candidates.append(
                    ("circadian_dip", idx[lo_d], idx[hi_d], idx[trough_rel],
                     float(day_z[trough_rel]))
                )

        # --- evening second wind ------------------------------------------
        evening = np.array(
            [_local(curve.times[i], offset).hour >= 16 for i in idx]
        )
        if evening.any():
            ev_idx = np.flatnonzero(evening)
            ev_peak_rel = int(ev_idx[np.argmax(day_z[ev_idx])])
            separated = (
                abs(ev_peak_rel - peak_rel) * step_minutes
                >= SECOND_WIND_MIN_SEPARATION_HOURS * 60
            )
            if separated and day_z[ev_peak_rel] > 0 and peak_kind != "second_wind":
                lo_e, hi_e = _window_around(
                    day_e, ev_peak_rel, WINDOW_DROP_ENERGY, True, max_focus
                )
                candidates.append(
                    ("second_wind", idx[lo_e], idx[hi_e], idx[ev_peak_rel],
                     float(day_z[ev_peak_rel]))
                )

        for kind, i0, i1, i_peak, peak_z in candidates:
            start_ts, end_ts = curve.times[i0], curve.times[i1]
            if (end_ts - start_ts) < timedelta(minutes=settings.min_block_minutes):
                # Pad a very sharp extremum out to the minimum useful length,
                # centred on the extremum rather than on the window.
                centre = curve.times[i_peak]
                half = timedelta(minutes=settings.min_block_minutes / 2)
                start_ts, end_ts = centre - half, centre + half
            start_ts, end_ts = _widen(start_ts, end_ts, conf)
            start_ts = _round_to(start_ts, settings.round_to_minutes)
            end_ts = _round_to(end_ts, settings.round_to_minutes)
            if end_ts <= start_ts:
                continue
            # Now drop what has already finished. A window you are still inside
            # stays: it is the one telling you what is happening right now.
            if end_ts <= from_ts:
                continue

            blocks.append(
                Block(
                    key=f"{FOCUS}:{kind}:{day.isoformat()}",
                    category=FOCUS,
                    kind=kind,
                    start=start_ts,
                    end=end_ts,
                    title=label(kind, _block_sd(conf)),
                    description=_describe_alertness(kind, peak_z, conf, dlmo_ts, ci, offset),
                    detail={"peak_z": round(peak_z, 2)},
                )
            )

    return blocks


def _block_sd(conf: Confidence) -> float:
    """Uncertainty to display on a block title, in minutes."""
    base = conf.detail.get("ci80_width_minutes", 90.0) / 2.56  # CI80 -> ~1 SD
    return float(max(base, 20.0))


def _describe_alertness(
    kind: str, peak_z: float, conf: Confidence, dlmo_ts, ci, offset
) -> str:
    intro = {
        "peak_focus": (
            "Predicted best window for demanding, focused work — circadian drive "
            "is high and sleep pressure has not yet caught up."
        ),
        "second_wind": (
            "Evening alertness rise (the wake-maintenance zone). Useful for work, "
            "but it is also the window where falling asleep is hardest — pushing "
            "into it tends to delay your clock further."
        ),
        "circadian_dip": (
            "Predicted low point. Better spent on routine or physical tasks than "
            "on anything cognitively demanding. A short nap here is the least "
            "disruptive time for one."
        ),
    }[kind]
    return "\n\n".join(
        [intro, f"Relative alertness: {peak_z:+.1f} SD", _phase_note(dlmo_ts, ci, offset),
         _confidence_note(conf)]
    )


# ---------------------------------------------------------------------------
# phase-derived blocks
# ---------------------------------------------------------------------------


# Physiological bounds on the gap between melatonin onset and falling asleep.
# Used to clamp the derived onset so an unusual target sleep length cannot place
# sleep before melatonin has meaningfully risen.
MIN_DLMO_TO_ONSET_HOURS = 1.5
MAX_DLMO_TO_ONSET_HOURS = 3.5


@dataclass(frozen=True)
class PredictedNight:
    """One night, derived once and shared by every builder that needs it.

    Sleep onset used to be re-derived in three places - the Sleep builder, the
    Light builder and the Body builder - and they drifted apart. Light spent a
    while computing the wake time without the sleep-debt adjustment, and Body
    kept an older `DLMO + 2h` long after the others had moved to a window
    centred on the estimated midpoint, which put the caffeine cutoff and the
    last-meal marker half an hour out. Deriving it once and passing the result
    is the only version of this that stays correct.
    """

    dlmo: datetime
    cbtmin: datetime
    onset: datetime
    wake: datetime
    target_hours: float


def plan_night(
    dlmo_ts: datetime,
    cbtmin_ts: datetime,
    settings: RuntimeSettings,
    target_hours: float | None = None,
) -> PredictedNight:
    hours = settings.target_sleep_hours if target_hours is None else target_hours
    # Rounded here, once. Every builder anchors to these instants, and rounding
    # each derived boundary separately is what let the bright-light window start
    # six minutes before the night it was clipped to end at.
    onset = _round_to(
        predicted_sleep_onset(dlmo_ts, settings, hours), settings.round_to_minutes
    )
    wake = _round_to(onset + timedelta(hours=hours), settings.round_to_minutes)
    return PredictedNight(
        dlmo=dlmo_ts, cbtmin=cbtmin_ts, onset=onset, wake=wake, target_hours=hours
    )


def predicted_sleep_onset(
    dlmo_ts: datetime, settings: RuntimeSettings, target_hours: float
) -> datetime:
    """Sleep onset such that the window is centred on the predicted midpoint.

    DLMO is derived from sleep midpoint by subtracting the chronotype's offset,
    so the inverse is what places the window back. Using an independent
    "onset = DLMO + 2h" constant instead meant the predicted night was not
    centred on the midpoint the model had just estimated: 36 minutes off for an
    evening type, 24 minutes the other way for a morning type. The error changed
    sign with chronotype, which is exactly how a constant like that escapes
    notice.
    """
    midpoint = dlmo_ts + timedelta(hours=settings.prior["dlmo_offset_hours"])
    onset = midpoint - timedelta(hours=target_hours / 2)
    gap = (onset - dlmo_ts).total_seconds() / 3600
    gap = min(max(gap, MIN_DLMO_TO_ONSET_HOURS), MAX_DLMO_TO_ONSET_HOURS)
    return dlmo_ts + timedelta(hours=gap)


def _duration_words(hours: float) -> str:
    total = int(round(hours * 60))
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}h {m:02d}m"
    return f"{h}h" if h else f"{m}m"


def _debt_sentence(debt) -> str:
    """Plain-language sleep debt, or an honest statement that it is too early."""
    if debt is None or not debt.is_meaningful:
        return (
            "Sleep debt needs a few more nights before it means anything - "
            "one short night is a short night, not a trend."
        )
    if debt.hours >= 0.5:
        return (
            f"Sleep debt: {_duration_words(debt.hours)} over the last "
            f"{debt.window_days} days. Paying it back gradually works; trying "
            f"to clear it in one night mostly produces a bedtime you will not keep."
        )
    if debt.hours <= -0.5:
        return (
            f"You are {_duration_words(abs(debt.hours))} ahead of your target "
            f"over the last {debt.window_days} days."
        )
    return f"Sleep debt is roughly clear over the last {debt.window_days} days."


def observed_sleep_blocks(
    nights,
    settings: RuntimeSettings,
    offset: int,
    debt=None,
    from_ts: datetime | None = None,
) -> list[Block]:
    """One block per night actually recorded, written where it happened.

    This is the half of the picture the calendar was missing. Everything else
    Circa writes is a forecast, so behind the current moment the calendar was
    simply blank - there was no way to see what last night actually was, or
    whether the plan it produced bore any relation to it.

    Each of these supersedes the predicted window for the same night: once the
    night has happened, the forecast for it is no longer the interesting object.
    """
    if not settings.show_recorded_sleep or settings.history_days <= 0:
        return []

    blocks: list[Block] = []
    for night in nights:
        if night.start_ts is None or night.end_ts is None:
            continue
        day = _local(night.end_ts, offset).date().isoformat()
        in_bed = (night.end_ts - night.start_ts).total_seconds() / 3600
        slept = (night.tst_minutes / 60.0) if night.tst_minutes else in_bed

        lines = [
            f"Asleep {_hhmm_exact(night.start_ts, offset)} - "
            f"{_hhmm_exact(night.end_ts, offset)} "
            f"({_duration_words(in_bed)} in bed, {_duration_words(slept)} asleep).",
        ]
        if night.n_segments > 1:
            lines.append(
                f"Recorded as {night.n_segments} segments and merged back into one "
                "night - a long awakening splits the session, not the night."
            )
        target = settings.target_sleep_hours
        delta = slept - target
        if abs(delta) >= 0.25:
            direction = "short of" if delta < 0 else "above"
            lines.append(
                f"{_duration_words(abs(delta))} {direction} your "
                f"{_duration_words(target)} target."
            )
        lines.append(_debt_sentence(debt))
        if night.wake_forced:
            lines.append(
                "This wake looks alarm-driven, so it says more about your "
                "calendar than your clock and is weighted down accordingly."
            )

        blocks.append(
            Block(
                key=f"{SLEEP}:sleep_actual:{day}",
                category=SLEEP,
                kind="sleep_actual",
                start=night.start_ts,
                end=night.end_ts,
                # No uncertainty label: this was measured, not predicted.
                title=label("sleep_actual", suffix=f" {_duration_words(slept)}"),
                description="\n\n".join(lines),
                detail={"n_segments": night.n_segments, "slept_hours": round(slept, 2)},
                supersedes_kinds=SUPERSEDED_BY_RECORD,
            )
        )
    return blocks


def grogginess_blocks(
    curve: AlertnessCurve,
    settings: RuntimeSettings,
    conf: Confidence,
    offset: int,
    from_ts: datetime,
) -> list[Block]:
    """The stretch after each wake where alertness has not caught up yet.

    Sleep inertia is already in the alertness curve, but it was never surfaced
    as a block - so the first ninety minutes of the day, which is when people
    most often mistake "I am broken" for "I am awake", had nothing on it.
    """
    blocks: list[Block] = []
    if not len(curve.times):
        return blocks

    for span_start, span_end in _awake_spans(curve):
        # A span that begins at the very edge of the window is a wake we did
        # not actually see, so its start is an artefact of where the forecast
        # happens to begin rather than a real transition.
        if span_start <= curve.times[0]:
            continue
        end = min(span_start + timedelta(hours=GROGGINESS_HOURS), span_end)
        if end - span_start < timedelta(minutes=settings.min_block_minutes):
            continue
        if end <= from_ts:
            continue
        day = _local(span_start, offset).date().isoformat()
        blocks.append(
            Block(
                key=f"{FOCUS}:grogginess:{day}",
                category=FOCUS,
                kind="grogginess",
                start=_round_to(span_start, settings.round_to_minutes),
                end=_round_to(end, settings.round_to_minutes),
                title=label("grogginess", _block_sd(conf)),
                description="\n\n".join([
                    "Sleep inertia. Alertness is genuinely low here and it is "
                    "not a reflection of how the rest of your day will go - it "
                    "clears on its own in about ninety minutes.",
                    "Light, movement and delaying the first coffee by an hour "
                    "all shorten it. Hard cognitive work does not go well in "
                    "this window and is better pushed into the peak that "
                    "follows it.",
                    _confidence_note(conf),
                ]),
            )
        )
    return blocks


def sleep_blocks(
    night: PredictedNight,
    ci: tuple[datetime, datetime],
    settings: RuntimeSettings,
    conf: Confidence,
    offset: int,
    debt=None,
) -> list[Block]:
    dlmo_ts, target_hours = night.dlmo, night.target_hours
    blocks: list[Block] = []
    day = _local(dlmo_ts, offset).date().isoformat()
    sd = _block_sd(conf)

    onset = night.onset

    if settings.enable_wind_down:
        wind_start = onset - timedelta(minutes=settings.wind_down_minutes)
        s, e = _widen(wind_start, onset, conf)
        blocks.append(
            Block(
                key=f"{SLEEP}:wind_down:{day}",
                category=SLEEP,
                kind="wind_down",
                start=_round_to(s, settings.round_to_minutes),
                end=_round_to(e, settings.round_to_minutes),
                title=label("wind_down", sd),
                description="\n\n".join([
                    "Start powering down for your predicted sleep window: dim "
                    "lights, screens down or warmed, nothing cognitively "
                    "demanding.",
                    _phase_note(dlmo_ts, ci, offset),
                    _confidence_note(conf),
                ]),
            )
        )

    # Anchored to the predicted bedtime, not to DLMO. DLMO is the *start* of the
    # rise and sits two to three hours before sleep is actually available, so a
    # marker there answered a question nobody asks. What you want to know is when
    # melatonin has climbed far enough to open the sleep gate - which is the same
    # instant as the ideal bedtime, and is what the reference app labels its
    # "melatonin window".
    #
    # Named for the plateau it marks rather than the true concentration maximum,
    # which comes later, near the core-temperature minimum. The description says
    # so rather than leaving the name to imply otherwise.
    melatonin_peak = onset
    blocks.append(
        Block(
            key=f"{SLEEP}:melatonin_window:{day}",
            category=SLEEP,
            kind="melatonin_window",
            start=_round_to(
                melatonin_peak - timedelta(minutes=MARKER_MINUTES),
                settings.round_to_minutes,
            ),
            end=_round_to(melatonin_peak, settings.round_to_minutes),
            title=label("melatonin_window", sd),
            description="\n\n".join([
                f"Melatonin has climbed to the level that opens your sleep gate. "
                f"This is your ideal bedtime - {_hhmm(melatonin_peak, offset)} - "
                "derived from your own sleep timing rather than from a target "
                "hour on the clock.",
                "Falling asleep from here takes the least effort it will all "
                "night. The rise itself began around "
                f"{_hhmm(dlmo_ts, offset)}, which is why the lights should "
                "already be down.",
                "(Melatonin's true concentration peak comes later, nearer your "
                f"temperature minimum at {_hhmm(night.cbtmin, offset)}. This "
                "marks the point it becomes useful to you, not its maximum.)",
                _phase_note(dlmo_ts, ci, offset),
                _confidence_note(conf),
            ]),
        )
    )

    # Predicted sleep window, lengthened to repay part of any accumulated debt.
    wake = night.wake
    s, e = _widen(onset, wake, conf)
    blocks.append(
        Block(
            key=f"{SLEEP}:sleep_window:{day}",
            category=SLEEP,
            kind="sleep_window",
            start=_round_to(s, settings.round_to_minutes),
            end=_round_to(e, settings.round_to_minutes),
            title=label("sleep_window", sd),
            description="\n\n".join([
                (
                    f"Predicted sleep window for {_duration_words(target_hours)} "
                    "of sleep, aligned to your estimated melatonin onset rather "
                    "than to the clock."
                    + (
                        f" That is {_duration_words(target_hours - settings.target_sleep_hours)} "
                        "longer than your usual target, to pay back part of your "
                        "sleep debt."
                        if target_hours > settings.target_sleep_hours + 1e-6
                        else ""
                    )
                ),
                _debt_sentence(debt),
                f"Estimated core-temperature minimum: {_hhmm(night.cbtmin, offset)} "
                "— the deepest point of your biological night, and the hardest "
                "time to be awake and functional.",
                _phase_note(dlmo_ts, ci, offset),
                _confidence_note(conf),
            ]),
        )
    )
    return blocks


def light_blocks(
    night: PredictedNight,
    ci: tuple[datetime, datetime],
    settings: RuntimeSettings,
    conf: Confidence,
    offset: int,
) -> list[Block]:
    """Morning-light and evening-dim windows, from the light PRC.

    Both are clipped to the predicted waking period. The phase-response curve
    is defined on circadian time, so the raw windows are anchored to CBTmin and
    DLMO - but a recommendation to seek bright light at 04:15 while the same
    model predicts you are asleep until 06:30 is incoherent advice, and the two
    blocks would visibly contradict each other on the calendar.
    """
    dlmo_ts, cbtmin_ts = night.dlmo, night.cbtmin
    blocks: list[Block] = []
    day = _local(cbtmin_ts, offset).date().isoformat()
    sd = _block_sd(conf)

    sleep_onset, predicted_wake = night.onset, night.wake

    # Light after CBTmin advances the clock, strongest in the first hours after
    # it - but it cannot reach you before you are awake.
    morning_start = max(cbtmin_ts + timedelta(hours=1), predicted_wake)
    morning_end = morning_start + timedelta(hours=settings.morning_light_hours)
    # Never let the block run past the point where light stops advancing and
    # starts having little effect.
    morning_end = min(morning_end, cbtmin_ts + timedelta(hours=8))
    if morning_end <= morning_start:
        morning_end = morning_start + timedelta(hours=1)
    s, e = _widen(morning_start, morning_end, conf)
    blocks.append(
        Block(
            key=f"{LIGHT}:morning_light:{day}",
            category=LIGHT,
            kind="morning_light",
            start=_round_to(s, settings.round_to_minutes),
            end=_round_to(e, settings.round_to_minutes),
            title=label("morning_light", sd),
            description="\n\n".join([
                "Bright light during this window shifts your clock EARLIER. "
                "Outdoors is worth far more than indoor lighting — even an "
                "overcast day is many times brighter than a lit room.",
                "This is the main lever you have against drifting later. "
                "Anchored to your estimated temperature minimum and clipped to "
                "when you are predicted to be awake.",
                _phase_note(dlmo_ts, ci, offset),
                _confidence_note(conf),
            ]),
            notify_minutes=settings.notification_minutes_before,
        )
    )

    # Light before CBTmin delays the clock. The practical boundary is a couple
    # of hours before DLMO, when melatonin is about to rise. It ends at
    # predicted sleep onset rather than an arbitrary offset - after that the
    # Sleep calendar takes over and a second overlapping bar adds nothing.
    # A cutoff, not an occupation. Written as a short marker the way the
    # caffeine cutoff is: the advice is "from here onwards, keep it dim", and
    # rendering that as a four-hour bar blocked out the whole evening on the
    # calendar as though it were an appointment.
    dim_start = dlmo_ts - timedelta(hours=settings.dim_light_lead_hours)
    dim_start = min(dim_start, sleep_onset - timedelta(minutes=MARKER_MINUTES))
    until = _hhmm(sleep_onset, offset)
    blocks.append(
        Block(
            key=f"{LIGHT}:dim_light:{day}",
            category=LIGHT,
            kind="dim_light",
            start=_round_to(dim_start, settings.round_to_minutes),
            end=_round_to(
                dim_start + timedelta(minutes=MARKER_MINUTES), settings.round_to_minutes
            ),
            title=label("dim_light", sd),
            description="\n\n".join([
                f"From now until roughly {until}, bright light pushes your clock "
                "LATER and suppresses the melatonin rise that makes sleep "
                "available.",
                "Dim overheads, warm the screens, and keep the last hour low-lit. "
                "This is a threshold rather than something to sit through, so it "
                "is marked at the point it starts.",
                _phase_note(dlmo_ts, ci, offset),
                _confidence_note(conf),
            ]),
            notify_minutes=settings.notification_minutes_before,
        )
    )
    return blocks


def body_blocks(
    night: PredictedNight,
    ci: tuple[datetime, datetime],
    settings: RuntimeSettings,
    conf: Confidence,
    offset: int,
) -> list[Block]:
    dlmo_ts, cbtmin_ts = night.dlmo, night.cbtmin
    blocks: list[Block] = []
    day = _local(dlmo_ts, offset).date().isoformat()
    sd = _block_sd(conf)
    onset = night.onset

    if settings.enable_workout_window:
        # Muscle strength and core temperature peak together, roughly 11-13 h
        # after CBTmin. Exercise here also carries a modest phase-advancing
        # effect for an evening type.
        w_start = cbtmin_ts + timedelta(hours=10)
        w_end = cbtmin_ts + timedelta(hours=13)
        s, e = _widen(w_start, w_end, conf)
        blocks.append(
            Block(
                key=f"{BODY}:workout:{day}",
                category=BODY,
                kind="workout",
                start=_round_to(s, settings.round_to_minutes),
                end=_round_to(e, settings.round_to_minutes),
                title=label("workout", sd),
                description="\n\n".join([
                    "Muscle strength, power output and core temperature peak "
                    "together in this window, and exercise timed here is a mild "
                    "phase-advancing signal.",
                    "Training much later than this can delay your clock and push "
                    "sleep onset back.",
                    _confidence_note(conf),
                ]),
            )
        )

    if settings.enable_caffeine_cutoff:
        # Time for caffeine to decay to the chosen residual fraction:
        #   t = half_life * log2(1 / fraction)
        hours = settings.caffeine_half_life_hours * np.log2(
            1.0 / max(settings.caffeine_residual_fraction, 1e-3)
        )
        cutoff = onset - timedelta(hours=float(hours))
        blocks.append(
            Block(
                key=f"{BODY}:caffeine_cutoff:{day}",
                category=BODY,
                kind="caffeine_cutoff",
                start=_round_to(cutoff, settings.round_to_minutes),
                end=_round_to(cutoff + timedelta(minutes=15), settings.round_to_minutes),
                title=label("caffeine_cutoff"),
                description="\n\n".join([
                    f"With a {settings.caffeine_half_life_hours:.0f}h half-life, "
                    f"caffeine after this leaves more than "
                    f"{settings.caffeine_residual_fraction:.0%} still circulating "
                    f"at your predicted sleep onset ({_hhmm(onset, offset)}).",
                    "Caffeine shortens deep sleep even when it does not stop you "
                    "falling asleep.",
                    _confidence_note(conf),
                ]),
                notify_minutes=settings.notification_minutes_before,
            )
        )

    if settings.enable_last_meal:
        last_meal = onset - timedelta(hours=settings.last_meal_hours_before_sleep)
        blocks.append(
            Block(
                key=f"{BODY}:last_meal:{day}",
                category=BODY,
                kind="last_meal",
                start=_round_to(last_meal, settings.round_to_minutes),
                end=_round_to(last_meal + timedelta(minutes=15), settings.round_to_minutes),
                title=label("last_meal"),
                description="\n\n".join([
                    "Aiming to finish eating by here keeps your peripheral "
                    "(metabolic) clocks aligned with your central clock.",
                    "Worth knowing: meal timing shifts peripheral rhythms but has "
                    "little effect on melatonin phase, so this is the weakest of "
                    "the recommendations here.",
                    _confidence_note(conf),
                ]),
            )
        )
    return blocks


def debug_block(
    dlmo_ts, cbtmin_ts, ci, conf, posterior_detail: dict, offset: int, settings
) -> Block:
    day = _local(dlmo_ts, offset).date().isoformat()
    lines = [
        f"DLMO {_hhmm(dlmo_ts, offset)}  CI80 {_hhmm(ci[0], offset)}–{_hhmm(ci[1], offset)}",
        f"CBTmin {_hhmm(cbtmin_ts, offset)}",
        f"tier {conf.tier} ({conf.tier_name}) q={conf.q}",
        f"  posterior={conf.q_posterior} coverage={conf.q_coverage} "
        f"agreement={conf.q_agreement} ood={conf.q_ood}",
        f"nights={conf.n_nights} flags={conf.ood_flags or 'none'}",
    ]
    for key in ("tau_mean", "tau_sd", "ess", "filter_sd_minutes", "model_adequacy_sd_minutes"):
        if key in posterior_detail:
            lines.append(f"{key}={posterior_detail[key]}")
    return Block(
        key=f"{DEBUG}:summary:{day}",
        category=DEBUG,
        kind="debug_summary",
        start=_round_to(dlmo_ts, settings.round_to_minutes),
        end=_round_to(dlmo_ts + timedelta(minutes=15), settings.round_to_minutes),
        title=f"Phase {_hhmm(dlmo_ts, offset)} ±{int(conf.detail.get('ci80_width_minutes', 0) / 2)}m",
        description="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _merge_spans(spans: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _complement(
    spans: list[tuple[datetime, datetime]], start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """[start, end) with `spans` removed."""
    out: list[tuple[datetime, datetime]] = []
    cursor = start
    for lo, hi in _merge_spans(spans):
        if lo > cursor:
            out.append((cursor, min(lo, end)))
        cursor = max(cursor, hi)
        if cursor >= end:
            break
    if cursor < end:
        out.append((cursor, end))
    return [(a, b) for a, b in out if b > a]


def _awake_spans(curve: AlertnessCurve) -> list[tuple[datetime, datetime]]:
    """Contiguous awake intervals from the alertness curve's sleep mask."""
    spans: list[tuple[datetime, datetime]] = []
    start: datetime | None = None
    step = (
        curve.times[1] - curve.times[0] if len(curve.times) > 1 else timedelta(minutes=10)
    )
    for ts, asleep in zip(curve.times, curve.asleep, strict=False):
        if not asleep and start is None:
            start = ts
        elif asleep and start is not None:
            spans.append((start, ts))
            start = None
    if start is not None:
        spans.append((start, curve.times[-1] + step))
    return spans


def _clip_to_wake(
    block: Block,
    spans: list[tuple[datetime, datetime]],
    covered: tuple[datetime, datetime],
    min_minutes: int,
    quantum: int = 5,
) -> Block | None:
    """Trim a block to the largest waking span it overlaps, or drop it.

    Returns None when the block cannot be placed in waking hours at all - which
    is the right answer, because advice you are asleep for is not advice.
    """
    # Only judge blocks the curve actually covers. A block outside the forecast
    # range has no sleep/wake information attached, and treating "no overlap"
    # as "asleep" would silently drop legitimate blocks near the window edge.
    #
    # The bound must come from the curve's own time range, NOT from the awake
    # spans: if the window opens mid-sleep, everything before the first waking
    # moment looks uncovered and escapes clipping entirely. That let a
    # "last substantial meal" marker land 30 minutes after sleep onset.
    covered_start, covered_end = covered
    if block.end <= covered_start or block.start >= covered_end:
        return block

    best: tuple[datetime, datetime] | None = None
    for span_start, span_end in spans:
        lo, hi = max(block.start, span_start), min(block.end, span_end)
        if hi > lo and (best is None or (hi - lo) > (best[1] - best[0])):
            best = (lo, hi)
    if best is None:
        return None

    lo, hi = best
    if lo == block.start and hi == block.end:
        return block
    # Clipping produces a boundary like 14:59; put it back on the quantum, but
    # only inwards, so trimming can never hand back time that was clipped away.
    lo = _round_up_to(lo, quantum)
    hi = _round_down_to(hi, quantum)
    if hi <= lo:
        lo, hi = best
    # A point-like marker (caffeine cutoff, last meal) survives any overlap; a
    # window has to retain enough length to still mean something.
    original = block.end - block.start
    is_marker = original <= timedelta(minutes=20)
    if not is_marker and (hi - lo) < timedelta(minutes=min_minutes):
        return None
    block.start, block.end = lo, hi
    return block


# A block set anchored on one DLMO reaches from the wind-down that precedes it
# to the light and body blocks keyed off the same night's CBTmin the next
# morning.
_ANCHOR_SPAN_BEFORE_HOURS = 4.0
_ANCHOR_SPAN_AFTER_HOURS = 20.0


def _phase_anchors(
    dlmo_ts: datetime,
    cbtmin_ts: datetime,
    from_ts: datetime,
    to_ts: datetime,
    covered: tuple[datetime, datetime] | None,
) -> list[tuple[datetime, datetime]]:
    """Every daily recurrence of the (DLMO, CBTmin) pair the horizon can support.

    The pair is shifted as a unit so the physiological gap between them is
    preserved; shifting them independently is what put CBTmin on the wrong
    night in the first place.

    An anchor is only used when the blocks it generates fit entirely inside the
    alertness curve. The curve is the sole authority on when the person is
    asleep, so a block reaching past its edge cannot be wake-checked - and
    `_clip_to_wake` deliberately passes such blocks through untouched rather
    than false-dropping them. Emitting one would reintroduce exactly the bug
    that check exists to prevent.
    """
    before = timedelta(hours=_ANCHOR_SPAN_BEFORE_HOURS)
    after = timedelta(hours=_ANCHOR_SPAN_AFTER_HOURS)
    window_start, window_end = from_ts - before, to_ts
    if covered is not None:
        window_start = max(window_start, covered[0] + before)
        window_end = min(window_end, covered[1] - after)

    anchors: list[tuple[datetime, datetime]] = []
    for k in range(-2, 5):
        shift = timedelta(days=k)
        anchor = dlmo_ts + shift
        if window_start <= anchor <= window_end:
            anchors.append((anchor, cbtmin_ts + shift))
    return anchors


def _resolve_category_overlaps(
    blocks: list[Block], min_minutes: int, round_to: int = 5
) -> list[Block]:
    """Stop two blocks on the same calendar from claiming the same minutes.

    Every window is widened for uncertainty and rounded to the display quantum
    independently of its neighbours, so two genuinely separate extrema can still
    be pushed into overlap - and the Focus calendar then shows "Second Wind"
    and "Circadian Dip" across the same half hour, which is not a forecast, it
    is a contradiction.

    The boundary goes to the middle of the overlap, which is roughly where the
    curve crosses between the two extrema. Markers and recorded nights are
    immovable: a cutoff is a single instant, and a record is a measurement.
    """
    marker = timedelta(minutes=MARKER_MINUTES)
    by_category: dict[str, list[Block]] = {}
    for block in blocks:
        by_category.setdefault(block.category, []).append(block)

    out: list[Block] = []
    for items in by_category.values():
        items.sort(key=lambda b: (b.start, b.end))
        for previous, current in zip(items, items[1:], strict=False):
            if current.start >= previous.end:
                continue
            prev_fixed = (
                previous.end - previous.start <= marker
                or previous.kind in RETROSPECTIVE_KINDS
            )
            curr_fixed = (
                current.end - current.start <= marker
                or current.kind in RETROSPECTIVE_KINDS
            )
            if curr_fixed and not prev_fixed:
                previous.end = current.start
            elif prev_fixed and not curr_fixed:
                current.start = previous.end
            elif not prev_fixed and not curr_fixed:
                boundary = current.start + (previous.end - current.start) / 2
                # Rounded like every other boundary, or the split lands on
                # something like 12:52 next to neighbours on clean quarters.
                boundary = _round_to(boundary, round_to)
                boundary = min(max(boundary, current.start), previous.end)
                previous.end, current.start = boundary, boundary
            # Two fixed blocks overlapping is not resolvable by trimming; leave
            # them and let the length filter below decide.
        for block in items:
            length = block.end - block.start
            if length <= timedelta(0):
                log.debug("blocks.dropped_overlap_collapsed", kind=block.kind)
                continue
            if length <= marker or block.kind in RETROSPECTIVE_KINDS:
                out.append(block)
                continue
            if length < timedelta(minutes=min_minutes):
                log.debug("blocks.dropped_overlap_too_short", kind=block.kind)
                continue
            out.append(block)
    return out


def build_all(
    curve: AlertnessCurve,
    dlmo_ts: datetime,
    cbtmin_ts: datetime,
    ci: tuple[datetime, datetime],
    conf: Confidence,
    settings: RuntimeSettings,
    offset: int,
    posterior_detail: dict | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    observed_nights=None,
    debt=None,
) -> list[Block]:
    """Generate every enabled block, then apply confidence and volume policy."""
    if settings.paused:
        return []

    from_ts = from_ts or datetime.now(UTC)
    to_ts = to_ts or from_ts + timedelta(hours=settings.forecast_horizon_hours)

    # The curve's own time range, needed both to choose phase anchors and to
    # decide which blocks can be wake-checked.
    covered: tuple[datetime, datetime] | None = None
    if len(curve.times):
        step = (
            curve.times[1] - curve.times[0]
            if len(curve.times) > 1
            else timedelta(minutes=10)
        )
        covered = (curve.times[0], curve.times[-1] + step)

    blocks: list[Block] = []
    if settings.enable_focus:
        blocks += alertness_blocks(
            curve, settings, conf, offset, dlmo_ts, ci, from_ts=from_ts
        )
        blocks += grogginess_blocks(curve, settings, conf, offset, from_ts)

    # The alertness curve already spans the whole horizon, but the sleep, light
    # and body blocks hang off a single DLMO instant - so they used to cover one
    # night no matter how long the horizon was. Worse, `dlmo_ts` is the *nearest*
    # occurrence to now, which for a night owl points at last night's melatonin
    # onset for every poll between midnight and 10:00. The morning sync then
    # produced a biological night that had already ended, it was filtered out as
    # past, and tonight's never got written at all. Anchor one set of blocks on
    # each daily recurrence the horizon touches instead.
    from circa.alertness.process_s import target_sleep_tonight

    nightly_target = (
        target_sleep_tonight(
            settings.target_sleep_hours, debt,
            settings.sleep_debt_payback_fraction, settings.max_debt_payback_hours,
        )
        if debt is not None
        else settings.target_sleep_hours
    )

    for anchor_dlmo, anchor_cbt in _phase_anchors(
        dlmo_ts, cbtmin_ts, from_ts, to_ts, covered
    ):
        night = plan_night(anchor_dlmo, anchor_cbt, settings, nightly_target)
        if settings.enable_sleep_blocks:
            blocks += sleep_blocks(night, ci, settings, conf, offset, debt=debt)
        if settings.enable_light:
            blocks += light_blocks(night, ci, settings, conf, offset)
        blocks += body_blocks(night, ci, settings, conf, offset)
    if settings.enable_debug_calendar:
        blocks.append(
            debug_block(dlmo_ts, cbtmin_ts, ci, conf, posterior_detail or {}, offset, settings)
        )

    # --- keep waking advice inside waking hours ---------------------------
    spans = _awake_spans(curve)
    if spans and covered is not None:
        kept: list[Block] = []
        for block in blocks:
            if block.kind not in WAKE_REQUIRED_KINDS:
                kept.append(block)
                continue
            clipped = _clip_to_wake(
                block, spans, covered, settings.min_block_minutes,
                settings.round_to_minutes,
            )
            if clipped is None:
                log.debug("blocks.dropped_no_waking_overlap", kind=block.kind)
                continue
            kept.append(clipped)
        blocks = kept

    # --- keep the calendars from contradicting each other -----------------
    # The alertness curve's sleep mask is built from observed and projected
    # sleep; the Sleep calendar's "Biological night" is built from the phase
    # estimate. They are different objects and disagree by a few minutes, and
    # each block is rounded independently afterwards, which can push them into
    # overlap even when the underlying times did not. Clipping against the
    # night that will actually be written is what stops the Light calendar
    # saying "get bright light" while the Sleep calendar still says
    # "biological night" - the contradiction that reached the real calendar.
    night_spans = [
        (b.start, b.end) for b in blocks if b.kind in SLEEP_SPAN_KINDS
    ]
    if night_spans and covered is not None:
        # The complement has to be taken over the whole forecast range, not over
        # the span of the nights themselves. Bounding it by the nights made the
        # waking set empty whenever a single night covered the range, and every
        # block that touched it was then dropped rather than trimmed - which is
        # what silently removed the morning bright-light window.
        lo = min(covered[0], min(s for s, _ in night_spans))
        hi = max(covered[1], max(e for _, e in night_spans))
        awake = _complement(night_spans, lo, hi)
        kept = []
        for block in blocks:
            if block.kind not in WAKE_REQUIRED_KINDS:
                kept.append(block)
                continue
            clipped = _clip_to_wake(
                block, awake, (lo, hi), settings.min_block_minutes,
                settings.round_to_minutes,
            )
            if clipped is None:
                log.debug("blocks.dropped_inside_predicted_night", kind=block.kind)
                continue
            kept.append(clipped)
        blocks = kept

    # --- what actually happened -------------------------------------------
    # Built before the gating and exempt from it. Everything above is a forecast
    # and is rightly withheld when the model is unsure; a recorded night is not
    # a forecast and there is nothing to be unsure about. Suppressing it would
    # blank the calendar behind you on exactly the days you most want to look
    # back at - and those are the low-confidence days.
    records = observed_sleep_blocks(
        observed_nights or [], settings, offset, debt=debt, from_ts=from_ts
    )

    # --- no two blocks may claim the same minutes -------------------------
    blocks = _resolve_category_overlaps(
        blocks, settings.min_block_minutes, settings.round_to_minutes
    )

    # --- low-confidence policy --------------------------------------------
    if conf.is_low:
        if settings.low_confidence_policy is LowConfidencePolicy.WRITE_NOTHING:
            log.info("blocks.suppressed_low_confidence", q=conf.q, tier=conf.tier)
            return sorted(records, key=lambda b: b.start)
        if settings.low_confidence_policy is LowConfidencePolicy.ROBUST_ONLY:
            blocks = [b for b in blocks if b.category in ROBUST_CATEGORIES]
        # WIDE_BLOCKS needs no filtering - widening already happened per block.

    # Deliberately no tier gating here any more.
    #
    # Tier 0 used to delete every Focus and Body block, so for the first week -
    # exactly the week someone decides whether this is worth keeping - the app
    # showed a sleep window and nothing else, and the daily energy pattern that
    # is the entire point was missing. "We are not certain yet" is information
    # that belongs *on* a block, in its width and its wording; it is not a
    # reason to withhold the block. The shape of the day (morning peak,
    # afternoon dip, evening peak) is a robust property of the two-process
    # model and barely moves with phase uncertainty - only its timing shifts,
    # and that shift is what the +/- label already states.
    #
    # Suppression is still available, but only where the user has explicitly
    # asked for it via `low_confidence_policy` above.

    blocks += records

    # --- notifications -----------------------------------------------------
    from circa.settings_store import NotificationPolicy

    for block in blocks:
        if settings.notifications is NotificationPolicy.NONE:
            block.notify_minutes = None
        elif settings.notifications is NotificationPolicy.ALL:
            block.notify_minutes = settings.notification_minutes_before
        else:  # ACTIONABLE
            block.notify_minutes = (
                settings.notification_minutes_before
                if block.kind in ACTIONABLE_KINDS
                else None
            )

    # --- volume cap --------------------------------------------------------
    by_day: dict[date, list[Block]] = {}
    exempt: list[Block] = []
    for block in blocks:
        if block.kind in RETROSPECTIVE_KINDS:
            # At most one a day, and the only block that cannot be regenerated
            # later - never let the volume cap be what drops it.
            exempt.append(block)
            continue
        by_day.setdefault(_local(block.start, offset).date(), []).append(block)
    capped: list[Block] = list(exempt)
    order = {kind: i for i, kind in enumerate(KIND_PRIORITY)}
    for day_blocks in by_day.values():
        if len(day_blocks) > settings.max_blocks_per_day:
            log.info(
                "blocks.day_capped",
                kept=settings.max_blocks_per_day, generated=len(day_blocks),
            )
        # Keep by importance, then restore chronological order.
        day_blocks.sort(key=lambda b: (order.get(b.kind, len(order)), b.start))
        capped.extend(day_blocks[: settings.max_blocks_per_day])
    return sorted(capped, key=lambda b: b.start)

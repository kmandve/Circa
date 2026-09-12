"""Confidence gating, alertness shape, and calendar block policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from circa.alertness.model import compute, grid
from circa.alertness.process_s import SleepWakeHistory, simulate
from circa.phase.confidence import assess, detect_ood
from circa.settings_store import LowConfidencePolicy, NotificationPolicy, RuntimeSettings

BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _history(days=6, start_hour=0.0, sleep_hours=8.0):
    eps = [
        (BASE + timedelta(days=d, hours=start_hour),
         BASE + timedelta(days=d, hours=start_hour + sleep_hours))
        for d in range(days)
    ]
    return SleepWakeHistory(episodes=eps, window_start=BASE, window_end=BASE + timedelta(days=days))


# --- Process S -------------------------------------------------------------


def test_sleep_pressure_rises_awake_and_falls_asleep():
    history = _history()
    times = grid(BASE + timedelta(days=3), BASE + timedelta(days=4), minutes=15)
    s = simulate(history, times, s_initial=0.3)

    asleep = np.array([history.is_asleep(t) for t in times])
    # Compare the first and last hour of each state within the day.
    awake_idx = np.flatnonzero(~asleep)
    assert s[awake_idx[-1]] > s[awake_idx[0]]
    sleep_idx = np.flatnonzero(asleep)
    assert s[sleep_idx[-1]] < s[sleep_idx[0]]
    assert (s >= 0).all() and (s <= 1).all()


def test_short_sleep_leaves_more_pressure_than_long_sleep():
    """Identical phase, different sleep: the whole reason S and C are separate."""
    times = grid(BASE + timedelta(days=4), BASE + timedelta(days=4, hours=12), minutes=15)
    long_night = simulate(_history(sleep_hours=8.5), times, s_initial=0.6)
    short_night = simulate(_history(sleep_hours=4.5), times, s_initial=0.6)
    assert short_night.mean() > long_night.mean()


# --- alertness curve -------------------------------------------------------


def test_alertness_curve_has_dip_and_evening_recovery():
    """The two features the calendar acts on must be genuine turning points."""
    history = _history()
    times = grid(BASE + timedelta(days=3), BASE + timedelta(days=4), minutes=10)
    curve = compute(times, 0, history, np.array([5.0]))

    awake = ~curve.asleep
    hours, z = curve.local_hours[awake], curve.alertness[awake]
    order = np.argsort(hours)
    hours, z = hours[order], z[order]

    def band(lo, hi):
        mask = (hours >= lo) & (hours < hi)
        return float(z[mask].mean())

    # Post-wake inertia, then recovery.
    assert band(10, 12) > band(8, 9.5)
    # Afternoon dip, then an evening rise.
    assert band(15, 17) < band(11, 13)
    assert band(18, 20.5) > band(15, 17)


def test_phase_uncertainty_widens_the_alertness_band():
    """A vaguer phase estimate must produce a visibly wider band, not a wider claim."""
    history = _history()
    times = grid(BASE + timedelta(days=3), BASE + timedelta(days=4), minutes=15)
    rng = np.random.default_rng(0)
    tight = compute(times, 0, history, rng.normal(5.0, 0.1, 300))
    loose = compute(times, 0, history, rng.normal(5.0, 1.5, 300))
    assert np.mean(loose.upper - loose.lower) > np.mean(tight.upper - tight.lower)


# --- confidence ------------------------------------------------------------


def test_nights_alone_never_advance_the_tier():
    """Plenty of nights but terrible data must not read as high confidence."""
    good = assess(40, ci80_width_minutes=90, data_coverage=0.95,
                  channel_agreement_minutes=15)
    starved = assess(40, ci80_width_minutes=300, data_coverage=0.2,
                     channel_agreement_minutes=240, ood_flags=["travel", "illness"])
    assert good.tier == 3
    assert starved.tier == 0
    assert starved.q < good.q


def test_ood_flags_reduce_confidence_multiplicatively():
    clean = assess(30, 90, 0.9, 20)
    one = assess(30, 90, 0.9, 20, ood_flags=["travel"])
    two = assess(30, 90, 0.9, 20, ood_flags=["travel", "all_nighter"])
    assert clean.q > one.q > two.q


def test_cold_start_is_tier_zero_however_good_the_data():
    conf = assess(3, ci80_width_minutes=85, data_coverage=1.0, channel_agreement_minutes=5)
    assert conf.tier == 0
    assert conf.is_low


def test_single_channel_is_neither_corroborated_nor_penalised_as_disagreement():
    single = assess(30, 90, 0.9, None)
    agreeing = assess(30, 90, 0.9, 10)
    disagreeing = assess(30, 90, 0.9, 300)
    assert disagreeing.q_agreement < single.q_agreement < agreeing.q_agreement


def test_detect_ood_catches_the_regimes_that_break_wearable_models():
    assert "all_nighter" in detect_ood(None, None, last_sleep_hours=1.5)
    assert "travel" in detect_ood(None, None, travel_mode=True)
    assert "irregular_sleep" in detect_ood(None, None, sleep_sd_hours=3.0)
    assert "low_hr_coverage" in detect_ood(None, None, hr_coverage=0.2)
    assert detect_ood(None, None, sleep_sd_hours=0.5, hr_coverage=0.9,
                      last_sleep_hours=7.5) == []


# --- blocks ----------------------------------------------------------------


def _blocks(conf, settings=None, cbtmin=5.0):
    from circa.gcal.blocks import build_all

    settings = settings or RuntimeSettings()
    history = _history(days=8)
    times = grid(BASE + timedelta(days=3), BASE + timedelta(days=5), minutes=10)
    curve = compute(times, 0, history, np.random.default_rng(0).normal(cbtmin, 0.4, 200))
    dlmo = BASE + timedelta(days=3, hours=22)
    cbt = BASE + timedelta(days=4, hours=cbtmin)
    return build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=cbt,
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=conf, settings=settings, offset=0,
        from_ts=BASE + timedelta(days=3),
    )


def test_focus_blocks_are_actionable_lengths():
    """Threshold crossing on a smooth curve produced 8-hour 'blocks'; caps matter."""
    conf = assess(30, 90, 0.95, 15)
    blocks = _blocks(conf)
    focus = [b for b in blocks if b.category == "focus"]
    assert focus, "expected focus blocks at high confidence"
    for b in focus:
        duration = (b.end - b.start).total_seconds() / 3600
        assert 0.5 <= duration <= 4.0, f"{b.kind} was {duration:.1f}h"


def test_titles_carry_uncertainty_not_false_precision():
    blocks = _blocks(assess(30, 90, 0.95, 15))
    focus = [b for b in blocks if b.category == "focus"]
    assert all("±" in b.title for b in focus)
    assert not any(":" in b.title.split("(")[0] for b in focus)


def test_cold_start_still_shows_the_shape_of_the_day():
    """Tier 0 used to delete every Focus block.

    That meant the first week - the week someone decides whether this is worth
    keeping - showed a sleep window and nothing else, with the daily energy
    pattern that is the whole point missing entirely. Uncertainty belongs in a
    block's width and its wording, not in whether it exists: the shape of the
    day is a robust property of the two-process model, and only its *timing*
    moves with phase uncertainty - which is exactly what the +/- states.
    """
    blocks = _blocks(assess(3, 200, 0.6, None))
    kinds = {b.kind for b in blocks}
    assert "peak_focus" in kinds, "no morning peak at cold start"
    assert "circadian_dip" in kinds, "no afternoon dip at cold start"
    assert "sleep_window" in kinds


def test_a_cold_start_block_is_wider_and_says_so_than_a_mature_one():
    """The honesty has to live somewhere, and this is where."""
    cold = _blocks(assess(3, 260, 0.5, None))
    mature = _blocks(assess(40, 85, 0.95, 10))

    def width(blocks, kind):
        found = [b for b in blocks if b.kind == kind]
        return (found[0].end - found[0].start) if found else None

    assert width(cold, "sleep_window") > width(mature, "sleep_window")
    note = next(b for b in cold if b.kind == "sleep_window").description.lower()
    assert "provisional" in note or "cold start" in note or "rough" in note


def test_suppression_is_still_available_when_asked_for():
    """Removing tier gating must not remove the explicit opt-out."""
    settings = RuntimeSettings(low_confidence_policy=LowConfidencePolicy.ROBUST_ONLY)
    blocks = _blocks(assess(3, 260, 0.5, None), settings)
    assert {b.category for b in blocks} <= {"sleep", "light"}


def test_low_confidence_write_nothing_policy_is_respected():
    settings = RuntimeSettings(low_confidence_policy=LowConfidencePolicy.WRITE_NOTHING)
    assert _blocks(assess(3, 300, 0.3, None), settings) == []


def test_paused_suppresses_everything():
    assert _blocks(assess(30, 90, 0.95, 15), RuntimeSettings(paused=True)) == []


def test_notification_policy_none_clears_all_reminders():
    blocks = _blocks(assess(30, 90, 0.95, 15),
                     RuntimeSettings(notifications=NotificationPolicy.NONE))
    assert all(b.notify_minutes is None for b in blocks)


def test_actionable_notifications_only_fire_on_timing_critical_blocks():
    blocks = _blocks(assess(30, 90, 0.95, 15),
                     RuntimeSettings(notifications=NotificationPolicy.ACTIONABLE))
    notified = {b.kind for b in blocks if b.notify_minutes is not None}
    assert notified <= {"morning_light", "dim_light", "caffeine_cutoff", "wind_down"}


def test_block_keys_are_stable_across_runs():
    """Idempotency depends on this: unstable keys would recreate every event."""
    conf = assess(30, 90, 0.95, 15)
    first = {b.key for b in _blocks(conf)}
    second = {b.key for b in _blocks(conf)}
    assert first == second
    assert len(first) == len(_blocks(conf))  # keys are unique


def test_light_windows_follow_the_phase_response_curve():
    """Bright light after CBTmin (advancing), dim light before DLMO (anti-delay)."""
    blocks = _blocks(assess(30, 90, 0.95, 15))
    morning = next(b for b in blocks if b.kind == "morning_light")
    dim = next(b for b in blocks if b.kind == "dim_light")
    cbt = BASE + timedelta(days=4, hours=5.0)
    dlmo = BASE + timedelta(days=3, hours=22)
    assert morning.start >= cbt          # after the temperature minimum
    assert dim.start <= dlmo             # before melatonin onset


# --- light proxy calibration ----------------------------------------------


def test_light_proxy_constants_produce_plausible_daily_exposure():
    """Guards a calibration error that over-advanced the oscillator.

    The original constants implied ~68,000 lux-hours/day - a person living
    outdoors - because global horizontal irradiance was used as eye-level
    illuminance and any incidental movement implied a 10% chance of being
    outside in every ten-minute bin. Light is the dominant zeitgeber, so a 15x
    over-estimate shifted the phase estimate over an hour early.
    """
    from circa.phase import light_proxy as lp

    # A sedentary indoor bin must not read as likely-outdoors.
    sedentary_p = lp.BASE_OUTDOOR_PROB + (1 - lp.BASE_OUTDOOR_PROB) * (
        1 / (1 + np.exp(-lp.OUTDOOR_STEPS_SLOPE * (0 - lp.OUTDOOR_STEPS_MIDPOINT)))
    )
    assert sedentary_p < 0.10, f"sitting still implies {sedentary_p:.0%} outdoors"

    # Sustained walking should read as likely outdoors.
    walking_p = lp.BASE_OUTDOOR_PROB + (1 - lp.BASE_OUTDOOR_PROB) * (
        1 / (1 + np.exp(-lp.OUTDOOR_STEPS_SLOPE * (110 - lp.OUTDOOR_STEPS_MIDPOINT)))
    )
    assert walking_p > 0.5

    # Midday outdoor illuminance at the eye, from a clear-sky ceiling.
    clear_sky_ghi = 900.0  # W/m^2, near summer maximum
    eye_lux = clear_sky_ghi * lp.LUMENS_PER_WATT * lp.EYE_LEVEL_FACTOR * lp.OUTDOOR_ATTENUATION_MEAN
    assert 3_000 < eye_lux < 25_000, f"midday outdoor implies {eye_lux:,.0f} lux at the eye"


def test_light_blocks_never_fall_inside_the_predicted_sleep_window():
    """Regression: the bright-light block started 2.25h before predicted wake.

    Anchoring to CBTmin without clipping to the waking period produced advice to
    seek bright light at 04:15 while the Sleep calendar simultaneously showed
    "Biological night" until 06:30 - two Circa blocks directly contradicting
    each other.
    """
    from circa.gcal.blocks import light_blocks, plan_night

    settings = RuntimeSettings()
    conf = assess(30, 90, 0.95, 15)
    dlmo = BASE + timedelta(days=3, hours=22)          # 22:00
    cbt = BASE + timedelta(days=4, hours=5)            # 05:00 next day
    night = plan_night(dlmo, cbt, settings)

    blocks = light_blocks(night, (dlmo, dlmo), settings, conf, offset=0)
    morning = next(b for b in blocks if b.kind == "morning_light")
    dim = next(b for b in blocks if b.kind == "dim_light")

    # Clipping happens after the low-confidence widening, so padding cannot
    # push the block back into the sleep window.
    assert morning.start >= night.wake, (
        f"bright light starts {morning.start} but wake is predicted {night.wake}"
    )
    # The dim-light marker hands over to the Sleep calendar at sleep onset.
    assert dim.end <= night.onset
    assert dim.start < dlmo


def test_morning_light_window_length_is_configurable():
    from circa.gcal.blocks import light_blocks, plan_night

    conf = assess(30, 90, 0.95, 15)
    dlmo = BASE + timedelta(days=3, hours=22)
    cbt = BASE + timedelta(days=4, hours=5)

    def window(hours):
        settings = RuntimeSettings(morning_light_hours=hours)
        blocks = light_blocks(
            plan_night(dlmo, cbt, settings), (dlmo, dlmo), settings, conf, offset=0
        )
        block = next(b for b in blocks if b.kind == "morning_light")
        return block.end - block.start

    assert window(4.0) > window(1.0)


def test_every_block_kind_emitted_has_a_display_title():
    """`KIND_TITLES` is the only place a user-facing name is allowed to live."""
    from circa.gcal.blocks import KIND_TITLES

    blocks = _blocks(assess(30, 90, 0.95, 15))
    for block in blocks:
        assert block.kind in KIND_TITLES, f"{block.kind} has no display title"
        assert not block.title.startswith(block.kind), block.title


# --- channel agreement -------------------------------------------------------


def _obs(channel, mu, day_offset=0):
    from circa.phase.particle_filter import ChannelObservation

    return ChannelObservation(
        day=(BASE + timedelta(days=day_offset)).date(),
        channel=channel,
        mu_hours=mu,
        kappa=4.0,
    )


def test_repeated_single_channel_is_not_mistaken_for_corroboration():
    """The sleep channel is replayed once per filter day.

    Comparing raw observations would score twenty-one identical copies of one
    channel as perfect agreement, handing an uncorroborated estimate the same
    q_agreement as two independent channels landing on the same minute.
    """
    from circa.phase.particle_filter import channel_agreement_minutes

    replayed = [_obs("sleep", 21.5, d) for d in range(21)]
    assert channel_agreement_minutes(replayed) is None
    assert assess(30, 90, 0.9, channel_agreement_minutes(replayed)).q_agreement < 1.0


def test_agreement_is_not_diluted_by_how_often_a_channel_is_replayed():
    """Disagreement must not shrink just because one channel repeats."""
    from circa.phase.particle_filter import channel_agreement_minutes

    once = channel_agreement_minutes([_obs("sleep", 21.5), _obs("hr", 23.5)])
    replayed = channel_agreement_minutes(
        [_obs("sleep", 21.5, d) for d in range(21)] + [_obs("hr", 23.5)]
    )
    assert once == pytest.approx(120.0)
    assert replayed == pytest.approx(once)


def test_agreement_wraps_around_midnight():
    from circa.phase.particle_filter import channel_agreement_minutes

    spread = channel_agreement_minutes([_obs("sleep", 23.5), _obs("hr", 0.5)])
    assert spread == pytest.approx(60.0)

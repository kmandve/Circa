"""Recording what happened, and letting it change what is recommended.

Circa only ever wrote forecasts, so behind the current moment the calendar was
blank: no way to see last night, and no way to tell whether the plan bore any
relation to it. Two properties matter here - records go into the past and
forecasts never do, and last night actually changes tonight's recommendation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from sqlalchemy import select

from circa.alertness.model import compute, grid
from circa.alertness.process_s import (
    SleepDebt,
    recent_sleep_debt,
    target_sleep_tonight,
)
from circa.db.models import CalendarBlock, SleepSession
from circa.gcal.blocks import RETROSPECTIVE_KINDS, build_all
from circa.gcal.sync import push
from circa.phase import sleep_phase
from circa.phase.confidence import assess
from circa.settings_store import RuntimeSettings
from tests.test_calendar_sync import FakeCalendarClient

TZ = ZoneInfo("America/Chicago")
OFFSET = -5 * 3600
NOW = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)  # 10:00 local


def _seed_nights(db, specs):
    """specs: list of (days_ago, onset_local_hour, hours_asleep)."""
    with db() as s:
        for days_ago, onset_h, slept in specs:
            day = (NOW - timedelta(days=days_ago)).astimezone(TZ).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start = (day + timedelta(hours=onset_h)).astimezone(UTC)
            end = start + timedelta(hours=slept + 0.3)
            s.add(SleepSession(
                external_id=f"sleep/-{days_ago}", start_ts=start, end_ts=end,
                tz_name="America/Chicago", utc_offset_seconds=OFFSET,
                sleep_date=end.astimezone(TZ).date(), is_main_sleep=True,
                tst_minutes=slept * 60, time_in_bed_minutes=(slept + 0.3) * 60,
                midpoint_ts=start + (end - start) / 2, excluded=False,
            ))


def _blocks(db, settings=None, debt=None):
    settings = settings or RuntimeSettings()
    with db() as s:
        nights = sleep_phase.load_nights(
            s, settings.model_copy(update={"sleep_history_days": settings.history_days}), NOW
        )
        debt = debt if debt is not None else recent_sleep_debt(
            s, NOW, settings.target_sleep_hours, settings.sleep_debt_window_days
        )
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)
    times = grid(NOW - timedelta(hours=12), NOW + timedelta(hours=48), minutes=10)
    from circa.alertness.process_s import SleepWakeHistory

    history = SleepWakeHistory(
        [(n.start_ts, n.end_ts) for n in nights], NOW - timedelta(days=10),
        NOW + timedelta(hours=48),
    )
    curve = compute(times, OFFSET, history, np.random.default_rng(0).normal(4.4, 0.4, 100))
    return build_all(
        curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
        ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
        conf=assess(30, 95, 0.9, 20), settings=settings, offset=OFFSET,
        from_ts=NOW, to_ts=NOW + timedelta(hours=48),
        observed_nights=nights, debt=debt,
    )


# --- records ---------------------------------------------------------------


def test_every_recorded_night_gets_a_block(db):
    _seed_nights(db, [(1, 23.5, 7.0), (2, 23.0, 6.5), (3, 0.5, 8.0)])
    records = [b for b in _blocks(db) if b.kind == "sleep_actual"]
    assert len(records) == 3


def test_a_recorded_block_sits_where_the_night_actually_was(db):
    _seed_nights(db, [(1, 23.5, 7.0)])
    rec = next(b for b in _blocks(db) if b.kind == "sleep_actual")
    with db() as s:
        row = s.scalars(select(SleepSession)).one()
    assert rec.start == row.start_ts and rec.end == row.end_ts
    assert rec.end < NOW, "a recorded night must be in the past"


def test_a_recorded_block_carries_no_uncertainty_label(db):
    """It was measured. A +/- on it would be a lie about what is known."""
    _seed_nights(db, [(1, 23.5, 7.0)])
    rec = next(b for b in _blocks(db) if b.kind == "sleep_actual")
    assert "±" not in rec.title
    assert "7h" in rec.title


def test_records_survive_low_confidence_and_cold_start(db):
    """Confidence is about the forecast. There is nothing uncertain about a
    night the watch already measured, and blanking the calendar behind you on
    a bad day is exactly backwards - those are the days worth looking at."""
    from circa.alertness.process_s import SleepWakeHistory
    from circa.settings_store import LowConfidencePolicy

    _seed_nights(db, [(1, 23.5, 7.0), (2, 23.0, 6.5), (3, 0.5, 8.0)])
    times = grid(NOW - timedelta(hours=12), NOW + timedelta(hours=48), minutes=10)
    curve = compute(
        times, OFFSET,
        SleepWakeHistory([], NOW - timedelta(days=10), NOW + timedelta(hours=48)),
        np.random.default_rng(0).normal(4.4, 0.4, 100),
    )
    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)

    for policy in LowConfidencePolicy:
        settings = RuntimeSettings(low_confidence_policy=policy)
        blocks = build_all(
            curve=curve, dlmo_ts=dlmo, cbtmin_ts=dlmo + timedelta(hours=7),
            ci=(dlmo - timedelta(minutes=45), dlmo + timedelta(minutes=45)),
            conf=assess(2, 300, 0.4, None),   # cold start, poor data
            settings=settings, offset=OFFSET,
            from_ts=NOW, to_ts=NOW + timedelta(hours=48),
            observed_nights=_nights(db, settings),
            debt=SleepDebt(2.0, 5, 14, 7.0, -1.0),
        )
        assert [b for b in blocks if b.kind == "sleep_actual"], (
            f"records suppressed under {policy}"
        )


def _nights(db, settings):
    with db() as s:
        return sleep_phase.load_nights(
            s, settings.model_copy(update={"sleep_history_days": settings.history_days}), NOW
        )


def test_records_can_be_switched_off(db):
    _seed_nights(db, [(1, 23.5, 7.0)])
    blocks = _blocks(db, RuntimeSettings(show_recorded_sleep=False))
    assert not [b for b in blocks if b.kind == "sleep_actual"]


def test_history_window_bounds_how_far_back_records_go(db):
    _seed_nights(db, [(1, 23.5, 7.0), (5, 23.5, 7.0), (20, 23.5, 7.0)])
    blocks = _blocks(db, RuntimeSettings(history_days=7))
    records = [b for b in blocks if b.kind == "sleep_actual"]
    assert len(records) == 2, [b.start.isoformat() for b in records]


# --- forecasts stay in the future ------------------------------------------


def test_no_forecast_block_is_ever_written_into_the_past(db):
    """The asymmetry is the whole design: a record behind you is the point, a
    prediction behind you is clutter."""
    _seed_nights(db, [(1, 23.5, 7.0), (2, 23.0, 7.5), (3, 0.5, 8.0)])
    settings = RuntimeSettings()
    client = FakeCalendarClient()
    with db() as s:
        push(s, _blocks(db), settings, client=client)

    with db() as s:
        rows = list(s.scalars(select(CalendarBlock)))
    assert rows
    for row in rows:
        if row.kind in RETROSPECTIVE_KINDS:
            continue
        assert row.start_ts >= datetime.now(UTC) - timedelta(hours=2), (
            f"forecast {row.kind} written at {row.start_ts}, in the past"
        )


def test_records_are_pushed_to_the_calendar(db):
    _seed_nights(db, [(1, 23.5, 7.0), (2, 23.0, 7.5)])
    client = FakeCalendarClient()
    with db() as s:
        report = push(s, _blocks(db), RuntimeSettings(), client=client)
    keys = [
        e["body"]["extendedProperties"]["private"]["block_key"]
        for e in client.events.values()
    ]
    assert [k for k in keys if "sleep_actual" in k], keys
    assert report.created >= 2


def test_a_record_supersedes_the_forecast_for_the_same_night(db):
    """Once a night has happened, the prediction for it is not the interesting
    object - and leaving both on the calendar reads as a contradiction."""
    from circa.db.models import CalendarLink
    from circa.gcal.blocks import Block

    settings = RuntimeSettings()
    client = FakeCalendarClient()
    now = datetime.now(UTC)
    night_start = now - timedelta(hours=11)
    night_end = now - timedelta(hours=3)

    # The forecast, as it would have been stored when it was still ahead.
    cal_id = "cal-sleep@group.calendar.google.com"
    client.calendars[cal_id] = {"summary": "Circa · Sleep"}
    client.events["evt-forecast"] = {
        "calendar": cal_id,
        "body": {"extendedProperties": {"private": {
            "block_key": "sleep:sleep_window:2026-09-09", "kind": "sleep_window"}}},
    }
    with db() as s:
        s.add(CalendarLink(category="sleep", calendar_id=cal_id,
                           summary="Circa · Sleep", color_id="1", enabled=True))
        s.add(CalendarBlock(
            block_key="sleep:sleep_window:2026-09-09", category="sleep",
            kind="sleep_window", target_date=night_start.date(),
            start_ts=night_start - timedelta(minutes=30),
            end_ts=night_end + timedelta(minutes=30),
            title="Biological night (±40m)", description="predicted",
            google_event_id="evt-forecast", model_version="0.2.0", deleted=False,
        ))

    record = Block(
        key="sleep:sleep_actual:2026-09-10", category="sleep", kind="sleep_actual",
        start=night_start, end=night_end,
        title="Slept 7h 00m", description="recorded",
        supersedes_kinds=frozenset({"sleep_window"}),
    )
    with db() as s:
        push(s, [record], settings, client=client)

    live = [
        e["body"]["extendedProperties"]["private"]["kind"]
        for e in client.events.values()
    ]
    assert "sleep_actual" in live
    assert "sleep_window" not in live, "the superseded forecast is still on the calendar"

    with db() as s:
        row = s.scalar(select(CalendarBlock).where(
            CalendarBlock.block_key == "sleep:sleep_window:2026-09-09"))
        assert row.deleted is True


# --- debt changes the recommendation ---------------------------------------


def test_debt_sums_per_night_not_per_session(db):
    """A split night is one short night, not two - otherwise the app invents
    debt that was never owed."""
    day = (NOW - timedelta(days=1)).astimezone(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    with db() as s:
        for i, (offset_h, hours) in enumerate([(23.0, 4.0), (28.0, 3.5)]):
            start = (day + timedelta(hours=offset_h)).astimezone(UTC)
            end = start + timedelta(hours=hours)
            s.add(SleepSession(
                external_id=f"split/{i}", start_ts=start, end_ts=end,
                tz_name="America/Chicago", utc_offset_seconds=OFFSET,
                sleep_date=(day + timedelta(days=1)).date(), is_main_sleep=True,
                tst_minutes=hours * 60, midpoint_ts=start + (end - start) / 2,
            ))
    with db() as s:
        debt = recent_sleep_debt(s, NOW, target_hours=8.0, days=14)
    assert debt.nights == 1
    assert debt.hours == pytest.approx(0.5, abs=0.01)   # 8.0 - (4.0 + 3.5)


def test_a_short_week_lengthens_tonights_window(db):
    _seed_nights(db, [(d, 23.5, 6.0) for d in range(1, 8)])   # 2h short, seven nights
    settings = RuntimeSettings()
    with db() as s:
        debt = recent_sleep_debt(s, NOW, settings.target_sleep_hours,
                                 settings.sleep_debt_window_days)
    assert debt.hours == pytest.approx(14.0, abs=0.1)

    rested = _blocks(db, settings, debt=SleepDebt(0.0, 7, 14, 8.0, 0.0))
    tired = _blocks(db, settings, debt=debt)

    def night(bs):
        b = next(b for b in bs if b.kind == "sleep_window")
        return (b.end - b.start).total_seconds() / 3600

    assert night(tired) > night(rested), (
        f"debt did not lengthen the window: {night(tired)}h vs {night(rested)}h"
    )
    # Repaid in slices, not all at once.
    assert night(tired) - night(rested) <= settings.max_debt_payback_hours + 0.3


def test_a_surplus_never_shortens_the_window():
    """Sleeping less than target is never the recommendation."""
    surplus = SleepDebt(hours=-6.0, nights=7, window_days=14,
                        last_night_hours=9.0, last_night_delta=1.0)
    assert target_sleep_tonight(8.0, surplus, 0.25, 1.5) == 8.0


def test_debt_from_one_or_two_nights_is_not_acted_on():
    """Two nights is not a trend, and a recommendation built on it is noise."""
    thin = SleepDebt(hours=4.0, nights=2, window_days=14,
                     last_night_hours=6.0, last_night_delta=-2.0)
    assert not thin.is_meaningful
    assert target_sleep_tonight(8.0, thin, 0.25, 1.5) == 8.0


def test_payback_is_capped():
    huge = SleepDebt(hours=40.0, nights=14, window_days=14,
                     last_night_hours=5.0, last_night_delta=-3.0)
    assert target_sleep_tonight(8.0, huge, 0.25, 1.5) == pytest.approx(9.5)


def test_no_sleep_data_yields_an_honest_zero(db):
    with db() as s:
        debt = recent_sleep_debt(s, NOW, 8.0, 14)
    assert debt.nights == 0
    assert debt.hours == 0.0
    assert not debt.is_meaningful
    assert debt.last_night_hours is None


# --- last night must actually change today ---------------------------------


def _today_curve(db, last_night_hours):
    """Identical everything except how long the person slept last night."""
    from circa.alertness.process_s import SleepWakeHistory

    with db() as s:
        s.query(SleepSession).delete()
    # A steady fortnight, then one night of the given length.
    _seed_nights(db, [(d, 23.5, 8.0) for d in range(2, 15)])
    _seed_nights(db, [(1, 23.5, last_night_hours)])

    with db() as s:
        nights = sleep_phase.load_nights(
            s, RuntimeSettings(sleep_history_days=20), NOW
        )
    history = SleepWakeHistory(
        [(n.start_ts, n.end_ts) for n in nights],
        NOW - timedelta(days=16), NOW + timedelta(hours=24),
    )
    times = grid(NOW, NOW + timedelta(hours=8), minutes=30)
    return compute(times, OFFSET, history,
                   np.random.default_rng(0).normal(4.4, 0.4, 100))


def test_a_short_night_lowers_todays_predicted_energy(db):
    """The whole point of driving the homeostat from recorded sleep.

    If today's forecast is identical whether you slept four hours or nine, the
    system is not adapting to anything - it is replaying a population average
    with your name on it.
    """
    rested = _today_curve(db, 9.0)
    wrecked = _today_curve(db, 4.0)
    assert wrecked.detail["energy_peak"] < rested.detail["energy_peak"], (
        f"a 4h night predicted the same ceiling as a 9h one: "
        f"{wrecked.detail['energy_peak']} vs {rested.detail['energy_peak']}"
    )


def test_the_gap_between_a_short_and_long_night_is_not_trivial(db):
    rested = _today_curve(db, 9.0)
    wrecked = _today_curve(db, 4.0)
    gap = rested.detail["energy_peak"] - wrecked.detail["energy_peak"]
    assert gap > 4.0, f"five hours of lost sleep moved the ceiling by only {gap:.1f} points"


def test_sleep_pressure_at_wake_reflects_how_long_you_slept(db):
    """Monotone: every extra hour of sleep must leave less pressure behind."""
    peaks = [_today_curve(db, h).homeostatic[0] for h in (4.0, 6.0, 8.0, 9.5)]
    assert peaks == sorted(peaks, reverse=True), (
        f"sleep pressure not monotone in sleep length: {[round(p, 3) for p in peaks]}"
    )


def test_the_z_curve_is_still_about_timing_not_magnitude(db):
    """`alertness` is deliberately window-relative - it is what block detection
    thresholds against. The absolute reading lives in `energy`; conflating the
    two is what hid this bug in the first place."""
    rested = _today_curve(db, 9.0)
    wrecked = _today_curve(db, 4.0)
    assert abs(float(np.mean(rested.alertness))) < 0.05
    assert abs(float(np.mean(wrecked.alertness))) < 0.05


def test_the_curve_projects_the_same_night_the_calendar_recommends(db):
    """Otherwise the energy forecast quietly disagrees with the advice beside it."""
    from circa.pipeline import _project_sleep

    dlmo = datetime(2026, 9, 10, 21, 30, tzinfo=TZ).astimezone(UTC)

    class _Phase:
        dlmo_ts = dlmo

    projected = _project_sleep(
        [], _Phase(), RuntimeSettings(), dlmo + timedelta(hours=30), target_hours=8.75
    )
    assert projected
    start, end = projected[0]
    assert (end - start).total_seconds() / 3600 == pytest.approx(8.75)

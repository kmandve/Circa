"""Normalisation: raw payloads -> the numbers the models consume."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select

from tests.conftest import hr_payload, make_raw, sleep_payload, steps_payload, utc

# --- sleep -----------------------------------------------------------------


def test_sleep_metrics_from_stages():
    """23:00 in bed, asleep 23:20, 20 min awake mid-night, final wake 07:00."""
    from circa.db.models import SleepSession
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    start = utc(2026, 9, 1, 23, 0)
    end = utc(2026, 9, 2, 7, 0)
    stages = [
        (start, start + timedelta(minutes=20), "AWAKE"),           # sleep latency
        (start + timedelta(minutes=20), utc(2026, 9, 2, 2, 0), "LIGHT"),
        (utc(2026, 9, 2, 2, 0), utc(2026, 9, 2, 2, 20), "AWAKE"),  # WASO
        (utc(2026, 9, 2, 2, 20), utc(2026, 9, 2, 4, 0), "DEEP"),
        (utc(2026, 9, 2, 4, 0), utc(2026, 9, 2, 7, 0), "REM"),
    ]

    with session_scope() as s:
        assert normalize_sleep(s, [make_raw("sleep", sleep_payload(start, end, stages))]) == 1

    with session_scope() as s:
        row = s.scalars(select(SleepSession)).one()

    assert row.latency_minutes == 20
    assert row.waso_minutes == 20            # the mid-night wake only
    assert row.tst_minutes == 440            # 480 in bed - 20 latency - 20 WASO
    assert row.time_in_bed_minutes == 480
    assert row.minutes_deep == 100
    assert row.minutes_rem == 180
    assert abs(row.efficiency - 440 / 480) < 1e-9
    assert row.is_main_sleep is True

    # Midpoint spans onset (23:20) to final wake (07:00) -> 03:10.
    assert row.midpoint_ts == utc(2026, 9, 2, 3, 10)


def test_sleep_without_stages_treats_session_as_sleep():
    """CLASSIC records carry no stages; the session itself is the sleep."""
    from circa.db.models import SleepSession
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    start, end = utc(2026, 9, 1, 23, 0), utc(2026, 9, 2, 6, 0)
    with session_scope() as s:
        normalize_sleep(s, [make_raw("sleep", sleep_payload(start, end))])

    with session_scope() as s:
        row = s.scalars(select(SleepSession)).one()

    assert row.tst_minutes == 420
    assert row.waso_minutes == 0
    assert row.midpoint_ts == utc(2026, 9, 2, 2, 30)


def test_short_session_is_a_nap_not_main_sleep():
    from circa.db.models import SleepSession
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    start, end = utc(2026, 9, 1, 14, 0), utc(2026, 9, 1, 14, 40)
    with session_scope() as s:
        normalize_sleep(s, [make_raw("sleep", sleep_payload(start, end, external_id="nap/1"))])

    with session_scope() as s:
        assert s.scalars(select(SleepSession)).one().is_main_sleep is False


def test_revised_session_replaces_stages_rather_than_duplicating():
    """Google reconciles sleep records after the fact; a re-fetch must not stack."""
    from circa.db.models import SleepSession, SleepStage
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    start, end = utc(2026, 9, 1, 23, 0), utc(2026, 9, 2, 7, 0)
    v1 = sleep_payload(start, end, [(start, end, "LIGHT")], external_id="sleep/x")
    v2 = sleep_payload(
        start,
        end,
        [(start, utc(2026, 9, 2, 3, 0), "LIGHT"), (utc(2026, 9, 2, 3, 0), end, "DEEP")],
        external_id="sleep/x",
    )

    with session_scope() as s:
        normalize_sleep(s, [make_raw("sleep", v1)])
    with session_scope() as s:
        normalize_sleep(s, [make_raw("sleep", v2)])

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(SleepSession)) == 1
        assert s.scalar(select(func.count()).select_from(SleepStage)) == 2
        assert s.scalars(select(SleepSession)).one().minutes_deep == 240


# --- heart rate ------------------------------------------------------------


def test_hr_minute_uses_median_not_mean():
    """One artefact spike must not drag the bin; that is why we use a median."""
    from circa.db.models import HeartRateMinute
    from circa.db.session import session_scope
    from circa.normalize.heart_rate import normalize_heart_rate

    base = utc(2026, 9, 1, 12, 0)
    bpms = [60, 61, 60, 62, 200]  # last one is an artefact
    raws = [
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=5 * i), bpm))
        for i, bpm in enumerate(bpms)
    ]
    with session_scope() as s:
        normalize_heart_rate(s, raws)

    with session_scope() as s:
        row = s.scalars(select(HeartRateMinute)).one()

    assert row.bpm_median == 61      # mean would be 88.6
    assert row.n_samples == 5
    assert row.bpm_max == 200


def test_hr_implausible_values_are_dropped():
    from circa.db.models import HeartRateSample
    from circa.db.session import session_scope
    from circa.normalize.heart_rate import normalize_heart_rate

    base = utc(2026, 9, 1, 12, 0)
    raws = [
        make_raw("heart-rate", hr_payload(base, 5)),                       # too low
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=5), 300)),  # too high
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=10), 62)),  # fine
    ]
    with session_scope() as s:
        normalize_heart_rate(s, raws)

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateSample)) == 1


def test_hr_minute_rebuild_is_idempotent():
    """Re-ingesting the same window must not double-count or duplicate rows."""
    from circa.db.models import HeartRateMinute
    from circa.db.session import session_scope
    from circa.normalize.heart_rate import normalize_heart_rate

    base = utc(2026, 9, 1, 12, 0)
    raws = [
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=5 * i), 60 + i))
        for i in range(12)
    ]
    with session_scope() as s:
        normalize_heart_rate(s, raws)
    with session_scope() as s:
        normalize_heart_rate(s, raws)

    with session_scope() as s:
        rows = list(s.scalars(select(HeartRateMinute)))
    assert len(rows) == 1
    assert rows[0].n_samples == 12


def test_hr_active_fraction():
    from circa.db.models import HeartRateMinute
    from circa.db.session import session_scope
    from circa.normalize.heart_rate import normalize_heart_rate

    base = utc(2026, 9, 1, 12, 0)
    raws = [
        make_raw("heart-rate", hr_payload(base, 130, "ACTIVE")),
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=5), 128, "ACTIVE")),
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=10), 90, "SEDENTARY")),
        make_raw("heart-rate", hr_payload(base + timedelta(seconds=15), 88, "SEDENTARY")),
    ]
    with session_scope() as s:
        normalize_heart_rate(s, raws)

    with session_scope() as s:
        assert s.scalars(select(HeartRateMinute)).one().active_fraction == 0.5


# --- steps -----------------------------------------------------------------


def test_step_bouts_conserve_total_when_redistributed():
    """Spreading a bout across minutes must not create or destroy steps."""
    from circa.db.models import StepMinute
    from circa.db.session import session_scope
    from circa.normalize.activity import normalize_steps

    start = utc(2026, 9, 1, 12, 0)
    raws = [
        make_raw("steps", steps_payload(start, start + timedelta(minutes=10), 1000)),
        make_raw("steps", steps_payload(start + timedelta(minutes=30),
                                        start + timedelta(minutes=33), 210)),
    ]
    with session_scope() as s:
        normalize_steps(s, raws)

    with session_scope() as s:
        total = s.scalar(select(func.sum(StepMinute.steps)))
        rows = list(s.scalars(select(StepMinute).order_by(StepMinute.ts)))

    assert abs(total - 1210) < 0.01
    assert rows[0].steps == 100          # 1000 over 10 minutes
    assert len(rows) == 13               # 10 + 3 minutes, gap not filled


def test_step_bout_straddling_minute_boundary_is_split_proportionally():
    from circa.db.models import StepMinute
    from circa.db.session import session_scope
    from circa.normalize.activity import normalize_steps

    # 12:00:30 -> 12:01:30 : half in each minute.
    start = utc(2026, 9, 1, 12, 0, 30)
    with session_scope() as s:
        normalize_steps(s, [make_raw("steps", steps_payload(start, start + timedelta(minutes=1), 100))])

    with session_scope() as s:
        rows = list(s.scalars(select(StepMinute).order_by(StepMinute.ts)))

    assert len(rows) == 2
    assert abs(rows[0].steps - 50) < 0.01
    assert abs(rows[1].steps - 50) < 0.01


# --- daily metrics ---------------------------------------------------------


def test_daily_sleep_temperature_is_stored_as_qc_only():
    """Skin temperature is one number per night - never a phase channel."""
    from circa.db.models import DailyMetric
    from circa.db.session import session_scope
    from circa.normalize.vitals import normalize_daily

    payload = {
        "dailySleepTemperatureDerivations": {
            "date": "2026-09-02",
            "nightlyTemperatureCelsius": 33.4,
            "baselineTemperatureCelsius": 33.1,
            "relativeNightlyStddev30dCelsius": 0.28,
        }
    }
    with session_scope() as s:
        normalize_daily(s, "daily-sleep-temperature-derivations",
                        [make_raw("daily-sleep-temperature-derivations", payload)])

    with session_scope() as s:
        row = s.scalars(select(DailyMetric)).one()

    assert row.value == 33.4
    assert row.metric_date.isoformat() == "2026-09-02"
    # The baseline is retained so an illness/anomaly flag can use the delta.
    assert row.extra["baselineTemperatureCelsius"] == 33.1


def test_daily_metric_upsert_overwrites_revised_value():
    from circa.db.models import DailyMetric
    from circa.db.session import session_scope
    from circa.normalize.vitals import normalize_daily

    def payload(bpm):
        return {"dailyRestingHeartRate": {"date": "2026-09-02", "beatsPerMinute": bpm}}

    with session_scope() as s:
        normalize_daily(s, "daily-resting-heart-rate",
                        [make_raw("daily-resting-heart-rate", payload(58))])
    with session_scope() as s:
        normalize_daily(s, "daily-resting-heart-rate",
                        [make_raw("daily-resting-heart-rate", payload(56))])

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(DailyMetric)) == 1
        assert s.scalars(select(DailyMetric)).one().value == 56

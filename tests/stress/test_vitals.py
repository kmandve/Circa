"""Daily metrics and HRV, against the payload shapes Google actually sends.

These feed the illness flag and the OOD detector, so a wrong value here does
not raise - it quietly changes how much the model trusts a night.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from circa.db.models import DailyMetric, ExerciseSession, HrvSample
from circa.normalize.vitals import normalize_daily, normalize_exercise, normalize_hrv
from tests.conftest import make_raw, utc


def _norm_daily(db, data_type, payloads):
    with db() as s:
        raws = [make_raw(data_type, p) for p in payloads]
        for r in raws:
            s.add(r)
        s.flush()
        return normalize_daily(s, data_type, raws)


def _values(db, metric):
    with db() as s:
        return {
            r.metric_date.isoformat(): r.value
            for r in s.scalars(select(DailyMetric).where(DailyMetric.metric == metric))
        }


# --- the real shapes --------------------------------------------------------

REAL = [
    ("daily-resting-heart-rate", {"dailyRestingHeartRate": {
        "date": {"year": 2026, "month": 9, "day": 9}, "beatsPerMinute": "66",
        "dailyRestingHeartRateMetadata": {"calculationMethod": "WITH_SLEEP"}}}, 66.0),
    ("daily-heart-rate-variability", {"dailyHeartRateVariability": {
        "date": {"year": 2026, "month": 9, "day": 9},
        "averageHeartRateVariabilityMilliseconds": 63.3,
        "nonRemHeartRateBeatsPerMinute": "58", "entropy": 2.986,
        "deepSleepRootMeanSquareOfSuccessiveDifferencesMilliseconds": 56.7}}, 63.3),
    ("daily-respiratory-rate", {"dailyRespiratoryRate": {
        "date": {"year": 2026, "month": 9, "day": 9}, "breathsPerMinute": 19}}, 19.0),
    ("daily-oxygen-saturation", {"dailyOxygenSaturation": {
        "date": {"year": 2026, "month": 9, "day": 9}, "averagePercentage": 97.4,
        "lowerBoundPercentage": 94.3, "upperBoundPercentage": 100}}, 97.4),
    ("daily-sleep-temperature-derivations", {"dailySleepTemperatureDerivations": {
        "date": {"year": 2026, "month": 9, "day": 9},
        "nightlyTemperatureCelsius": 34.407, "baselineTemperatureCelsius": "NaN",
        "relativeNightlyStddev30dCelsius": "NaN"}}, 34.407),
]


@pytest.mark.parametrize("data_type,payload,expected", REAL)
def test_the_right_field_is_picked_from_a_real_payload(db, data_type, payload, expected):
    assert _norm_daily(db, data_type, [payload]) == 1
    assert _values(db, data_type)["2026-09-09"] == pytest.approx(expected, abs=1e-3)


def test_a_nan_baseline_does_not_poison_the_temperature(db):
    """`baselineTemperatureCelsius` is the literal string "NaN" until ~30
    nights. Reading it as a float would silently break every comparison."""
    payload = REAL[-1][1]
    _norm_daily(db, "daily-sleep-temperature-derivations", [payload])
    with db() as s:
        row = s.scalars(select(DailyMetric)).one()
    assert row.value == pytest.approx(34.407, abs=1e-3)
    assert row.extra["baselineTemperatureCelsius"] == "NaN"


# --- the schema moving underneath us ---------------------------------------


def test_a_renamed_field_does_not_get_the_wrong_number(db):
    """The fallback used to take the first plausible scalar in the body.

    For heart-rate variability that is `nonRemHeartRateBeatsPerMinute` - so a
    renamed field would have stored a heart rate, in beats per minute, as an
    HRV in milliseconds, and nothing downstream could tell.
    """
    payload = {"dailyHeartRateVariability": {
        "date": {"year": 2026, "month": 9, "day": 9},
        "someNewNameForRmssd": 63.3,
        "nonRemHeartRateBeatsPerMinute": "58",
        "entropy": 2.986,
    }}
    _norm_daily(db, "daily-heart-rate-variability", [payload])
    stored = _values(db, "daily-heart-rate-variability")["2026-09-09"]
    assert stored != 58.0, "a heart rate was stored as heart-rate variability"
    assert stored is None, "guessed a value from an ambiguous payload"

    # The payload is kept, so the day can be recovered once the name is known.
    with db() as s:
        row = s.scalars(select(DailyMetric)).one()
    assert row.extra["someNewNameForRmssd"] == 63.3


def test_an_unambiguous_new_type_still_works(db):
    """One scalar in the body is not a guess."""
    payload = {"dailyThingWeHaveNeverSeen": {
        "date": {"year": 2026, "month": 9, "day": 9}, "someValue": 42.0}}
    _norm_daily(db, "daily-thing-we-have-never-seen", [payload])
    assert _values(db, "daily-thing-we-have-never-seen")["2026-09-09"] == 42.0


@pytest.mark.parametrize("payload", [
    {}, {"dailyRestingHeartRate": {}},
    {"dailyRestingHeartRate": {"beatsPerMinute": 60}},          # no date
    {"dailyRestingHeartRate": {"date": "not-a-date", "beatsPerMinute": 60}},
    {"dailyRestingHeartRate": {"date": {"year": 2026}, "beatsPerMinute": "NaN"}},
])
def test_malformed_daily_payloads_are_skipped_not_fatal(db, payload):
    _norm_daily(db, "daily-resting-heart-rate", [payload])


def test_a_revised_day_overwrites_rather_than_duplicating(db):
    base = {"date": {"year": 2026, "month": 9, "day": 9}}
    _norm_daily(db, "daily-resting-heart-rate",
                [{"dailyRestingHeartRate": {**base, "beatsPerMinute": 66}}])
    _norm_daily(db, "daily-resting-heart-rate",
                [{"dailyRestingHeartRate": {**base, "beatsPerMinute": 61}}])
    with db() as s:
        rows = list(s.scalars(select(DailyMetric)))
    assert len(rows) == 1 and rows[0].value == 61.0


# --- HRV --------------------------------------------------------------------


def test_hrv_samples_are_stored_and_deduplicated(db):
    ts = utc(2026, 9, 9, 4, 0)
    # Two distinct raw payloads that describe the same instant - a revision.
    payloads = [
        {"heartRateVariability": {"sampleTime": ts.isoformat().replace("+00:00", "Z"),
                                  "rootMeanSquareOfSuccessiveDifferencesMilliseconds": 55.0}},
        {"heartRateVariability": {"sampleTime": ts.isoformat().replace("+00:00", "Z"),
                                  "rootMeanSquareOfSuccessiveDifferencesMilliseconds": 55.0,
                                  "standardDeviationMilliseconds": 41.0}},
    ]
    with db() as s:
        raws = [make_raw("heart-rate-variability", p) for p in payloads]
        for r in raws:
            s.add(r)
        s.flush()
        normalize_hrv(s, raws)
    with db() as s:
        assert len(list(s.scalars(select(HrvSample)))) == 1


def test_hrv_without_any_usable_number_is_skipped(db):
    payload = {"heartRateVariability": {
        "sampleTime": "2026-09-09T04:00:00Z", "somethingElse": 1}}
    with db() as s:
        raw = make_raw("heart-rate-variability", payload)
        s.add(raw)
        s.flush()
        assert normalize_hrv(s, [raw]) == 0


# --- exercise ---------------------------------------------------------------


def test_exercise_sessions_upsert_by_external_id(db):
    start = utc(2026, 9, 9, 17, 0)
    def payload(minutes):
        return {"exercise": {
            "name": "exercise/1",
            "interval": {"startTime": start.isoformat().replace("+00:00", "Z"),
                         "endTime": (start + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")},
            "activityType": "RUN", "calories": 300, "averageHeartRate": 145, "steps": 4000}}
    for minutes in (30, 45):
        with db() as s:
            raw = make_raw("exercise", payload(minutes))
            s.add(raw)
            s.flush()
            normalize_exercise(s, [raw])
    with db() as s:
        rows = list(s.scalars(select(ExerciseSession)))
    assert len(rows) == 1
    assert rows[0].end_ts - rows[0].start_ts == timedelta(minutes=45)


@pytest.mark.parametrize("payload", [
    {"exercise": {}},
    {"exercise": {"interval": {"startTime": "2026-09-09T17:00:00Z"}}},         # no end
    {"exercise": {"interval": {"startTime": "2026-09-09T18:00:00Z",
                               "endTime": "2026-09-09T17:00:00Z"}}},           # reversed
])
def test_malformed_exercise_is_skipped(db, payload):
    with db() as s:
        raw = make_raw("exercise", payload)
        s.add(raw)
        s.flush()
        assert normalize_exercise(s, [raw]) == 0

"""Normalisation against the messy shapes a real wearable actually emits.

Fitbit splits nights, revises them hours later, records naps, and occasionally
sends a session with no usable stage data at all. Each of those turns into a
number the phase estimate trusts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from circa.db.models import RawDataPoint, SleepSession, SleepStage
from circa.normalize.sleep import normalize_sleep
from tests.conftest import sleep_payload, utc

TZ = ZoneInfo("America/Chicago")


def _norm(db, payloads):
    """Store then normalise, the way the poller does.

    Goes through persist_raw so a repeated identical payload is deduplicated by
    content hash rather than violating the unique constraint - which is exactly
    what the real ingest path does on every overlapping poll.
    """
    from circa.ingest.datatypes import BY_NAME
    from circa.ingest.store import persist_raw

    with db() as s:
        persist_raw(s, BY_NAME["sleep"], list(payloads))
    with db() as s:
        raws = list(s.scalars(select(RawDataPoint).where(RawDataPoint.data_type == "sleep")))
        return normalize_sleep(s, raws)


def _sessions(db):
    with db() as s:
        return list(s.scalars(select(SleepSession).order_by(SleepSession.start_ts)))


# --- revision ---------------------------------------------------------------


def test_a_revised_session_updates_in_place(db):
    start = utc(2026, 9, 8, 4, 0)
    _norm(db, [sleep_payload(start, start + timedelta(hours=7), external_id="sleep/1")])
    _norm(db, [sleep_payload(start, start + timedelta(hours=8, minutes=30),
                             external_id="sleep/1")])
    rows = _sessions(db)
    assert len(rows) == 1
    assert rows[0].end_ts - rows[0].start_ts == timedelta(hours=8, minutes=30)


def test_revised_stages_replace_rather_than_accumulate(db):
    start = utc(2026, 9, 8, 4, 0)
    stages_a = [(start, start + timedelta(hours=3), "LIGHT"),
                (start + timedelta(hours=3), start + timedelta(hours=7), "DEEP")]
    stages_b = [(start, start + timedelta(hours=2), "LIGHT"),
                (start + timedelta(hours=2), start + timedelta(hours=5), "REM"),
                (start + timedelta(hours=5), start + timedelta(hours=7), "DEEP")]
    _norm(db, [sleep_payload(start, start + timedelta(hours=7), stages_a, "sleep/1")])
    _norm(db, [sleep_payload(start, start + timedelta(hours=7), stages_b, "sleep/1")])
    with db() as s:
        assert len(list(s.scalars(select(SleepStage)))) == 3


# --- malformed --------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    {"sleep": {}},
    {"sleep": {"interval": {}}},
    {"sleep": {"interval": {"startTime": "not-a-time", "endTime": "also-not"}}},
    {"sleep": {"interval": {"startTime": "2026-09-08T04:00:00Z"}}},          # no end
    {"sleep": {"interval": {"startTime": "2026-09-08T12:00:00Z",
                            "endTime": "2026-09-08T04:00:00Z"}}},            # end < start
    {"sleep": {"interval": {"startTime": "2026-09-08T04:00:00Z",
                            "endTime": "2026-09-08T04:00:00Z"}}},            # zero length
    {},
])
def test_unusable_sessions_are_skipped_not_fatal(db, bad):
    assert _norm(db, [bad]) == 0
    assert _sessions(db) == []


def test_a_stage_with_a_junk_duration_does_not_take_down_the_session(db):
    """`float(seconds)` on a malformed duration raised out of normalisation and
    lost the whole night, not just the one stage."""
    start = utc(2026, 9, 8, 4, 0)
    payload = {
        "sleep": {
            "name": "sleep/1",
            "interval": {"startTime": start.isoformat().replace("+00:00", "Z"),
                         "endTime": (start + timedelta(hours=8)).isoformat().replace("+00:00", "Z")},
            "stages": [
                {"startTime": start.isoformat().replace("+00:00", "Z"),
                 "duration": {"seconds": "not-a-number"}, "type": "LIGHT"},
                {"interval": {
                    "startTime": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                    "endTime": (start + timedelta(hours=4)).isoformat().replace("+00:00", "Z")},
                 "type": "DEEP"},
            ],
        }
    }
    assert _norm(db, [payload]) == 1
    rows = _sessions(db)
    assert len(rows) == 1
    with db() as s:
        stages = list(s.scalars(select(SleepStage)))
    assert len(stages) == 1 and stages[0].stage == "DEEP"


# --- split nights and naps --------------------------------------------------


def test_a_nap_is_not_counted_as_a_night(db):
    night = utc(2026, 9, 8, 4, 0)
    nap = utc(2026, 9, 8, 19, 0)   # 14:00 local
    _norm(db, [
        sleep_payload(night, night + timedelta(hours=8), external_id="sleep/night"),
        sleep_payload(nap, nap + timedelta(minutes=35), external_id="sleep/nap"),
    ])
    rows = _sessions(db)
    assert len(rows) == 2
    mains = [r for r in rows if r.is_main_sleep]
    assert len(mains) == 1, "the nap was treated as a main sleep"
    assert mains[0].external_id == "sleep/night"


def test_a_night_split_into_two_sessions_does_not_double_count(db):
    """Fitbit splits a night at a long awakening. Both halves can clear the
    main-sleep threshold, and the sleep channel then averages two midpoints
    from the same night as though they were two nights."""
    from circa.phase import sleep_phase
    from circa.settings_store import RuntimeSettings

    a = utc(2026, 9, 8, 4, 0)
    b = utc(2026, 9, 8, 9, 30)
    _norm(db, [
        sleep_payload(a, a + timedelta(hours=5), external_id="sleep/a"),
        sleep_payload(b, b + timedelta(hours=3, minutes=30), external_id="sleep/b"),
    ])
    with db() as s:
        nights = sleep_phase.load_nights(s, RuntimeSettings(), as_of=utc(2026, 9, 9))
        dates = [n.sleep_date for n in nights]
        assert len(set(dates)) == len(dates), (
            f"two sessions from one night both counted: {dates}"
        )


# --- dates and zones --------------------------------------------------------


def test_night_is_attributed_to_the_local_date_of_waking(db):
    # 23:30 local on the 7th -> 07:30 local on the 8th.
    start = datetime(2026, 9, 7, 23, 30, tzinfo=TZ).astimezone(UTC)
    _norm(db, [sleep_payload(start, start + timedelta(hours=8), external_id="s")])
    assert _sessions(db)[0].sleep_date.isoformat() == "2026-09-08"


def test_a_night_spanning_spring_forward_keeps_its_real_duration(db):
    """2026-03-08: 02:00 local becomes 03:00. A 23:00->07:00 night is 7 real
    hours, not 8, and the stored duration must be the real one."""
    start = datetime(2026, 3, 7, 23, 0, tzinfo=TZ).astimezone(UTC)
    end = datetime(2026, 3, 8, 7, 0, tzinfo=TZ).astimezone(UTC)
    _norm(db, [sleep_payload(start, end, external_id="dst")])
    row = _sessions(db)[0]
    assert row.end_ts - row.start_ts == timedelta(hours=7)
    assert row.time_in_bed_minutes == pytest.approx(420, abs=1)


def test_a_night_spanning_fall_back_keeps_its_real_duration(db):
    """2026-11-01: 02:00 local repeats, so 23:00->07:00 is 9 real hours."""
    start = datetime(2026, 10, 31, 23, 0, tzinfo=TZ).astimezone(UTC)
    end = datetime(2026, 11, 1, 7, 0, tzinfo=TZ).astimezone(UTC)
    _norm(db, [sleep_payload(start, end, external_id="dst2")])
    row = _sessions(db)[0]
    assert row.end_ts - row.start_ts == timedelta(hours=9)


def test_midpoint_is_the_real_midpoint(db):
    start = utc(2026, 9, 8, 4, 0)
    _norm(db, [sleep_payload(start, start + timedelta(hours=8), external_id="m")])
    row = _sessions(db)[0]
    assert row.midpoint_ts == start + timedelta(hours=4)


def test_normalising_the_same_payload_repeatedly_is_stable(db):
    start = utc(2026, 9, 8, 4, 0)
    stages = [(start, start + timedelta(hours=8), "LIGHT")]
    payload = sleep_payload(start, start + timedelta(hours=8), stages, "stable")
    for _ in range(4):
        _norm(db, [payload])
    assert len(_sessions(db)) == 1
    with db() as s:
        assert len(list(s.scalars(select(SleepStage)))) == 1

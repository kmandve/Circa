"""Timestamp handling.

These are the highest-value tests in the project. A one-hour timestamp error is
physically indistinguishable from a real one-hour circadian phase shift, so a
silent DST or timezone bug would not look like a bug — it would look like a
finding.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from tests.conftest import make_raw, sleep_payload, utc


def test_naive_datetime_is_rejected():
    """Storing a naive datetime must fail loudly rather than assume a zone."""
    from circa.db.models import HeartRateSample
    from circa.db.session import session_scope

    with pytest.raises(Exception) as exc_info, session_scope() as s:  # noqa: PT011
        s.add(HeartRateSample(ts=datetime(2026, 9, 1, 12, 0), bpm=60))
    assert "naive" in str(exc_info.value).lower()


def test_aware_datetime_round_trips_as_utc():
    from circa.db.models import HeartRateSample
    from circa.db.session import session_scope

    chicago = ZoneInfo("America/Chicago")
    local = datetime(2026, 9, 1, 7, 30, tzinfo=chicago)

    with session_scope() as s:
        s.add(HeartRateSample(ts=local, bpm=58))

    with session_scope() as s:
        row = s.scalars(select(HeartRateSample)).one()

    assert row.ts.tzinfo is not None
    assert row.ts == local              # same instant
    assert row.ts.hour == 12            # stored and returned as UTC


def test_dst_spring_forward_does_not_fabricate_a_phase_shift():
    """Two nights either side of a DST transition, both 23:00->07:00 local.

    Local wall-clock timing is identical, so the *UTC* midpoints must differ by
    exactly one hour — and crucially, the model must be able to see that,
    because the true circadian phase did not move. Deriving phase from bare
    local timestamps would report no change; deriving it from UTC alone would
    report a spurious one-hour shift. Storing both is what makes the difference
    visible.
    """
    from circa.db.models import SleepSession
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    chicago = ZoneInfo("America/Chicago")
    # US DST began 2026-03-08.
    before = datetime(2026, 3, 6, 23, 0, tzinfo=chicago)
    after = datetime(2026, 3, 10, 23, 0, tzinfo=chicago)

    payloads = []
    for i, local_start in enumerate((before, after)):
        local_end = local_start + timedelta(hours=8)
        payloads.append(
            sleep_payload(
                local_start.astimezone(UTC),
                local_end.astimezone(UTC),
                external_id=f"sleep/dst-{i}",
            )
        )

    with session_scope() as s:
        normalize_sleep(s, [make_raw("sleep", p) for p in payloads])

    with session_scope() as s:
        rows = list(s.scalars(select(SleepSession).order_by(SleepSession.start_ts)))

    assert len(rows) == 2
    utc_gap = (rows[1].midpoint_ts - rows[0].midpoint_ts).total_seconds() / 3600
    # 4 calendar days minus the hour lost to DST.
    assert abs(utc_gap - (96 - 1)) < 0.01

    # The stored offsets are what let the phase layer tell "the clocks changed"
    # from "the person shifted".
    assert rows[0].utc_offset_seconds == -6 * 3600  # CST
    assert rows[1].utc_offset_seconds == -5 * 3600  # CDT

    # Same local wall-clock time on both nights.
    local_mids = [r.midpoint_ts.astimezone(chicago) for r in rows]
    assert local_mids[0].hour == local_mids[1].hour
    assert local_mids[0].minute == local_mids[1].minute


def test_sleep_attributed_to_local_wake_date_across_midnight():
    """A 23:00->07:00 sleep belongs to the morning it ended, not the night it began."""
    from circa.db.models import SleepSession
    from circa.db.session import session_scope
    from circa.normalize.sleep import normalize_sleep

    chicago = ZoneInfo("America/Chicago")
    start = datetime(2026, 9, 1, 23, 15, tzinfo=chicago)
    end = datetime(2026, 9, 2, 7, 5, tzinfo=chicago)

    with session_scope() as s:
        normalize_sleep(
            s, [make_raw("sleep", sleep_payload(start.astimezone(UTC), end.astimezone(UTC)))]
        )

    with session_scope() as s:
        row = s.scalars(select(SleepSession)).one()

    assert row.sleep_date.isoformat() == "2026-09-02"


def test_parse_ts_handles_the_shapes_google_actually_returns():
    from circa.ingest.parsing import parse_ts

    expected = utc(2026, 9, 1, 12, 30, 0)
    assert parse_ts("2026-09-01T12:30:00Z") == expected
    assert parse_ts("2026-09-01T12:30:00+00:00") == expected
    assert parse_ts("2026-09-01T07:30:00-05:00") == expected
    assert parse_ts("2026-09-01T12:30:00") == expected            # civil, assumed UTC
    assert parse_ts({"seconds": int(expected.timestamp())}) == expected
    assert parse_ts(expected.timestamp()) == expected             # epoch seconds
    assert parse_ts(expected.timestamp() * 1000) == expected      # epoch millis
    # Nanosecond precision must not blow up fromisoformat.
    assert parse_ts("2026-09-01T12:30:00.123456789Z") is not None
    assert parse_ts(None) is None
    assert parse_ts("not a timestamp") is None


def test_parse_offset_seconds():
    from circa.ingest.parsing import parse_offset_seconds

    assert parse_offset_seconds("2026-09-01T07:30:00-05:00") == -5 * 3600
    assert parse_offset_seconds("2026-09-01T12:30:00Z") == 0
    assert parse_offset_seconds("2026-09-01T12:30:00+0530") == 5 * 3600 + 30 * 60
    assert parse_offset_seconds("2026-09-01T12:30:00") is None

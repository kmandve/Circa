"""Hostile-input sweep of the payload parsers.

The module contract is explicit: "every extraction here tries a list of
plausible paths and returns None rather than raising". A parser that raises
takes the whole poll down, and because the watermark only advances on success,
one malformed point can wedge ingestion permanently. So the property under
test is simply: never raise, whatever arrives.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from circa.ingest.parsing import (
    as_float,
    deep_get,
    extract_interval,
    extract_point_time,
    first,
    parse_date,
    parse_duration_seconds,
    parse_local_offset_seconds,
    parse_offset_seconds,
    parse_ts,
)

# Anything Google's JSON could plausibly decode to, including the shapes that
# have actually shown up: "NaN" strings, protobuf durations, nested nulls.
json_atoms = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=40),
    st.sampled_from(
        ["NaN", "Infinity", "-Infinity", "", "   ", "Z", "z", "-18000s", "0s",
         "2026-09-10", "2026-09-10T12:00:00Z", "2026-13-45T99:99:99Z",
         "2026-09-10T12:00:00.123456789Z", "+00:00", "1e400", "٣", "12:00"]
    ),
)
json_values = st.recursive(
    json_atoms,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(
            st.one_of(
                st.text(max_size=12),
                st.sampled_from(
                    ["seconds", "nanos", "physicalTime", "civilTime", "value",
                     "year", "month", "day", "time", "date", "hours", "minutes",
                     "startTime", "endTime", "interval", "utcOffset", "count"]
                ),
            ),
            children,
            max_size=5,
        ),
    ),
    max_leaves=12,
)

PARSERS = [
    parse_ts,
    parse_date,
    parse_duration_seconds,
    parse_offset_seconds,
    parse_local_offset_seconds,
    as_float,
]


@settings(max_examples=600)
@given(json_values)
def test_no_parser_raises_on_arbitrary_json(value):
    for fn in PARSERS:
        fn(value)


@settings(max_examples=300)
@given(json_values)
def test_extractors_survive_arbitrary_json(value):
    obj = value if isinstance(value, dict) else {"sleep": value}
    extract_point_time(obj, "sleep")
    extract_interval(obj.get("sleep") if isinstance(obj.get("sleep"), dict) else obj)
    deep_get(obj, "a.b.c")
    first(obj, "a", "b.c")


# --- specific shapes that have bitten before --------------------------------


@pytest.mark.parametrize(
    "value",
    [
        {"seconds": 2**62},
        {"seconds": -(2**62)},
        {"seconds": 10**18, "nanos": 10**9},
        1e20,
        -1e20,
        2**63 - 1,
        float("inf"),
        float("-inf"),
        float("nan"),
        "9999999999999999999999",
        {"physicalTime": {"seconds": 10**19}},
        {"year": 999999, "month": 1, "day": 1},
        {"year": 2026, "month": 0, "day": 0},
        {"civilTime": {"date": {"year": 2026, "month": 2, "day": 30}, "time": {}}},
    ],
)
def test_out_of_range_timestamps_are_absent_not_fatal(value):
    """A single absurd point must not be able to wedge the poller.

    The watermark advances only on a clean pass, so an exception here stops
    ingestion for good rather than skipping one bad row.
    """
    assert parse_ts(value) is None or isinstance(parse_ts(value), datetime)


def test_boolean_is_not_a_timestamp():
    """`isinstance(True, int)` is True, so a stray bool parsed as epoch second."""
    assert parse_ts(True) is None
    assert parse_ts(False) is None


def test_nan_string_is_absent_not_poison():
    # baselineTemperatureCelsius is the literal string "NaN" until ~30 nights.
    assert as_float("NaN") is None
    assert as_float(float("nan")) is None
    assert as_float("inf") is None


def test_z_suffix_is_not_a_local_offset():
    """A Z timestamp says nothing about the user's wall clock."""
    assert parse_local_offset_seconds("2026-09-10T12:00:00Z") is None
    assert parse_offset_seconds("2026-09-10T12:00:00Z") == 0
    assert parse_local_offset_seconds("2026-09-10T12:00:00-05:00") == -18000


def test_duration_string_round_trips():
    assert parse_duration_seconds("-18000s") == -18000
    assert parse_duration_seconds("0s") == 0
    assert parse_duration_seconds({"seconds": 3600}) == 3600
    assert parse_duration_seconds("not-a-duration") is None


def test_subsecond_precision_beyond_micros_is_trimmed():
    ts = parse_ts("2026-09-10T12:00:00.123456789Z")
    assert ts == datetime(2026, 9, 10, 12, 0, 0, 123456, tzinfo=UTC)

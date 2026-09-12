"""Raw store dedup and poller watermark behaviour.

The property being protected: intermittent uptime must be safe. A collector
that is down for days should backfill on restart with no gap and no duplicates,
because that is what makes free, non-always-on hosting a reasonable choice.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from tests.conftest import hr_payload, sleep_payload, utc

# --- raw store -------------------------------------------------------------


def test_identical_payload_is_only_stored_once():
    """Overlap re-fetching is deliberate, so dedup has to be reliable."""
    from circa.db.models import RawDataPoint
    from circa.db.session import session_scope
    from circa.ingest.datatypes import BY_NAME
    from circa.ingest.store import persist_raw

    dt = BY_NAME["heart-rate"]
    points = [hr_payload(utc(2026, 9, 1, 12, 0), 60)]

    with session_scope() as s:
        inserted, dupes = persist_raw(s, dt, points)
    assert (inserted, dupes) == (1, 0)

    with session_scope() as s:
        inserted, dupes = persist_raw(s, dt, points)
    assert (inserted, dupes) == (0, 1)

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(RawDataPoint)) == 1


def test_revised_payload_is_stored_alongside_the_original():
    """A changed record is new content, so both versions are kept for audit."""
    from circa.db.models import RawDataPoint
    from circa.db.session import session_scope
    from circa.ingest.datatypes import BY_NAME
    from circa.ingest.store import persist_raw

    dt = BY_NAME["sleep"]
    start, end = utc(2026, 9, 1, 23, 0), utc(2026, 9, 2, 7, 0)
    v1 = sleep_payload(start, end, external_id="sleep/x")
    v2 = sleep_payload(start, end + timedelta(minutes=12), external_id="sleep/x")

    with session_scope() as s:
        persist_raw(s, dt, [v1])
        persist_raw(s, dt, [v2])

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(RawDataPoint)) == 2


def test_duplicates_within_one_batch_are_collapsed():
    from circa.db.session import session_scope
    from circa.ingest.datatypes import BY_NAME
    from circa.ingest.store import persist_raw

    dt = BY_NAME["heart-rate"]
    point = hr_payload(utc(2026, 9, 1, 12, 0), 60)

    with session_scope() as s:
        inserted, _ = persist_raw(s, dt, [point, point, point])
    assert inserted == 1


# --- poller ----------------------------------------------------------------


class FakeClient:
    """Stands in for HealthClient, recording the windows it was asked for."""

    def __init__(self, points_by_type: dict[str, list[dict]], last_sync: datetime | None = None):
        self.points_by_type = points_by_type
        self.last_sync = last_sync or datetime.now(UTC)
        self.windows: list[tuple[str, datetime, datetime]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def paired_devices(self):
        return [{"id": "fitbit-air-1", "deviceName": "Fitbit Air",
                 "lastSyncTime": self.last_sync.isoformat().replace("+00:00", "Z")}]

    def fetch(self, dt, start, end, page_size=1000, max_pages=500):
        from circa.ingest.health_api import FetchResult

        self.windows.append((dt.name, start, end))
        points = [
            p for p in self.points_by_type.get(dt.name, [])
            if start <= _point_time(p) < end
        ]
        return FetchResult(data_type=dt.name, points=points, pages=1, server_filtered=True)


def _point_time(payload: dict) -> datetime:
    from circa.ingest.parsing import extract_point_time

    for field in ("heartRate", "sleep", "steps"):
        ts = extract_point_time(payload, field)
        if ts:
            return ts
    raise AssertionError(f"no timestamp in {payload}")


@pytest.fixture
def patch_client(monkeypatch):
    def _apply(fake):
        monkeypatch.setattr("circa.ingest.poller.HealthClient", lambda *a, **k: fake)
        return fake

    return _apply


def test_watermark_advances_and_second_run_finds_nothing_new(patch_client):
    from circa.db.models import SyncState
    from circa.db.session import session_scope
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    points = [hr_payload(now - timedelta(minutes=30 + i), 60 + (i % 5)) for i in range(20)]
    fake = patch_client(FakeClient({"heart-rate": points}, last_sync=now))

    first = sync_once(force=True, only=["heart-rate"])
    assert first.types[0].inserted == 20

    with session_scope() as s:
        watermark = s.get(SyncState, "heart-rate").watermark
    assert watermark is not None

    # Nothing new on the device, and the overlap window re-reads the same points.
    fake.windows.clear()
    second = sync_once(force=True, only=["heart-rate"])
    assert second.types[0].inserted == 0
    assert second.types[0].duplicates == 20

    # The second window starts at the watermark minus the overlap, not from scratch.
    _, start, _ = fake.windows[0]
    assert start > now - timedelta(days=1)


def test_downtime_is_backfilled_without_gaps(patch_client):
    """The collector is off for three days while the watch keeps syncing.

    This is the property that justifies not paying for always-on hosting: on
    restart the poller must resume from the last instant data could have
    existed, not from the present.
    """
    from circa.db.models import HeartRateMinute, SyncState
    from circa.db.session import session_scope
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    t0 = now - timedelta(days=4)

    # First run: the watch has synced up to t0, and that is all that exists.
    early = [hr_payload(t0 - timedelta(minutes=i), 60) for i in range(60)]
    fake = patch_client(FakeClient({"heart-rate": early}, last_sync=t0))
    sync_once(force=True, only=["heart-rate"])

    with session_scope() as s:
        watermark_before = s.get(SyncState, "heart-rate").watermark

    # The watermark must sit at the device sync time, not at "now" - otherwise
    # the three days that follow would be skipped forever.
    assert abs((watermark_before - t0).total_seconds()) < 15 * 60

    # Collector down for three days; the watch kept syncing the whole time.
    during_downtime = [
        hr_payload(t0 + timedelta(minutes=i), 62)
        for i in range(0, 60 * 24 * 3, 37)
    ]
    fake.points_by_type["heart-rate"] = early + during_downtime
    fake.last_sync = now

    report = sync_once(force=True, only=["heart-rate"])
    assert report.types[0].inserted == len(during_downtime)

    with session_scope() as s:
        watermark_after = s.get(SyncState, "heart-rate").watermark
        minutes = s.scalar(select(func.count()).select_from(HeartRateMinute))

    assert watermark_after > watermark_before
    assert minutes > 60


def test_watermark_never_advances_past_device_sync_time(patch_client):
    """Guards the clamp directly: an unsynced watch must not move the mark."""
    from circa.db.models import SyncState
    from circa.db.session import session_scope
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    stale = now - timedelta(hours=10)
    patch_client(FakeClient({"heart-rate": []}, last_sync=stale))

    sync_once(force=True, only=["heart-rate"])

    with session_scope() as s:
        watermark = s.get(SyncState, "heart-rate").watermark

    assert watermark <= stale
    assert watermark > stale - timedelta(minutes=30)


def test_high_frequency_types_are_skipped_when_watch_has_not_synced(patch_client):
    """Polling faster than the watch syncs just burns quota."""
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    patch_client(FakeClient({"heart-rate": []}, last_sync=now - timedelta(hours=6)))

    sync_once(force=True, only=["heart-rate"])       # establishes last_success
    report = sync_once(force=False, only=["heart-rate"])  # device has not synced since

    assert report.skipped_high_frequency is True
    assert report.types == []


def test_one_failing_type_does_not_abort_the_rest(patch_client):
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    fake = FakeClient({"heart-rate": [hr_payload(now - timedelta(minutes=5), 60)]}, last_sync=now)
    original_fetch = fake.fetch

    def flaky(dt, start, end, **kwargs):
        if dt.name == "steps":
            raise RuntimeError("simulated API failure")
        return original_fetch(dt, start, end, **kwargs)

    fake.fetch = flaky
    patch_client(fake)

    report = sync_once(force=True, only=["steps", "heart-rate"])
    by_type = {t.data_type: t for t in report.types}

    assert by_type["steps"].error is not None
    assert by_type["heart-rate"].inserted == 1


def test_device_sync_snapshot_is_recorded(patch_client):
    from circa.db.models import DeviceSync
    from circa.db.session import session_scope
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC).replace(microsecond=0)
    patch_client(FakeClient({}, last_sync=now))
    report = sync_once(force=True, only=["heart-rate"])

    assert report.device_synced_at == now
    with session_scope() as s:
        row = s.scalars(select(DeviceSync)).one()
    assert row.device_id == "fitbit-air-1"
    assert row.last_sync_time == now


# --- the three-names-per-type trap ----------------------------------------


def test_data_types_expose_three_distinct_names():
    """Path, filter and payload names differ; deriving one from another loses data.

    Regression guard: `heart-rate` is `heart_rate` in a filter expression but
    `heartRate` in the JSON body. Using the filter name to read the payload
    silently extracts nothing, which shows up much later as empty aggregates
    rather than as an error.
    """
    from circa.ingest.datatypes import BY_NAME

    hr = BY_NAME["heart-rate"]
    assert hr.path == "users/me/dataTypes/heart-rate/dataPoints"
    assert hr.filter_field == "heart_rate"
    assert hr.payload_key == "heartRate"

    hrv = BY_NAME["heart-rate-variability"]
    assert hrv.payload_key == "heartRateVariability"
    assert BY_NAME["daily-sleep-temperature-derivations"].payload_key == (
        "dailySleepTemperatureDerivations"
    )
    # Single-word types are the same in all three forms - that is what made the
    # original bug hide (sleep and steps worked fine).
    assert BY_NAME["sleep"].payload_key == "sleep"


def test_raw_point_time_is_extracted_for_every_registered_type():
    """A NULL point_time silently drops the row out of windowed re-normalisation."""
    from circa.db.models import RawDataPoint
    from circa.db.session import session_scope
    from circa.ingest.datatypes import BY_NAME
    from circa.ingest.store import persist_raw

    ts = utc(2026, 9, 1, 12, 0)
    cases = {
        "heart-rate": hr_payload(ts, 60),
        "sleep": sleep_payload(ts, ts + timedelta(hours=7)),
    }
    for name, payload in cases.items():
        with session_scope() as s:
            persist_raw(s, BY_NAME[name], [payload])

    with session_scope() as s:
        rows = list(s.scalars(select(RawDataPoint)))

    assert len(rows) == len(cases)
    assert all(r.point_time is not None for r in rows), (
        "point_time must resolve for every type, or windowed re-normalisation skips it"
    )


def test_watermark_holds_back_when_device_sync_time_is_unknown(patch_client):
    """A failed pairedDevices call must not let the watermark run to `now`.

    Re-reading a window is cheap; stepping over data the watch had not yet
    uploaded loses it permanently.
    """
    from circa.db.models import SyncState
    from circa.db.session import session_scope
    from circa.ingest.poller import sync_once

    now = datetime.now(UTC)
    fake = FakeClient({"heart-rate": []}, last_sync=now)

    def no_devices():
        from circa.ingest.health_api import HealthApiError

        raise HealthApiError("pairedDevices unavailable", status=403)

    fake.paired_devices = no_devices
    patch_client(fake)

    sync_once(force=True, only=["heart-rate"])
    with session_scope() as s:
        watermark = s.get(SyncState, "heart-rate").watermark

    assert watermark < now - timedelta(hours=5), (
        "without lastSyncTime the watermark must stay well behind the wall clock"
    )

"""The collector under adverse conditions.

Data that is never collected cannot be recovered later, so the properties that
matter are about loss: the watermark must never step over a window that still
had data coming, and a failure must never advance it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from circa.db.models import RawDataPoint, SyncState
from circa.ingest.poller import sync_once
from tests.conftest import hr_payload, sleep_payload, steps_payload
from tests.test_ingest import FakeClient, _point_time  # noqa: F401

NOW = datetime.now(UTC)


@pytest.fixture
def patch_client(monkeypatch):
    def _apply(fake):
        monkeypatch.setattr("circa.ingest.poller.HealthClient", lambda *a, **k: fake)
        return fake

    return _apply


class ScriptedClient(FakeClient):
    """A FakeClient that can fail, go unfiltered, or revise records."""

    def __init__(self, points_by_type, last_sync=None, *, unfiltered=(),
                 fail_types=(), fail_times=0, device_error=False):
        super().__init__(points_by_type, last_sync)
        self.unfiltered = set(unfiltered)
        self.fail_types = set(fail_types)
        self.fail_times = fail_times
        self.failures = 0
        self.device_error = device_error

    def paired_devices(self):
        if self.device_error:
            from circa.ingest.health_api import HealthApiError

            raise HealthApiError("no scope", status=403)
        return super().paired_devices()

    def fetch(self, dt, start, end, page_size=1000, max_pages=500):
        from circa.ingest.health_api import FetchResult

        if dt.name in self.fail_types and self.failures < self.fail_times:
            self.failures += 1
            from circa.ingest.health_api import HealthApiError

            raise HealthApiError("upstream 503", status=503)
        self.windows.append((dt.name, start, end))
        pts = self.points_by_type.get(dt.name, [])
        if dt.name in self.unfiltered:
            return FetchResult(dt.name, list(pts), pages=1, server_filtered=False)
        return FetchResult(
            dt.name,
            [p for p in pts if start <= _point_time(p) < end],
            pages=1, server_filtered=True,
        )


def _hr_over(start, hours, every_minutes=5):
    n = int(hours * 60 / every_minutes)
    return [hr_payload(start + timedelta(minutes=every_minutes * i), 60 + i % 10)
            for i in range(n)]


def _watermark(db, name):
    with db() as s:
        row = s.get(SyncState, name)
        return row.watermark if row else None


def _raw_count(db, name):
    with db() as s:
        return s.scalar(
            select(func.count()).select_from(RawDataPoint).where(
                RawDataPoint.data_type == name
            )
        )


# --- loss ------------------------------------------------------------------


def test_no_data_is_lost_across_repeated_short_outages(patch_client, db):
    """Restarting mid-stream must not leave a hole.

    The watermark is the only memory of what has been seen, so any window it
    steps over is gone permanently.
    """
    start = NOW - timedelta(hours=20)
    points = _hr_over(start, 18)
    client = ScriptedClient({"heart-rate": points}, last_sync=NOW)
    patch_client(client)

    for _ in range(5):  # five separate "process restarts"
        sync_once(only=["heart-rate"])

    stored = _raw_count(db, "heart-rate")
    assert stored == len(points), f"{len(points) - stored} points never landed"


def test_a_failing_type_does_not_advance_its_watermark(patch_client, db):
    points = _hr_over(NOW - timedelta(hours=6), 5)
    client = ScriptedClient({"heart-rate": points}, last_sync=NOW,
                            fail_types={"heart-rate"}, fail_times=99)
    patch_client(client)

    before = _watermark(db, "heart-rate")
    report = sync_once(only=["heart-rate"])
    after = _watermark(db, "heart-rate")

    assert report.errors
    assert after == before, "watermark moved despite the fetch failing"

    with db() as s:
        state = s.get(SyncState, "heart-rate")
        assert state.consecutive_failures == 1
        assert state.last_error


def test_recovery_after_failure_collects_everything(patch_client, db):
    points = _hr_over(NOW - timedelta(hours=6), 5)
    client = ScriptedClient({"heart-rate": points}, last_sync=NOW,
                            fail_types={"heart-rate"}, fail_times=1)
    patch_client(client)

    sync_once(only=["heart-rate"])      # fails
    sync_once(only=["heart-rate"])      # recovers
    assert _raw_count(db, "heart-rate") == len(points)
    with db() as s:
        assert s.get(SyncState, "heart-rate").consecutive_failures == 0


def test_watermark_never_passes_the_last_device_sync(patch_client, db):
    """Nothing exists in the API until the watch uploads it."""
    stale = NOW - timedelta(hours=9)
    client = ScriptedClient({"heart-rate": _hr_over(NOW - timedelta(hours=30), 20)},
                            last_sync=stale)
    patch_client(client)
    sync_once(only=["heart-rate"])
    assert _watermark(db, "heart-rate") <= stale


def test_watermark_holds_back_when_the_device_call_fails(patch_client, db):
    client = ScriptedClient({"heart-rate": _hr_over(NOW - timedelta(hours=10), 8)},
                            device_error=True)
    patch_client(client)
    sync_once(only=["heart-rate"])
    wm = _watermark(db, "heart-rate")
    assert wm is not None
    assert wm <= NOW - timedelta(hours=5), "watermark advanced with no freshness signal"


# --- unfiltered types -------------------------------------------------------


def test_unfiltered_type_still_advances_its_watermark(patch_client, db):
    """`exercise` is fetched with NO_FILTER, so the server returns everything.

    The loop breaks out of the window walk on the second pass to avoid
    re-reading the same pages - but if the watermark is only advanced from the
    windows actually walked, it creeps forward a fraction of the elapsed time
    per poll and the type falls further behind every day.
    """
    old = NOW - timedelta(days=40)
    points = [steps_payload(old + timedelta(days=d), old + timedelta(days=d, hours=1), 500)
              for d in range(40)]
    client = ScriptedClient({"steps": points}, last_sync=NOW, unfiltered={"steps"})
    patch_client(client)

    sync_once(only=["steps"])
    first_wm = _watermark(db, "steps")
    assert first_wm is not None
    assert first_wm >= NOW - timedelta(hours=7), (
        f"watermark stuck at {first_wm}, {(NOW - first_wm).days} days behind"
    )


def test_unfiltered_type_does_not_discard_the_points_it_fetched(patch_client, db):
    old = NOW - timedelta(days=30)
    points = [steps_payload(old + timedelta(days=d), old + timedelta(days=d, hours=1), 400)
              for d in range(30)]
    client = ScriptedClient({"steps": points}, last_sync=NOW, unfiltered={"steps"})
    patch_client(client)
    sync_once(only=["steps"])
    assert _raw_count(db, "steps") == len(points)


# --- revision ---------------------------------------------------------------


def test_a_revised_sleep_session_is_picked_up(patch_client, db):
    """Google rewrites sleep hours after the fact; the overlap window exists
    precisely so the revision is seen."""
    from circa.db.models import SleepSession

    start = NOW - timedelta(hours=30)
    first_version = sleep_payload(start, start + timedelta(hours=7), external_id="sleep/a")
    client = ScriptedClient({"sleep": [first_version]}, last_sync=NOW)
    patch_client(client)
    sync_once(only=["sleep"])

    revised = sleep_payload(start, start + timedelta(hours=8, minutes=20),
                            external_id="sleep/a")
    client.points_by_type["sleep"] = [revised]
    sync_once(only=["sleep"])

    with db() as s:
        rows = s.scalars(select(SleepSession)).all()
        assert len(rows) == 1, "revision created a duplicate session"
        assert (rows[0].end_ts - rows[0].start_ts) == timedelta(hours=8, minutes=20)


# --- idempotence ------------------------------------------------------------


def test_repeated_polls_do_not_grow_the_database(patch_client, db):
    points = _hr_over(NOW - timedelta(hours=8), 6)
    client = ScriptedClient({"heart-rate": points}, last_sync=NOW)
    patch_client(client)
    sync_once(only=["heart-rate"])
    after_first = _raw_count(db, "heart-rate")
    for _ in range(4):
        sync_once(only=["heart-rate"])
    assert _raw_count(db, "heart-rate") == after_first


def test_clock_skew_cannot_push_the_watermark_into_the_future(patch_client, db):
    client = ScriptedClient({"heart-rate": _hr_over(NOW - timedelta(hours=4), 3)},
                            last_sync=NOW + timedelta(days=2))
    patch_client(client)
    sync_once(only=["heart-rate"])
    assert _watermark(db, "heart-rate") <= datetime.now(UTC)

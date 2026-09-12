"""The collector.

Design decisions worth stating:

* **Poll, don't webhook.** Google Health webhooks are project-level, need a
  service account, a public HTTPS endpoint and ECDSA P-256 signature
  verification. For one user that is a lot of surface area to maintain for a
  15-minute latency improvement.

* **Gate on `lastSyncTime`.** Data only exists in the API after the watch syncs
  to the phone. Polling faster than the watch syncs just burns quota, so we
  check `users.pairedDevices` first and skip high-frequency fetches when
  nothing new has arrived.

* **Intermittent uptime is safe.** Every type carries a high-water mark, and
  each poll re-requests an overlap window on top of it. If the collector is
  down for three days it backfills three days on restart. Nothing is lost;
  only calendar freshness degrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import DeviceSync, SyncState
from circa.db.session import session_scope
from circa.ingest.datatypes import HIGH_FREQUENCY, DataType, ordered
from circa.ingest.health_api import HealthApiError, HealthClient
from circa.ingest.parsing import first, parse_ts
from circa.ingest.store import persist_raw
from circa.normalize.dispatch import normalize_type

log = structlog.get_logger(__name__)

# How far back to re-request on top of the watermark. Google revises records
# after the fact, so this is not optional — a sleep session can change hours
# after it was first written.
_OVERLAP = {
    "sleep": timedelta(days=3),
    "exercise": timedelta(days=3),
    "_daily": timedelta(days=3),
    "_default": timedelta(hours=2),
}

# Cap the first-ever fetch. High-frequency types would otherwise pull ~8,700
# points/day; deep history is better retrieved via Google Takeout.
# How far behind the wall clock to hold the watermark when the device's
# lastSyncTime cannot be read. Generous on purpose: re-reading is cheap,
# skipping data is permanent.
_NO_DEVICE_INFO_MARGIN = timedelta(hours=6)

_INITIAL_CAP_DAYS = {
    "heart-rate": 30,
    "heart-rate-variability": 30,
    "oxygen-saturation": 14,
    "steps": 60,
}


@dataclass
class TypeReport:
    data_type: str
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    normalized: int = 0
    windows: int = 0
    server_filtered: bool = True
    error: str | None = None


@dataclass
class SyncReport:
    started_at: datetime
    finished_at: datetime | None = None
    device_synced_at: datetime | None = None
    skipped_high_frequency: bool = False
    types: list[TypeReport] = field(default_factory=list)

    @property
    def total_inserted(self) -> int:
        return sum(t.inserted for t in self.types)

    @property
    def errors(self) -> list[str]:
        return [f"{t.data_type}: {t.error}" for t in self.types if t.error]


def _overlap_for(dt: DataType) -> timedelta:
    if dt.name in _OVERLAP:
        return _OVERLAP[dt.name]
    if dt.name.startswith("daily-"):
        return _OVERLAP["_daily"]
    return _OVERLAP["_default"]


def _get_state(session: Session, data_type: str) -> SyncState:
    state = session.get(SyncState, data_type)
    if state is None:
        state = SyncState(data_type=data_type)
        session.add(state)
        session.flush()
    return state


def record_device_sync(session: Session, devices: list[dict]) -> datetime | None:
    """Store a pairedDevices snapshot; return the most recent lastSyncTime."""
    latest: datetime | None = None
    now = datetime.now(UTC)

    for device in devices:
        last_sync = parse_ts(first(device, "lastSyncTime", "lastSyncedTime", "syncTime"))
        device_id = str(first(device, "id", "name", "deviceId", "serialNumber") or "unknown")
        existing = session.scalar(
            select(DeviceSync).where(
                DeviceSync.device_id == device_id, DeviceSync.last_sync_time == last_sync
            )
        )
        if existing is None:
            battery = first(device, "batteryLevel", "battery", "batteryPercent")
            session.add(
                DeviceSync(
                    device_id=device_id,
                    device_name=str(first(device, "deviceName", "displayName", "type") or ""),
                    last_sync_time=last_sync,
                    battery_percent=int(battery) if isinstance(battery, (int, float)) else None,
                    observed_at=now,
                )
            )
        if last_sync and (latest is None or last_sync > latest):
            latest = last_sync
    return latest


def sync_once(
    force: bool = False,
    only: list[str] | None = None,
    since: datetime | None = None,
) -> SyncReport:
    """Run one collection pass.

    `force` bypasses the lastSyncTime gate; `since` overrides the watermark for
    a manual backfill.
    """
    report = SyncReport(started_at=datetime.now(UTC))
    now = datetime.now(UTC)

    with HealthClient() as client:
        # --- freshness gate ------------------------------------------------
        device_synced_at: datetime | None = None
        try:
            devices = client.paired_devices()
            with session_scope() as s:
                device_synced_at = record_device_sync(s, devices)
            report.device_synced_at = device_synced_at
        except HealthApiError as exc:
            log.warning("poller.device_check_failed", error=str(exc))

        skip_high_frequency = False
        if not force and device_synced_at is not None:
            with session_scope() as s:
                hr_state = s.get(SyncState, "heart-rate")
                last_success = hr_state.last_success_at if hr_state else None
            if last_success and device_synced_at <= last_success:
                skip_high_frequency = True
                log.info(
                    "poller.no_new_device_sync",
                    last_device_sync=device_synced_at.isoformat(),
                )
        report.skipped_high_frequency = skip_high_frequency

        # --- per-type collection -------------------------------------------
        for dt in ordered():
            if only and dt.name not in only:
                continue
            if skip_high_frequency and dt.name in HIGH_FREQUENCY:
                continue
            report.types.append(_sync_type(client, dt, now, since, device_synced_at))

    report.finished_at = datetime.now(UTC)
    log.info(
        "poller.done",
        inserted=report.total_inserted,
        errors=len(report.errors),
        skipped_hf=report.skipped_high_frequency,
    )
    return report


def _sync_type(
    client: HealthClient,
    dt: DataType,
    now: datetime,
    since_override: datetime | None,
    device_synced_at: datetime | None = None,
) -> TypeReport:
    settings = get_settings()
    report = TypeReport(data_type=dt.name)

    with session_scope() as s:
        state = _get_state(s, dt.name)
        watermark = state.watermark
        state.last_run_at = now

    if since_override is not None:
        start = since_override
    elif watermark is not None:
        start = watermark - _overlap_for(dt)
    else:
        days = _INITIAL_CAP_DAYS.get(dt.name, settings.initial_backfill_days)
        start = now - timedelta(days=days)
        log.info("poller.initial_backfill", data_type=dt.name, days=days)

    max_window = timedelta(days=settings.max_fetch_window_days)
    cursor = start
    highest_seen = watermark

    try:
        while cursor < now:
            window_end = min(cursor + max_window, now)
            result = client.fetch(dt, cursor, window_end)
            report.windows += 1
            report.fetched += len(result.points)
            report.server_filtered = result.server_filtered

            if result.points:
                with session_scope() as s:
                    inserted, dupes = persist_raw(s, dt, result.points)
                    report.inserted += inserted
                    report.duplicates += dupes

                    # Re-normalise the whole window, not just newly-inserted
                    # rows. Normalisation is idempotent, and gating it on
                    # `inserted > 0` meant that raw data which arrived but
                    # failed to normalise could never be recovered - the next
                    # run would see only duplicates and skip it forever.
                    from circa.ingest.store import load_raw

                    fresh = load_raw(s, dt.name, since=cursor - _overlap_for(dt))
                    report.normalized += normalize_type(s, dt.name, fresh)

            if not result.server_filtered:
                # The server ignored the time filter and returned everything it
                # holds, so this single pass has already covered the whole
                # range. Walking further windows would re-read the same pages -
                # but stopping without recording that let the watermark creep
                # forward just one window per poll. `exercise` is the only
                # NO_FILTER type, and it sat 27 days behind every other type in
                # production because of exactly this.
                highest_seen = now
                break

            highest_seen = window_end
            cursor = window_end

        with session_scope() as s:
            state = _get_state(s, dt.name)
            state.watermark = _next_watermark(highest_seen or now, device_synced_at, now)
            state.last_success_at = datetime.now(UTC)
            state.last_error = None
            state.consecutive_failures = 0
            state.points_ingested = (state.points_ingested or 0) + report.inserted

    except Exception as exc:  # noqa: BLE001 - one bad type must not stop the rest
        report.error = str(exc)
        log.error("poller.type_failed", data_type=dt.name, error=str(exc))
        with session_scope() as s:
            state = _get_state(s, dt.name)
            state.last_error = str(exc)[:1000]
            state.consecutive_failures = (state.consecutive_failures or 0) + 1

    return report


def _next_watermark(
    scanned_to: datetime,
    device_synced_at: datetime | None,
    now: datetime,
) -> datetime:
    """Where the next poll should resume from.

    The subtle part: the watermark must track *how far data could possibly
    exist*, not how far we happened to ask. Nothing reaches the API until the
    watch syncs to the phone, so advancing past `lastSyncTime` would skip over a
    window that is still empty only because the sync had not happened yet — and
    that data would then be missed permanently.

    Clamping here is what makes intermittent uptime safe. A collector that is
    off for three days resumes from the last instant it could have seen data,
    and backfills the gap, rather than jumping to the present.
    """
    ceiling = scanned_to
    if device_synced_at is not None:
        ceiling = min(ceiling, device_synced_at)
    else:
        # No lastSyncTime available (scope missing, or the call failed). Assume
        # nothing about how fresh the data is and hold the watermark well back,
        # so the next run re-reads rather than stepping over a window the watch
        # had not yet uploaded.
        ceiling = min(ceiling, now - _NO_DEVICE_INFO_MARGIN)
    # Never let a clock skew push the watermark into the future.
    ceiling = min(ceiling, now)
    # Small margin so a record written moments before the sync is not stranded.
    return ceiling - timedelta(minutes=5)

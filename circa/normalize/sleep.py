"""Sleep session normalisation.

Produces the timings the phase model actually consumes: onset, offset,
midpoint, latency, WASO. Stage *proportions* are recorded but deliberately
given little weight — Google infers Fitbit stages from movement, heart rate and
HRV rather than EEG, so they inform the homeostatic/recovery model but must not
be allowed to move the phase estimate much.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import RawDataPoint, SleepSession, SleepStage
from circa.ingest.parsing import as_float, extract_interval, first, parse_ts

log = structlog.get_logger(__name__)

ASLEEP_STAGES = {"LIGHT", "DEEP", "REM", "ASLEEP", "RESTLESS"}
AWAKE_STAGES = {"AWAKE", "WAKE"}

# Sessions shorter than this are treated as naps, not the main sleep episode.
MAIN_SLEEP_MIN_MINUTES = 180


def _stage_name(raw: dict) -> str:
    value = first(raw, "type", "stage", "level", "sleepStageType") or "UNKNOWN"
    return str(value).upper().replace("SLEEP_STAGE_TYPE_", "").replace("SLEEP_", "")


def _extract_stages(payload: dict, field: str = "sleep") -> list[dict]:
    stages = first(payload, f"{field}.stages", "stages", f"{field}.levels.data", "levels.data")
    return stages if isinstance(stages, list) else []


def normalize_sleep(session: Session, raw_points: list[RawDataPoint]) -> int:
    """Upsert SleepSession rows from raw payloads. Returns rows written."""
    settings = get_settings()
    default_tz = ZoneInfo(settings.timezone)
    written = 0

    for raw in raw_points:
        payload = raw.payload or {}
        interval = first(payload, "sleep.interval", "interval") or {}
        start, end, offset = extract_interval(interval)
        if start is None:
            start = parse_ts(first(payload, "sleep.startTime", "startTime"))
        if end is None:
            end = parse_ts(first(payload, "sleep.endTime", "endTime"))
        if start is None or end is None or end <= start:
            log.warning("sleep.unparseable", raw_id=raw.id)
            continue

        external_id = str(
            first(payload, "name", "sleep.name", "id", "sleep.logId") or f"raw:{raw.id}"
        )

        tz_name = settings.timezone
        if offset is None:
            offset = int(end.astimezone(default_tz).utcoffset().total_seconds())

        stages_raw = _extract_stages(payload)
        parsed_stages: list[tuple[datetime, datetime, str]] = []
        for entry in stages_raw:
            s_start, s_end, _ = extract_interval(entry.get("interval", entry))
            if s_start is None:
                s_start = parse_ts(first(entry, "startTime", "dateTime", "start"))
            if s_end is None and s_start is not None:
                # as_float, not float(): one malformed stage duration used to
                # raise out of here and lose the entire night with it.
                seconds = as_float(first(entry, "duration.seconds", "seconds", "durationSeconds"))
                if seconds is not None and 0 < seconds <= 86400:
                    s_end = s_start + timedelta(seconds=seconds)
            if s_start is None or s_end is None or s_end <= s_start:
                continue
            parsed_stages.append((s_start, s_end, _stage_name(entry)))

        parsed_stages.sort(key=lambda x: x[0])
        metrics = _derive_metrics(start, end, parsed_stages)

        # Attribute the night to the local date of *wake*, so "last night's
        # sleep" belongs to the morning it ended.
        local_end = end.astimezone(ZoneInfo(tz_name))
        sleep_date = local_end.date()

        row = session.scalar(
            select(SleepSession).where(SleepSession.external_id == external_id)
        )
        if row is None:
            row = SleepSession(external_id=external_id)
            session.add(row)

        row.start_ts = start
        row.end_ts = end
        row.tz_name = tz_name
        row.utc_offset_seconds = offset
        row.sleep_date = sleep_date
        row.session_type = str(first(payload, "sleep.type", "type") or "").upper() or None
        row.is_main_sleep = metrics["time_in_bed_minutes"] >= MAIN_SLEEP_MIN_MINUTES
        for key, value in metrics.items():
            setattr(row, key, value)

        session.flush()
        # Stages are fully replaced: Google may revise a session after the fact.
        for existing in list(row.stages):
            session.delete(existing)
        session.flush()
        for s_start, s_end, name in parsed_stages:
            session.add(
                SleepStage(session_id=row.id, start_ts=s_start, end_ts=s_end, stage=name)
            )
        written += 1

    return written


def _derive_metrics(
    start: datetime, end: datetime, stages: list[tuple[datetime, datetime, str]]
) -> dict:
    """Compute the sleep timings used downstream.

    With no stage detail we fall back to treating the whole session as sleep,
    which is what a CLASSIC-type record means anyway.
    """
    tib = (end - start).total_seconds() / 60.0
    per_stage = {"LIGHT": 0.0, "DEEP": 0.0, "REM": 0.0, "AWAKE": 0.0, "ASLEEP": 0.0}

    if not stages:
        onset, offset_ts = start, end
        tst = tib
        waso = 0.0
        latency = 0.0
    else:
        for s_start, s_end, name in stages:
            minutes = (s_end - s_start).total_seconds() / 60.0
            key = "AWAKE" if name in AWAKE_STAGES else name
            per_stage[key] = per_stage.get(key, 0.0) + minutes

        asleep = [s for s in stages if s[2] in ASLEEP_STAGES]
        if not asleep:
            return {
                "tst_minutes": 0.0,
                "time_in_bed_minutes": tib,
                "waso_minutes": tib,
                "latency_minutes": tib,
                "efficiency": 0.0,
                "midpoint_ts": start + (end - start) / 2,
                "minutes_light": per_stage.get("LIGHT"),
                "minutes_deep": per_stage.get("DEEP"),
                "minutes_rem": per_stage.get("REM"),
                "minutes_awake": per_stage.get("AWAKE"),
            }

        onset = asleep[0][0]
        offset_ts = asleep[-1][1]
        tst = sum((s[1] - s[0]).total_seconds() / 60.0 for s in asleep)
        latency = (onset - start).total_seconds() / 60.0
        # WASO counts only wake *between* first and last sleep, not the tail.
        waso = sum(
            (s[1] - s[0]).total_seconds() / 60.0
            for s in stages
            if s[2] in AWAKE_STAGES and s[0] >= onset and s[1] <= offset_ts
        )

    return {
        "tst_minutes": tst,
        "time_in_bed_minutes": tib,
        "waso_minutes": waso,
        "latency_minutes": max(latency, 0.0),
        "efficiency": (tst / tib) if tib > 0 else None,
        # Midpoint of the *sleep* episode (onset..offset), not of time in bed.
        "midpoint_ts": onset + (offset_ts - onset) / 2,
        "minutes_light": per_stage.get("LIGHT"),
        "minutes_deep": per_stage.get("DEEP"),
        "minutes_rem": per_stage.get("REM"),
        "minutes_awake": per_stage.get("AWAKE"),
    }


def sleep_intervals(session: Session, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Sleep spans overlapping [start, end) — used to mask HR de-masking regressors."""
    rows = session.scalars(
        select(SleepSession).where(
            SleepSession.end_ts > start,
            SleepSession.start_ts < end,
            SleepSession.excluded.is_(False),
        )
    )
    return [(r.start_ts, r.end_ts) for r in rows]


def utc_now() -> datetime:
    return datetime.now(UTC)

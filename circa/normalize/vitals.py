"""HRV, SpO2, exercise, and the daily-summary data types.

Note the asymmetry Google's API forces on us:

  * ``heart-rate-variability`` is a real RMSSD **time series**, so it can in
    principle carry phase information (an ablation candidate).
  * ``daily-sleep-temperature-derivations`` is **one number per night** against
    a 30-day baseline. The high-frequency overnight samples that produced it are
    not exposed, so skin temperature can never be a phase channel here — it is
    stored purely as illness/anomaly QC.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from circa.db.models import DailyMetric, ExerciseSession, HrvSample, RawDataPoint
from circa.ingest.parsing import as_float, extract_interval, first, parse_date, parse_ts

log = structlog.get_logger(__name__)


def normalize_hrv(session: Session, raw_points: list[RawDataPoint]) -> int:
    rows: dict[datetime, dict] = {}
    for raw in raw_points:
        payload = raw.payload or {}
        ts = parse_ts(
            first(payload, "heartRateVariability.sampleTime", "sampleTime")
        )
        if ts is None:
            continue
        rmssd = as_float(
            first(
                payload,
                "heartRateVariability.rootMeanSquareOfSuccessiveDifferencesMilliseconds",
                "rootMeanSquareOfSuccessiveDifferencesMilliseconds",
                "rmssd",
            )
        )
        sdnn = as_float(
            first(
                payload,
                "heartRateVariability.standardDeviationMilliseconds",
                "standardDeviationMilliseconds",
                "sdnn",
            )
        )
        if rmssd is None and sdnn is None:
            continue
        rows[ts] = {"ts": ts, "rmssd_ms": rmssd, "sdnn_ms": sdnn}

    if not rows:
        return 0
    for chunk in _chunked(list(rows.values()), 500):
        session.execute(
            sqlite_insert(HrvSample).values(chunk).on_conflict_do_nothing(index_elements=["ts"])
        )
    session.flush()
    return len(rows)


def normalize_exercise(session: Session, raw_points: list[RawDataPoint]) -> int:
    written = 0
    for raw in raw_points:
        payload = raw.payload or {}
        interval = first(payload, "exercise.interval", "interval") or {}
        start, end, _ = extract_interval(interval)
        if start is None or end is None or end <= start:
            continue
        external_id = str(first(payload, "name", "exercise.name", "id") or f"raw:{raw.id}")

        row = session.scalar(
            select(ExerciseSession).where(ExerciseSession.external_id == external_id)
        )
        if row is None:
            row = ExerciseSession(external_id=external_id)
            session.add(row)
        row.start_ts = start
        row.end_ts = end
        activity = first(payload, "exercise.activityType", "activityType", "exercise.type", "type")
        row.activity_type = str(activity) if activity else None
        row.calories = as_float(first(payload, "exercise.calories", "calories"))
        row.avg_hr = as_float(
            first(payload, "exercise.averageHeartRate", "averageHeartRate", "exercise.avgHeartRate")
        )
        steps = as_float(first(payload, "exercise.steps", "steps"))
        row.steps = int(steps) if steps is not None else None
        written += 1
    return written


# Which numeric field to pull out of each daily type, in priority order.
_DAILY_FIELDS: dict[str, tuple[str, ...]] = {
    "daily-resting-heart-rate": ("beatsPerMinute", "restingHeartRate"),
    "daily-heart-rate-variability": (
        "averageHeartRateVariabilityMilliseconds",
        "dailyRmssd",
    ),
    "daily-respiratory-rate": ("breathsPerMinute", "respiratoryRate", "averageBreathsPerMinute"),
    "daily-sleep-temperature-derivations": ("nightlyTemperatureCelsius",),
    "daily-oxygen-saturation": ("averagePercentage", "percentage"),
}


def normalize_daily(session: Session, data_type: str, raw_points: list[RawDataPoint]) -> int:
    """Fold any `daily-*` type into the generic DailyMetric table."""
    camel = _to_camel(data_type)
    candidates = _DAILY_FIELDS.get(data_type, ())
    rows: dict[tuple, dict] = {}

    for raw in raw_points:
        payload = raw.payload or {}
        body = payload.get(camel, payload)
        metric_date = parse_date(first(payload, f"{camel}.date", "date", f"{camel}.day"))
        if metric_date is None:
            ts = parse_ts(first(payload, f"{camel}.interval.startTime", "interval.startTime"))
            metric_date = ts.date() if ts else None
        if metric_date is None:
            continue

        value = None
        for name in candidates:
            value = as_float(first(payload, f"{camel}.{name}", name))
            if value is not None:
                break
        if value is None and isinstance(body, dict):
            # The named field is missing - Google renamed it, or this is a type
            # we have no mapping for. Falling back to "first plausible scalar in
            # the body" is not safe: a heart-rate-variability payload whose
            # named field vanished would hand back `nonRemHeartRateBeatsPerMinute`,
            # and a resting heart rate would be stored as an HRV in
            # milliseconds. A number that is wrong is worse than one that is
            # missing, because nothing downstream can tell.
            #
            # A single unambiguous scalar is still taken, which keeps genuinely
            # new daily types working. Anything more is left as None - `extra`
            # preserves the whole payload, so the day can be re-normalised once
            # the field name is known.
            scalars = {
                key: as_float(candidate)
                for key, candidate in body.items()
                if key not in {"date", "day"} and as_float(candidate) is not None
            }
            if len(scalars) == 1:
                value = next(iter(scalars.values()))
            elif scalars:
                log.warning(
                    "vitals.ambiguous_daily_payload",
                    data_type=data_type,
                    expected=candidates,
                    found=sorted(scalars),
                )

        extra = body if isinstance(body, dict) else None
        rows[(metric_date, data_type)] = {
            "metric_date": metric_date,
            "metric": data_type,
            "value": value,
            "extra": extra,
        }

    if not rows:
        return 0
    for chunk in _chunked(list(rows.values()), 200):
        session.execute(
            sqlite_insert(DailyMetric)
            .values(chunk)
            .on_conflict_do_update(
                index_elements=["metric_date", "metric"],
                set_={"value": sqlite_insert(DailyMetric).excluded.value,
                      "extra": sqlite_insert(DailyMetric).excluded.extra},
            )
        )
    session.flush()
    return len(rows)


def _to_camel(kebab: str) -> str:
    head, *tail = kebab.split("-")
    return head + "".join(p.title() for p in tail)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def utc_now() -> datetime:
    return datetime.now(UTC)

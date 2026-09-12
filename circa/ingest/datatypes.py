"""Registry of the Google Health API data types Circa ingests.

Two conventions from Google's docs that are easy to trip over:

  * the data type is **kebab-case in the URL path** (`.../dataTypes/body-fat/...`)
  * but **snake_case inside a filter expression** (`body_fat.interval...`)

Filter members below were **verified against the live API**, not taken from
documentation - Google's published examples turned out to be wrong for most
types. What `circa probe` established:

  * sample types    `<type>.sample_time.physical_time`   (UTC instant)
  * interval types  `<type>.interval.start_time`
  * sleep           `sleep.interval.END_time` - start_time is explicitly
                    rejected, which is a per-type quirk rather than a shape
                    rule. Filtering on wake time also matches how a night is
                    attributed to its wake date.
  * daily types     `<type>.date`
  * exercise        no time filter supported at all
  * total-calories  `list` is not supported at all; only rollup actions are

Anything without a filter falls back to an unfiltered page-and-trim fetch,
which is fine at these volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    SAMPLE = "sample"  # point-in-time reading
    INTERVAL = "interval"  # bounded bucket (steps, calories)
    SESSION = "session"  # long episode (sleep, exercise)
    DAILY = "daily"  # one value per calendar day


@dataclass(frozen=True, slots=True)
class DataType:
    """One data type, under all three of the names Google gives it.

    These are genuinely different strings and must never be derived from one
    another on the fly - conflating them is the single easiest way to silently
    read nothing out of a payload:

      name         kebab-case   URL path        .../dataTypes/heart-rate/...
      filter_field snake_case   filter syntax   heart_rate.sample_time >= "..."
      payload_key  camelCase    JSON body       {"heartRate": {...}}
    """

    name: str
    filter_field: str
    payload_key: str
    kind: Kind
    time_filter: str | None
    # How the filter's right-hand side must be formatted: an RFC3339 UTC
    # instant, or a bare calendar date.
    filter_format: str  # "utc" | "date"
    priority: int  # lower runs first; sleep before HR so context exists
    note: str = ""

    @property
    def path(self) -> str:
        return f"users/me/dataTypes/{self.name}/dataPoints"


def _to_camel(kebab: str) -> str:
    head, *tail = kebab.split("-")
    return head + "".join(part.title() for part in tail)


# Sentinel: this type supports no server-side time filter.
NO_FILTER = object()


def _dt(name, kind, priority, time_filter=None, note="") -> DataType:
    filter_field = name.replace("-", "_")
    if time_filter is NO_FILTER:
        resolved = None
    elif time_filter is None:
        suffix = {
            Kind.SAMPLE: "sample_time.physical_time",
            Kind.DAILY: "date",
        }.get(kind, "interval.start_time")
        resolved = f"{filter_field}.{suffix}"
    else:
        resolved = f"{filter_field}.{time_filter}"
    return DataType(
        name=name,
        filter_field=filter_field,
        payload_key=_to_camel(name),
        kind=kind,
        time_filter=resolved,
        filter_format="date" if kind is Kind.DAILY else "utc",
        priority=priority,
        note=note,
    )


# Ordered by modelling value. Sleep first: it anchors everything else, and the
# HR de-masking regression needs to know which samples were during sleep.
REGISTRY: list[DataType] = [
    _dt("sleep", Kind.SESSION, 10, time_filter="interval.end_time",
        note="Sessions + stages. Primary phase observation channel. "
             "Filters on END time - start_time is rejected by the API."),
    _dt("steps", Kind.INTERVAL, 20,
        note="Variable-length bouts. Doubles as the light proxy input."),
    _dt("heart-rate", Kind.SAMPLE, 30,
        note="~5s resolution, ~8.7k points/day. Primary physiological channel."),
    _dt("heart-rate-variability", Kind.SAMPLE, 40,
        note="RMSSD time series. Ablation candidate - cut if it does not earn its place."),
    _dt("exercise", Kind.SESSION, 50, time_filter=NO_FILTER,
        note="Nonphotic zeitgeber + HR de-masking covariate. No time filter "
             "is supported, so this is fetched unfiltered."),
    _dt("daily-resting-heart-rate", Kind.DAILY, 60),
    _dt("daily-heart-rate-variability", Kind.DAILY, 61),
    _dt("daily-respiratory-rate", Kind.DAILY, 62),
    _dt("daily-sleep-temperature-derivations", Kind.DAILY, 63,
        note="Nightly mean vs 30d baseline. Daily-only, so QC/illness use ONLY - "
             "it can never be a phase channel."),
    _dt("oxygen-saturation", Kind.SAMPLE, 70, note="QC only - not a phase marker."),
    _dt("daily-oxygen-saturation", Kind.DAILY, 71),
    _dt("sedentary-period", Kind.INTERVAL, 80, note="Behavioural probe input."),
    _dt("active-zone-minutes", Kind.INTERVAL, 81),
    # total-calories deliberately omitted: the API rejects `list` for it
    # entirely ("only rollup, dailyRollup are supported"), and it carries no
    # circadian information anyway.
]

BY_NAME: dict[str, DataType] = {d.name: d for d in REGISTRY}

# Types worth pulling on every poll vs. once a day.
HIGH_FREQUENCY = {"heart-rate", "steps", "heart-rate-variability", "oxygen-saturation"}


def ordered() -> list[DataType]:
    return sorted(REGISTRY, key=lambda d: d.priority)

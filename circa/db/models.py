"""Database schema.

Three layers, per the plan:

  raw/         exact API responses, immutable, retention-limited
  normalized/  numeric time series at usable resolution, kept forever
  derived/     phase posteriors, alertness curves, calendar block state

The retention split is what keeps this free-tier sized: 5-second heart rate is
~8,700 samples/day (~3.2M rows/year), but the phase model only ever consumes
5-minute bins. So raw samples age out after 90 days while 1-minute medians are
kept forever at ~15 MB/year.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from circa.db.types import UTCDateTime


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# auth + sync bookkeeping
# ---------------------------------------------------------------------------


class OAuthToken(Base):
    """Google OAuth credentials, one row per API family.

    Two tokens, not one. The Google Health API rejects any access token that
    also carries Calendar scopes:

        403 DISALLOWED_OAUTH_SCOPES
        disallowed_scopes: cl_app_created,cl_readonly

    So Health and Calendar are authorised separately and their tokens kept
    apart. Requesting them together, or using incremental authorisation
    (`include_granted_scopes=true`), merges the scope sets onto one token and
    breaks every Health request.

    The refresh token is Fernet-encrypted at rest; the access token is not,
    since it is short-lived and useless once expired.
    """

    __tablename__ = "oauth_token"

    purpose: Mapped[str] = mapped_column(String(16), primary_key=True)
    access_token: Mapped[str | None] = mapped_column(Text)
    refresh_token_encrypted: Mapped[bytes | None] = mapped_column()
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    scopes: Mapped[str] = mapped_column(Text, default="")
    issued_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class SyncState(Base):
    """Per-data-type high-water mark, so restarts resume with no gap or dupes."""

    __tablename__ = "sync_state"

    data_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Everything up to this instant has been successfully fetched and stored.
    watermark: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_run_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    points_ingested: Mapped[int] = mapped_column(Integer, default=0)


class DeviceSync(Base):
    """Snapshot of `users.pairedDevices` — drives the poll gate."""

    __tablename__ = "device_sync"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String(128), index=True)
    device_name: Mapped[str | None] = mapped_column(String(128))
    last_sync_time: Mapped[datetime | None] = mapped_column(UTCDateTime)
    battery_percent: Mapped[int | None] = mapped_column(Integer)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)

    __table_args__ = (UniqueConstraint("device_id", "last_sync_time", name="uq_device_sync"),)


# ---------------------------------------------------------------------------
# raw layer
# ---------------------------------------------------------------------------


class RawDataPoint(Base):
    """Verbatim API payload. Never mutated; aged out by the retention job.

    `content_hash` deduplicates re-fetches of the same point, which happen
    routinely because we re-request an overlap window on every poll to catch
    late-arriving and reconciled records.
    """

    __tablename__ = "raw_datapoint"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data_type: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16), default="api")  # api | takeout
    point_time: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime)
    content_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)

    __table_args__ = (
        UniqueConstraint("data_type", "content_hash", name="uq_raw_dedupe"),
        Index("ix_raw_type_time", "data_type", "point_time"),
    )


# ---------------------------------------------------------------------------
# normalized layer
# ---------------------------------------------------------------------------


class HeartRateSample(Base):
    """~5-second resolution samples. Aged out after `hr_sample_retention_days`."""

    __tablename__ = "hr_sample"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    bpm: Mapped[int] = mapped_column(Integer)
    motion_context: Mapped[str | None] = mapped_column(String(24))


class HeartRateMinute(Base):
    """1-minute robust aggregates. Kept forever — this is what modelling uses."""

    __tablename__ = "hr_minute"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    bpm_median: Mapped[float] = mapped_column(Float)
    bpm_min: Mapped[int] = mapped_column(Integer)
    bpm_max: Mapped[int] = mapped_column(Integer)
    n_samples: Mapped[int] = mapped_column(Integer)
    active_fraction: Mapped[float | None] = mapped_column(Float)


class StepInterval(Base):
    """Raw step bouts as returned by the API (variable length, not minute-uniform)."""

    __tablename__ = "step_interval"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_ts: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    end_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    count: Mapped[int] = mapped_column(Integer)

    __table_args__ = (UniqueConstraint("start_ts", "end_ts", name="uq_step_interval"),)


class StepMinute(Base):
    """Steps redistributed onto a uniform 1-minute grid."""

    __tablename__ = "step_minute"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    steps: Mapped[float] = mapped_column(Float)


class HrvSample(Base):
    """Intraday HRV time series. Sampling interval is undocumented — measure it."""

    __tablename__ = "hrv_sample"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    rmssd_ms: Mapped[float | None] = mapped_column(Float)
    sdnn_ms: Mapped[float | None] = mapped_column(Float)


class SleepSession(Base):
    """One sleep episode plus the derived timings the phase model consumes."""

    __tablename__ = "sleep_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    start_ts: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    end_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    # Local wall-clock context, stored explicitly rather than re-derived.
    tz_name: Mapped[str | None] = mapped_column(String(64))
    utc_offset_seconds: Mapped[int | None] = mapped_column(Integer)
    # The local calendar date this sleep is attributed to (the morning of wake).
    sleep_date: Mapped[date | None] = mapped_column(Date, index=True)

    session_type: Mapped[str | None] = mapped_column(String(16))  # CLASSIC | STAGES
    is_main_sleep: Mapped[bool] = mapped_column(Boolean, default=True)

    # derived metrics
    tst_minutes: Mapped[float | None] = mapped_column(Float)
    time_in_bed_minutes: Mapped[float | None] = mapped_column(Float)
    waso_minutes: Mapped[float | None] = mapped_column(Float)
    latency_minutes: Mapped[float | None] = mapped_column(Float)
    efficiency: Mapped[float | None] = mapped_column(Float)
    midpoint_ts: Mapped[datetime | None] = mapped_column(UTCDateTime)
    minutes_light: Mapped[float | None] = mapped_column(Float)
    minutes_deep: Mapped[float | None] = mapped_column(Float)
    minutes_rem: Mapped[float | None] = mapped_column(Float)
    minutes_awake: Mapped[float | None] = mapped_column(Float)

    # Set by the phase layer: was this wake alarm-forced (calendar) or spontaneous?
    wake_forced: Mapped[bool | None] = mapped_column(Boolean)
    # User override from the web app ("ignore last night").
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)

    stages: Mapped[list[SleepStage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SleepStage(Base):
    __tablename__ = "sleep_stage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sleep_session.id", ondelete="CASCADE"))
    start_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    end_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    stage: Mapped[str] = mapped_column(String(16))

    session: Mapped[SleepSession] = relationship(back_populates="stages")


class ExerciseSession(Base):
    """Structured activity — a nonphotic zeitgeber and an HR de-masking covariate."""

    __tablename__ = "exercise_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    start_ts: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    end_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    activity_type: Mapped[str | None] = mapped_column(String(64))
    calories: Mapped[float | None] = mapped_column(Float)
    avg_hr: Mapped[float | None] = mapped_column(Float)
    steps: Mapped[int | None] = mapped_column(Integer)


class DailyMetric(Base):
    """Generic bucket for every `daily-*` data type.

    Skin temperature lands here rather than in its own time-series table: the
    API exposes only one `nightlyTemperatureCelsius` per night against a 30-day
    baseline, so it can never be a phase channel. It serves as illness/anomaly
    QC only.
    """

    __tablename__ = "daily_metric"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_date: Mapped[date] = mapped_column(Date, index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (UniqueConstraint("metric_date", "metric", name="uq_daily_metric"),)


class DataQualityDay(Base):
    """Per-day coverage and anomaly flags. Feeds observation-variance inflation.

    Abnormal nights inflate variance rather than being deleted, so the filter
    says "I am less certain today" instead of silently dropping evidence.
    """

    __tablename__ = "data_quality_day"

    metric_date: Mapped[date] = mapped_column(Date, primary_key=True)
    hr_coverage: Mapped[float | None] = mapped_column(Float)
    step_coverage: Mapped[float | None] = mapped_column(Float)
    has_sleep: Mapped[bool] = mapped_column(Boolean, default=False)
    nonwear_minutes: Mapped[float | None] = mapped_column(Float)
    illness_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    ood_flags: Mapped[dict | None] = mapped_column(JSON)


# ---------------------------------------------------------------------------
# derived layer
# ---------------------------------------------------------------------------


class PhaseObservation(Base):
    """One channel's phase estimate for one day, as a von Mises (mu, kappa)."""

    __tablename__ = "phase_observation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    obs_date: Mapped[date] = mapped_column(Date, index=True)
    channel: Mapped[str] = mapped_column(String(24))  # sleep | hr | hrv | probe:*
    mu_hours: Mapped[float] = mapped_column(Float)  # circadian phase, 0-24
    kappa: Mapped[float] = mapped_column(Float)  # von Mises concentration
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    model_version: Mapped[str] = mapped_column(String(32))
    detail: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        UniqueConstraint("obs_date", "channel", "model_version", name="uq_phase_obs"),
    )


class PhaseEstimate(Base):
    """Fused posterior over circadian phase for a given day."""

    __tablename__ = "phase_estimate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    model_version: Mapped[str] = mapped_column(String(32))

    dlmo_ts: Mapped[datetime | None] = mapped_column(UTCDateTime)
    dlmo_ci80_low: Mapped[datetime | None] = mapped_column(UTCDateTime)
    dlmo_ci80_high: Mapped[datetime | None] = mapped_column(UTCDateTime)
    cbtmin_ts: Mapped[datetime | None] = mapped_column(UTCDateTime)
    phase_sd_minutes: Mapped[float | None] = mapped_column(Float)

    confidence_tier: Mapped[int] = mapped_column(Integer, default=0)
    confidence_q: Mapped[float | None] = mapped_column(Float)
    n_nights: Mapped[int | None] = mapped_column(Integer)
    channel_agreement_minutes: Mapped[float | None] = mapped_column(Float)
    detail: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        UniqueConstraint("target_date", "model_version", name="uq_phase_estimate"),
    )


class CalendarLink(Base):
    """Maps a Circa category to a real Google Calendar."""

    __tablename__ = "calendar_link"

    category: Mapped[str] = mapped_column(String(24), primary_key=True)
    calendar_id: Mapped[str | None] = mapped_column(String(256))
    summary: Mapped[str] = mapped_column(String(128))
    color_id: Mapped[str | None] = mapped_column(String(8))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class CalendarBlock(Base):
    """A block Circa has written (or will write) to a calendar.

    `block_key` is the idempotency handle: sync patches the existing event with
    this key rather than delete/recreate, which would otherwise fire a
    notification storm and churn event IDs on every poll.
    """

    __tablename__ = "calendar_block"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    block_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(24), index=True)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(48))

    start_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    end_ts: Mapped[datetime] = mapped_column(UTCDateTime)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text)

    google_event_id: Mapped[str | None] = mapped_column(String(256))
    forecast_revision: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str] = mapped_column(String(32))
    synced_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    # Part of the event body, so it has to be part of the "has this changed?"
    # comparison. Without it, turning notifications on left every existing
    # event untouched and the setting silently never took effect.
    notify_minutes: Mapped[int | None] = mapped_column(Integer)
    # Also part of the event body, and so also part of "has this changed?".
    event_color_id: Mapped[str | None] = mapped_column(String(8))


class ForecastCache(Base):
    """The most recent computed forecast, ready to serve.

    The model costs about five seconds, almost all of it integrating the
    circadian oscillator. That is fine on a fifteen-minute schedule and
    unacceptable per HTTP request - but the web app used to call the whole
    pipeline on every page load *and* again for the chart, so opening Today
    cost nearly nine seconds of recomputation to arrive at exactly the numbers
    the scheduler had produced minutes earlier. The scheduler writes here; the
    web app only reads.
    """

    __tablename__ = "forecast_cache"

    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    model_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


class Setting(Base):
    """Runtime key/value settings editable from the web app."""

    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

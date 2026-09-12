"""Runtime settings — everything tunable, editable from the web app.

Deliberately separate from `circa.config.Settings`: that holds deployment
concerns (credentials, paths, ports) which live in the environment. This holds
*personal* parameters, which belong in the UI so the system can be adjusted
without touching code or restarting.

Stored as a single JSON row so adding a field never needs a migration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from circa.db.models import Setting

SETTINGS_KEY = "runtime"


class Chronotype(StrEnum):
    NIGHT_OWL = "night_owl"
    SLIGHT_OWL = "slight_owl"
    NEUTRAL = "neutral"
    MORNING = "morning"


class LowConfidencePolicy(StrEnum):
    WIDE_BLOCKS = "wide_blocks"      # write, but wider and labelled
    WRITE_NOTHING = "write_nothing"
    ROBUST_ONLY = "robust_only"      # sleep + light only


class NotificationPolicy(StrEnum):
    NONE = "none"
    ACTIONABLE = "actionable"
    ALL = "all"


# Population priors for the sleep-midpoint -> DLMO offset, by chronotype.
#
# DLMO precedes sleep midpoint by roughly 5-7 h in aligned sleepers, and the
# angle differs systematically with morningness/eveningness (Duffy et al.).
# These are *priors only* - the filter moves off them as personal data arrives,
# which is the whole point of the tier system.
CHRONOTYPE_PRIORS: dict[str, dict[str, float]] = {
    # dlmo_offset_hours: hours DLMO precedes sleep midpoint
    # tau_hours: prior mean intrinsic period (evening types run longer)
    Chronotype.NIGHT_OWL:  {"dlmo_offset_hours": 6.6, "tau_hours": 24.40, "prior_sd_hours": 1.2},
    Chronotype.SLIGHT_OWL: {"dlmo_offset_hours": 6.3, "tau_hours": 24.30, "prior_sd_hours": 1.1},
    Chronotype.NEUTRAL:    {"dlmo_offset_hours": 6.0, "tau_hours": 24.18, "prior_sd_hours": 1.0},
    Chronotype.MORNING:    {"dlmo_offset_hours": 5.6, "tau_hours": 24.05, "prior_sd_hours": 1.1},
}


class RuntimeSettings(BaseModel):
    """Personal parameters. Defaults match the answers given at setup."""

    # --- who you are ------------------------------------------------------
    chronotype: Chronotype = Chronotype.NIGHT_OWL
    target_sleep_hours: float = Field(default=8.0, ge=4.0, le=12.0)

    # --- schedule ---------------------------------------------------------
    # Used only as a fallback when the calendar has no signal; real commitments
    # come from the Google Calendar read scope.
    workday_start_local: str = "09:00"
    workday_end_local: str = "17:00"
    # A calendar event starting before this hour marks the wake as alarm-forced.
    forced_wake_before_hour: int = Field(default=10, ge=0, le=23)

    # --- history ----------------------------------------------------------
    # How far back to write what actually happened. Forecasts are never written
    # into the past; only recorded facts are, so the calendar reads as a record
    # behind you and a plan ahead of you.
    history_days: int = Field(default=7, ge=0, le=60)
    show_recorded_sleep: bool = True

    # Sleep debt: cumulative shortfall against the target. 14 days is the window
    # the consumer apps use, and roughly the horizon over which recovery sleep
    # still tracks the deficit.
    sleep_debt_window_days: int = Field(default=14, ge=3, le=60)
    # Debt is paid back gradually - trying to clear it in one night mostly
    # produces a bedtime nobody keeps.
    sleep_debt_payback_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    max_debt_payback_hours: float = Field(default=1.5, ge=0.0, le=4.0)

    # --- calendar behaviour ----------------------------------------------
    low_confidence_policy: LowConfidencePolicy = LowConfidencePolicy.WIDE_BLOCKS
    notifications: NotificationPolicy = NotificationPolicy.NONE
    notification_minutes_before: int = Field(default=10, ge=0, le=120)
    forecast_horizon_hours: int = Field(default=48, ge=6, le=168)

    # --- which blocks to generate ----------------------------------------
    enable_focus: bool = True
    enable_sleep_blocks: bool = True
    enable_light: bool = True
    enable_caffeine_cutoff: bool = True
    enable_workout_window: bool = True
    enable_last_meal: bool = True
    enable_wind_down: bool = True
    enable_debug_calendar: bool = False

    # --- block tuning -----------------------------------------------------
    # Alertness z-score thresholds; hysteresis prevents noise splitting a block.
    high_alertness_z: float = Field(default=0.7, ge=0.0, le=3.0)
    low_alertness_z: float = Field(default=-0.6, ge=-3.0, le=0.0)
    hysteresis_z: float = Field(default=0.15, ge=0.0, le=1.0)
    min_block_minutes: int = Field(default=45, ge=5, le=240)
    round_to_minutes: int = Field(default=15, ge=1, le=60)
    # A full day now generates about twelve blocks, so the old default of 12
    # sat exactly on the cap - one more and the evening was dropped.
    max_blocks_per_day: int = Field(default=18, ge=1, le=50)

    # --- physiology knobs -------------------------------------------------
    caffeine_half_life_hours: float = Field(default=5.0, ge=1.0, le=12.0)
    # Aim for caffeine to be below this fraction of the dose at sleep onset.
    caffeine_residual_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    last_meal_hours_before_sleep: float = Field(default=3.0, ge=0.0, le=12.0)
    wind_down_minutes: int = Field(default=60, ge=0, le=240)
    # Length of the morning bright-light block. The phase-advance region of the
    # light PRC is broad (several hours after CBTmin), but a shorter block is
    # more actionable - "get outside for this stretch" beats "sometime in these
    # five hours".
    morning_light_hours: float = Field(default=2.5, ge=0.25, le=8.0)
    # How long before predicted melatonin onset to start dimming.
    dim_light_lead_hours: float = Field(default=2.0, ge=0.0, le=8.0)

    # --- modelling --------------------------------------------------------
    n_particles: int = Field(default=1500, ge=200, le=8000)
    # Integration cost scales with (timesteps x light histories), NOT with
    # particle count - each history needs its own pass through the ODE, while
    # particles inside a history are vectorised for free. So particles are
    # cheap and light histories are expensive. 10 histories keeps a full
    # 21-day run near 3 s locally (~10 s on a shared-core e2-micro).
    n_light_samples: int = Field(default=10, ge=4, le=200)
    sleep_history_days: int = Field(default=42, ge=3, le=365)
    # 0 is a documented sentinel on the settings page: it switches the
    # heart-rate channel off entirely by leaving it an empty window.
    hr_window_hours: int = Field(default=72, ge=0, le=336)
    enable_hrv_channel: bool = False  # ablation candidate; off until it earns it

    # --- manual overrides -------------------------------------------------
    travel_mode: bool = False
    paused: bool = False

    @property
    def prior(self) -> dict[str, float]:
        return CHRONOTYPE_PRIORS[self.chronotype]


# Values that were only ever the default. If one of these is stored unchanged
# and the default later moves, the stored copy is stale rather than chosen, and
# keeping it would silently pin a user to a number they never picked.
_SUPERSEDED_DEFAULTS = {"max_blocks_per_day": 12}


def load_settings(session: Session) -> RuntimeSettings:
    row = session.get(Setting, SETTINGS_KEY)
    if row is None or not row.value:
        return RuntimeSettings()
    value = dict(row.value)
    for field, superseded in _SUPERSEDED_DEFAULTS.items():
        if value.get(field) == superseded:
            value.pop(field)
    try:
        return RuntimeSettings.model_validate(value)
    except Exception:  # noqa: BLE001 - a bad row must not brick the service
        return RuntimeSettings()


def save_settings(session: Session, settings: RuntimeSettings) -> None:
    row = session.get(Setting, SETTINGS_KEY)
    if row is None:
        row = Setting(key=SETTINGS_KEY)
        session.add(row)
    row.value = settings.model_dump(mode="json")
    row.updated_at = datetime.now(UTC)

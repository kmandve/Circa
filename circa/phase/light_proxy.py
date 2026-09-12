"""Probabilistic light exposure inferred from activity.

The Fitbit Air has no ambient light sensor, and light is the dominant zeitgeber.
The naive fix — assume 500 lux awake, 0 asleep — silently injects a fabricated
certainty into the most influential input the oscillator has.

Instead we treat light as a distribution and sample from it. Huang et al. 2021
found that scaled step counts fed to a circadian model matched, and under
misalignment *beat*, measured wrist light — wrist photometry is frequently
occluded by sleeves, so activity is not the poor substitute it sounds like.

Each sampled history is integrated separately; the spread of the resulting
phases becomes an explicit, reported component of uncertainty rather than a
hidden modelling assumption.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.config import get_settings
from circa.db.models import HeartRateMinute, StepMinute
from circa.normalize.sleep import sleep_intervals

log = structlog.get_logger(__name__)

BIN_MINUTES = 10

# Luminous efficacy of daylight: ~105-120 lm/W. Used to turn a clear-sky
# irradiance (W/m^2) into an illuminance ceiling (lux).
LUMENS_PER_WATT = 110.0

# Global horizontal irradiance is measured on an upward-facing surface. What
# reaches the eye is roughly vertical-plane illuminance, which is substantially
# lower - the sun is rarely straight ahead, and the sky hemisphere is half
# occluded by ground and buildings. Without this factor the proxy treats a
# person outdoors as receiving full horizontal-plane daylight at the cornea.
EYE_LEVEL_FACTOR = 0.35

# A day needs at least this many 10-minute bins carrying real device data
# before its light history is treated as reconstructed rather than assumed.
MIN_OBSERVED_BINS_PER_DAY = 36  # 6 hours

# How much wider the light distribution gets on bins nothing was recorded for.
UNOBSERVED_SD_INFLATION = 1.8

# Indoor illuminance is remarkably consistent across built environments:
# roughly 50-500 lux, log-normally distributed.
INDOOR_LOG_MEAN = np.log(150.0)
INDOOR_LOG_SD = 0.75

# Sleeping in a dark room. Not exactly zero - streetlight, dawn through
# curtains - and the oscillator is nonlinear at low intensities, so it matters
# that this is small rather than absent.
SLEEP_LUX_MEAN = 1.0
SLEEP_LUX_SD = 1.5

# Step rate (steps/min) at which being outdoors becomes as likely as not.
# Sustained walking pace is ~100 steps/min, so the midpoint sits where movement
# is clearly deliberate rather than incidental. The earlier calibration put the
# midpoint at 40 with a shallow slope, which made sitting perfectly still a ~10%
# chance of being outdoors in every single ten-minute bin.
OUTDOOR_STEPS_MIDPOINT = 60.0
OUTDOOR_STEPS_SLOPE = 0.06
# Sitting indoors, occasionally near a window.
BASE_OUTDOOR_PROB = 0.02
# Outdoors is rarely full clear-sky exposure: cloud, shade, buildings,
# orientation. Combined with EYE_LEVEL_FACTOR this puts a midday walk at
# roughly 10,000 lux at the eye, which matches field measurements far better
# than the ~38,000 the previous calibration implied.
OUTDOOR_ATTENUATION_MEAN = 0.35

# Sanity target for the calibration above. Published actigraphy/photometry work
# puts typical adult daily light exposure in the low thousands of lux-hours,
# with only an hour or two above 1000 lux. An earlier calibration produced
# ~68,000 lux-hours/day - roughly fifteen times too much - which over-advanced
# the oscillator and pulled the phase estimate over an hour early.
PLAUSIBLE_DAILY_LUX_HOURS = (1_000, 20_000)


@dataclass(slots=True)
class LightEnsemble:
    """M sampled light histories on a shared time grid."""

    times: np.ndarray        # datetimes, UTC
    hours: np.ndarray        # hours since grid start
    light: np.ndarray        # (M, n_bins) lux
    wake: np.ndarray         # (n_bins,) 1.0 awake, 0.0 asleep
    p_outdoor: np.ndarray    # (n_bins,) diagnostic
    clear_sky_lux: np.ndarray  # (n_bins,) ceiling
    observed: np.ndarray     # (n_bins,) True where real device data backs the bin


@functools.lru_cache(maxsize=8)
def _clearsky_series(lat: float, lon: float, tz: str, day_iso: str) -> tuple:
    """Clear-sky GHI for one local day, cached (it is identical year to year)."""
    import pandas as pd
    from pvlib.location import Location

    location = Location(latitude=lat, longitude=lon, tz=tz)
    times = pd.date_range(
        start=f"{day_iso} 00:00", end=f"{day_iso} 23:59",
        freq=f"{BIN_MINUTES}min", tz=tz,
    )
    clearsky = location.get_clearsky(times, model="ineichen")
    return tuple(times.tz_convert("UTC").to_pydatetime()), tuple(clearsky["ghi"].to_numpy())


def clear_sky_lux(grid: np.ndarray) -> np.ndarray:
    """Upper bound on natural illuminance at each grid point.

    This is the physical ceiling: no activity pattern can imply more light than
    the sun is actually delivering. It is what stops the proxy inventing
    daylight at 3am.
    """
    settings = get_settings()
    lookup: dict[datetime, float] = {}
    for day in sorted({t.astimezone(UTC).date() for t in grid}):
        for offset in (-1, 0, 1):  # cover timezone edges
            iso = (day + timedelta(days=offset)).isoformat()
            try:
                times, ghi = _clearsky_series(
                    settings.latitude, settings.longitude, settings.timezone, iso
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("light_proxy.clearsky_failed", day=iso, error=str(exc))
                continue
            for t, g in zip(times, ghi, strict=False):
                lookup[t.replace(tzinfo=UTC)] = float(g)

    if not lookup:
        # Fall back to a crude solar-elevation-free daylight window rather than
        # failing outright.
        # Local hours, not UTC: for a Chicago user, `t.hour` in UTC would put
        # "daylight" at 02:00-14:00 local and invent bright light in the middle
        # of the biological night - the exact failure this ceiling exists to
        # prevent.
        tz = ZoneInfo(settings.timezone)
        return np.where(
            np.array([7 <= t.astimezone(tz).hour < 19 for t in grid]),
            20000.0 * EYE_LEVEL_FACTOR, 0.0,
        )

    keys = np.array(sorted(lookup))
    values = np.array([lookup[k] for k in keys])
    idx = np.clip(np.searchsorted(keys, grid), 0, len(keys) - 1)
    return values[idx] * LUMENS_PER_WATT * EYE_LEVEL_FACTOR


def build_ensemble(
    session: Session,
    start: datetime,
    end: datetime,
    n_samples: int,
    rng: np.random.Generator | None = None,
    habitual_sleep: tuple[float, float] | None = None,
    utc_offset_seconds: int = 0,
) -> LightEnsemble:
    """Sample `n_samples` plausible light histories over [start, end).

    `habitual_sleep` is (onset_hour, wake_hour) in local time, used only to
    place darkness on days with no device data. Bins it covers are flagged
    `observed=False` on the result.
    """
    rng = rng or np.random.default_rng()
    n_bins = max(int((end - start).total_seconds() // (BIN_MINUTES * 60)), 1)
    grid = np.array([start + timedelta(minutes=BIN_MINUTES * i) for i in range(n_bins)])
    hours = np.arange(n_bins) * (BIN_MINUTES / 60.0)

    # --- observed behaviour ------------------------------------------------
    steps = np.zeros(n_bins)
    for ts, count in session.execute(
        select(StepMinute.ts, StepMinute.steps).where(StepMinute.ts >= start, StepMinute.ts < end)
    ).all():
        idx = int((ts - start).total_seconds() // (BIN_MINUTES * 60))
        if 0 <= idx < n_bins:
            steps[idx] += float(count)
    steps_per_min = steps / BIN_MINUTES

    asleep = np.zeros(n_bins, dtype=bool)
    for s_start, s_end in sleep_intervals(session, start, end):
        lo = max(0, int((s_start - start).total_seconds() // (BIN_MINUTES * 60)))
        hi = min(n_bins, int((s_end - start).total_seconds() // (BIN_MINUTES * 60)) + 1)
        asleep[lo:hi] = True

    # --- what did the device actually see? ---------------------------------
    # Absence of data is not evidence of wakefulness. Without this, a day the
    # watch was not worn arrives as "awake, indoors, lights on, all night" -
    # roughly 90 lux straight through the biological night - and the oscillator
    # is driven by a schedule nobody lived. A 21-day filter window backfilled
    # from three days of history is eighteen fabricated nights of phase-delaying
    # light - which is the normal state of affairs in the first week.
    observed = np.zeros(n_bins, dtype=bool)
    observed |= steps > 0
    observed |= asleep
    for (ts,) in session.execute(
        select(HeartRateMinute.ts).where(HeartRateMinute.ts >= start, HeartRateMinute.ts < end)
    ).all():
        idx = int((ts - start).total_seconds() // (BIN_MINUTES * 60))
        if 0 <= idx < n_bins:
            observed[idx] = True

    # Coverage is judged per local day: a day with a handful of stray samples is
    # still a day we cannot reconstruct.
    day_index = np.array([int((t - start).total_seconds() // 86400) for t in grid])
    bins_per_day = max(int(86400 // (BIN_MINUTES * 60)), 1)
    for day in np.unique(day_index):
        mask = day_index == day
        if observed[mask].sum() < MIN_OBSERVED_BINS_PER_DAY * mask.sum() / bins_per_day:
            observed[mask] = False

    # On unobserved days, fall back to the person's habitual sleep window so the
    # night is dark rather than lit. This is a stated assumption, not data - the
    # returned `observed` mask is what lets the filter widen its uncertainty
    # over these stretches instead of trusting them.
    if not observed.all() and habitual_sleep is not None:
        onset, offset_hour = habitual_sleep
        local_hour = np.array(
            [(t + timedelta(seconds=utc_offset_seconds)).hour
             + (t + timedelta(seconds=utc_offset_seconds)).minute / 60.0
             for t in grid]
        )
        in_window = (
            ((local_hour >= onset) | (local_hour < offset_hour))
            if onset > offset_hour
            else ((local_hour >= onset) & (local_hour < offset_hour))
        )
        asleep = np.where(observed, asleep, in_window)

    ceiling = clear_sky_lux(grid)

    # --- outdoor probability ----------------------------------------------
    # Logistic in step rate, floored at a small base rate, and forced to zero
    # when the sun is down (being outdoors at night implies no daylight) or
    # while asleep.
    activity_term = 1.0 / (
        1.0 + np.exp(-OUTDOOR_STEPS_SLOPE * (steps_per_min - OUTDOOR_STEPS_MIDPOINT))
    )
    p_outdoor = BASE_OUTDOOR_PROB + (1.0 - BASE_OUTDOOR_PROB) * activity_term
    p_outdoor = np.where(ceiling > 100.0, p_outdoor, 0.0)
    p_outdoor = np.where(asleep, 0.0, p_outdoor)

    # --- sample ------------------------------------------------------------
    light = np.empty((n_samples, n_bins))

    # Bins with no device data behind them are a guess, so they are sampled
    # with a wider spread. That extra spread is the honest cost of not knowing,
    # and it propagates into the posterior instead of being hidden.
    indoor_sd = np.where(observed, INDOOR_LOG_SD, INDOOR_LOG_SD * UNOBSERVED_SD_INFLATION)
    indoor = np.exp(
        rng.normal(INDOOR_LOG_MEAN, indoor_sd[None, :], size=(n_samples, n_bins))
    )
    # Artificial light does not vanish after sunset, but evening indoor levels
    # are lower than daytime ones.
    evening = ceiling <= 100.0
    indoor[:, evening] *= 0.45

    outdoor_draw = rng.random((n_samples, n_bins)) < p_outdoor[None, :]
    attenuation = rng.beta(2.0, 2.0 * (1 - OUTDOOR_ATTENUATION_MEAN) / OUTDOOR_ATTENUATION_MEAN,
                           size=(n_samples, n_bins))
    outdoor = ceiling[None, :] * attenuation
    # Outdoors is never dimmer than the indoor alternative.
    outdoor = np.maximum(outdoor, indoor)

    light = np.where(outdoor_draw, outdoor, indoor)

    sleep_lux = np.abs(rng.normal(SLEEP_LUX_MEAN, SLEEP_LUX_SD, size=(n_samples, n_bins)))
    light[:, asleep] = sleep_lux[:, asleep]

    np.clip(light, 0.0, 120000.0, out=light)

    return LightEnsemble(
        times=grid,
        hours=hours,
        light=light,
        wake=np.where(asleep, 0.0, 1.0),
        p_outdoor=p_outdoor,
        clear_sky_lux=ceiling,
        observed=observed,
    )

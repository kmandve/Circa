"""Activity-corrected heart-rate phase channel.

Clean-room implementation of the approach in Kim et al. 2023 (*J. R. Soc.
Interface* 20:20230030): remove the behavioural drivers of heart rate, then fit
a 24-hour harmonic to what is left. The published MATLAB reference has no
licence file, so nothing is vendored from it — the equations come from the
paper and the numerics are checked independently.

The key point is that **the time of minimum raw heart rate is not a circadian
marker**. Raw HR is dominated by movement, posture, stress and sleep state. What
carries phase is the residual rhythm after those are regressed out, and Kim et
al. found the resulting HR phase can sit more than two hours from sleep midpoint
in the same person — which is precisely why this is worth computing as an
independent channel rather than assumed to agree with sleep timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from circa.db.models import ExerciseSession, HeartRateMinute, StepMinute
from circa.normalize.sleep import sleep_intervals
from circa.phase.circular import kappa_from_sd_hours, wrap_hours
from circa.settings_store import RuntimeSettings

log = structlog.get_logger(__name__)

BIN_MINUTES = 5
# Below this, the harmonic is fitting noise and the phase is meaningless.
MIN_COVERAGE = 0.45
MIN_HOURS_SPAN = 30.0
# A real circadian heart-rate rhythm has an amplitude of roughly 5-10 bpm.
# Anything much below this is noise wearing a sinusoid.
MIN_AMPLITUDE_BPM = 2.0
# Amplitude-to-residual-noise ratio at which the channel is trusted fully.
# Below it, kappa is scaled down proportionally.
REFERENCE_SNR = 0.75
# Below this the rhythm is not distinguishable from noise at all and the channel
# abstains rather than voting weakly.
MIN_SNR = 0.35
# The fitted acrophase must actually sit inside the observed data. Excluding
# sleep leaves only ~15 h of each cycle, and a sinusoid fitted to a partial
# cycle will happily place its peak in the unobserved gap, where nothing
# constrains it. Require this fraction of observations within +/- 3 h of the
# fitted peak.
MIN_ACROPHASE_SUPPORT = 0.04
# Exercise raises HR for a while after it ends; this is the decay constant of
# that recovery term, not a claim about physiology beyond "it fades".
EXERCISE_RECOVERY_HOURS = 3.0

# Chain from the fitted HR acrophase to DLMO:
#   HR trough  = acrophase + 12 h
#   CBTmin     ~ HR trough           (peripheral proxy, imperfect)
#   DLMO       = CBTmin - 7 h        (Hilaire07's cbt_to_dlmo default)
# so DLMO ~ acrophase + 5 h. Every link is a population approximation, so the
# offset carries real uncertainty - see HR_OFFSET_SD_HOURS.
HR_ACROPHASE_TO_DLMO_HOURS = 5.0
HR_OFFSET_SD_HOURS = 1.5


@dataclass(slots=True)
class HRPhaseObservation:
    mu_hours: float        # estimated DLMO, local clock hours
    kappa: float
    acrophase_hours: float  # time of peak of the de-masked rhythm
    amplitude_bpm: float
    n_bins: int
    coverage: float
    r_squared: float
    detail: dict


def _load_series(
    session: Session, start: datetime, end: datetime
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, hr, steps, asleep, exercise_recency) on a uniform grid."""
    n_bins = int((end - start).total_seconds() // (BIN_MINUTES * 60))
    grid = np.array([start + timedelta(minutes=BIN_MINUTES * i) for i in range(n_bins)])

    hr = np.full(n_bins, np.nan)
    rows = session.execute(
        select(HeartRateMinute.ts, HeartRateMinute.bpm_median)
        .where(HeartRateMinute.ts >= start, HeartRateMinute.ts < end)
        .order_by(HeartRateMinute.ts)
    ).all()
    buckets: dict[int, list[float]] = {}
    for ts, bpm in rows:
        idx = int((ts - start).total_seconds() // (BIN_MINUTES * 60))
        if 0 <= idx < n_bins:
            buckets.setdefault(idx, []).append(float(bpm))
    for idx, values in buckets.items():
        hr[idx] = float(np.median(values))

    steps = np.zeros(n_bins)
    step_rows = session.execute(
        select(StepMinute.ts, StepMinute.steps)
        .where(StepMinute.ts >= start, StepMinute.ts < end)
    ).all()
    for ts, count in step_rows:
        idx = int((ts - start).total_seconds() // (BIN_MINUTES * 60))
        if 0 <= idx < n_bins:
            steps[idx] += float(count)

    asleep = np.zeros(n_bins)
    for s_start, s_end in sleep_intervals(session, start, end):
        lo = max(0, int((s_start - start).total_seconds() // (BIN_MINUTES * 60)))
        hi = min(n_bins, int((s_end - start).total_seconds() // (BIN_MINUTES * 60)) + 1)
        asleep[lo:hi] = 1.0

    # Time-decaying "recently exercised" regressor.
    recency = np.zeros(n_bins)
    for ex in session.scalars(
        select(ExerciseSession).where(
            ExerciseSession.end_ts > start - timedelta(hours=12),
            ExerciseSession.start_ts < end,
        )
    ):
        hours_since = np.array(
            [(g - ex.end_ts).total_seconds() / 3600 for g in grid], dtype=float
        )
        contribution = np.where(
            hours_since >= 0, np.exp(-hours_since / EXERCISE_RECOVERY_HOURS), 0.0
        )
        recency = np.maximum(recency, contribution)

    return grid, hr, steps, asleep, recency


def estimate(
    session: Session,
    settings: RuntimeSettings,
    as_of: datetime | None = None,
    tz_offset_seconds: int | None = None,
) -> HRPhaseObservation | None:
    """Fit the activity-corrected HR rhythm and return its phase."""
    as_of = as_of or datetime.now(UTC)
    start = as_of - timedelta(hours=settings.hr_window_hours)
    if (as_of - start).total_seconds() / 3600 < MIN_HOURS_SPAN:
        return None

    grid, hr, steps, asleep, exercise = _load_series(session, start, as_of)
    valid = ~np.isnan(hr)
    coverage = float(valid.mean()) if valid.size else 0.0
    if coverage < MIN_COVERAGE or valid.sum() < 100:
        log.info("hr_phase.insufficient_coverage", coverage=round(coverage, 3))
        return None

    # Local clock hours, so the resulting phase is directly comparable with the
    # sleep channel and interpretable on a calendar.
    if tz_offset_seconds is None:
        tz_offset_seconds = 0
    local = grid + timedelta(seconds=tz_offset_seconds)
    hours_of_day = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in local])
    omega = 2 * np.pi / 24.0

    # Exclude sleep rather than model it - see the module docstring.
    awake = valid & (asleep < 0.5)
    awake_coverage = float(awake.sum()) / max(int((~np.isnan(hr)).sum()), 1)
    if awake.sum() < 100:
        log.info("hr_phase.insufficient_wake_data", n=int(awake.sum()))
        return None
    valid = awake

    log_steps = np.log1p(steps)
    columns = [
        ("intercept", np.ones_like(hours_of_day)),
        ("cos", np.cos(omega * hours_of_day)),
        ("sin", np.sin(omega * hours_of_day)),
        # --- behavioural de-masking terms ---
        ("steps", log_steps),
        ("steps2", log_steps**2),
        ("exercise", exercise),
    ]

    # Drop de-masking regressors that are constant or collinear within this
    # window. On a very regular schedule the quadratic step term and the sleep
    # indicator can be near-duplicates, which makes the design rank-deficient
    # and the phase standard error meaningless. The first three columns are
    # structural and never dropped.
    keep = list(range(3))
    for i in range(3, len(columns)):
        candidate = np.column_stack([columns[j][1] for j in [*keep, i]])[valid]
        if np.std(columns[i][1][valid]) < 1e-9:
            continue
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            keep.append(i)
    dropped = [columns[i][0] for i in range(3, len(columns)) if i not in keep]

    design = np.column_stack([columns[i][1] for i in keep])[valid]
    y = hr[valid]

    try:
        import statsmodels.api as sm

        # Heteroskedasticity-robust: HR residual variance is much larger while
        # awake and moving than during sleep, which would otherwise understate
        # the phase standard error.
        fit = sm.OLS(y, design).fit(cov_type="HC1")
        params, cov = fit.params, fit.cov_params()
        r_squared = float(fit.rsquared)
    except Exception as exc:  # noqa: BLE001
        log.warning("hr_phase.fit_failed", error=str(exc))
        return None

    a, b = float(params[1]), float(params[2])
    amplitude = float(np.hypot(a, b))
    if amplitude < MIN_AMPLITUDE_BPM:
        log.info("hr_phase.amplitude_too_small", amplitude=round(amplitude, 2))
        return None

    # Signal-to-noise of the rhythm itself. A 3 bpm oscillation buried in 12 bpm
    # of residual scatter carries little phase information however tidy the
    # point estimate looks.
    residual_sd = float(np.std(fit.resid)) if hasattr(fit, "resid") else float("nan")
    snr = amplitude / residual_sd if residual_sd > 1e-6 else 0.0
    if snr < MIN_SNR:
        log.info("hr_phase.rhythm_indistinguishable_from_noise",
                 amplitude=round(amplitude, 2), residual_sd=round(residual_sd, 2),
                 snr=round(snr, 3))
        return None

    # atan2(b, a) is the phase of the peak of a*cos + b*sin.
    acrophase = float(wrap_hours(np.arctan2(b, a) / omega))

    # Is the peak actually observed, or extrapolated into the sleep gap?
    observed_hours = hours_of_day[valid]
    distance = np.abs(((observed_hours - acrophase + 12.0) % 24.0) - 12.0)
    support = float(np.mean(distance <= 3.0))
    if support < MIN_ACROPHASE_SUPPORT:
        log.info(
            "hr_phase.acrophase_unobserved",
            acrophase=round(acrophase, 2), support=round(support, 4),
            note="fitted peak lies outside the observed window; refusing to guess",
        )
        return None

    # Propagate the regression covariance into a phase standard error rather
    # than inventing a confidence heuristic: for phi = atan2(b, a),
    #   var(phi) = (a^2 var(b) + b^2 var(a) - 2ab cov(a,b)) / (a^2 + b^2)^2
    var_a = float(cov[1, 1])
    var_b = float(cov[2, 2])
    cov_ab = float(cov[1, 2])
    denom = (a * a + b * b) ** 2
    var_phi_rad = (a * a * var_b + b * b * var_a - 2 * a * b * cov_ab) / max(denom, 1e-12)
    sd_hours_fit = float(np.sqrt(max(var_phi_rad, 0.0)) / omega)

    # The acrophase->DLMO chain is a population approximation; its uncertainty
    # adds to the fit's own.
    sd_total = float(np.hypot(sd_hours_fit, HR_OFFSET_SD_HOURS))
    kappa = kappa_from_sd_hours(sd_total)
    # Poor coverage means the residual rhythm is partly interpolation.
    kappa *= float(np.clip(coverage, 0.0, 1.0))
    # And a weak rhythm should not speak loudly. Without this the channel
    # claimed a concentration comparable to sleep timing while fitting a 3 bpm
    # wobble.
    kappa *= float(np.clip(snr / REFERENCE_SNR, 0.05, 1.0))

    dlmo_hours = float(wrap_hours(acrophase + HR_ACROPHASE_TO_DLMO_HOURS))

    return HRPhaseObservation(
        mu_hours=dlmo_hours,
        kappa=float(max(kappa, 1e-3)),
        acrophase_hours=acrophase,
        amplitude_bpm=amplitude,
        n_bins=int(valid.sum()),
        coverage=coverage,
        r_squared=r_squared,
        detail={
            "sd_hours_fit": round(sd_hours_fit, 3),
            "sd_hours_total": round(sd_total, 3),
            "offset_applied_hours": HR_ACROPHASE_TO_DLMO_HOURS,
            "window_hours": settings.hr_window_hours,
            "residual_sd_bpm": round(residual_sd, 2),
            "amplitude_snr": round(snr, 3),
            "acrophase_support": round(support, 3),
            "awake_fraction_of_valid": round(awake_coverage, 3),
            "sleep_excluded": True,
            "regressors": [columns[i][0] for i in keep],
            "dropped_regressors": dropped,
            "coefficients": {
                columns[i][0]: round(float(params[k]), 4)
                for k, i in enumerate(keep)
                if i >= 3
            },
        },
    )

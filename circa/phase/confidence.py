"""Confidence tiers and the quality score.

Confidence is never a single magic percentage. It is derived from measurable
components and, critically, **night count alone never advances a tier** — poor
data coverage, disagreeing channels, or an out-of-distribution day can hold it
back or knock it down. That is what stops the system looking most confident
precisely when a schedule has just been disrupted, which is when published
wearable models degrade worst.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Night thresholds for each tier. Necessary but not sufficient.
TIER_NIGHTS = {0: 0, 1: 7, 2: 14, 3: 30}

TIER_NAMES = {
    0: "cold start",
    1: "provisional",
    2: "personalized",
    3: "mature",
}

TIER_NOTES = {
    0: "Population priors only. Blocks are wide and few, and say so.",
    1: "Provisional personal estimate. Focus blocks enabled.",
    2: "Personal offsets beginning to override population priors.",
    3: "Full detail; personal offsets fitted; backtest metrics meaningful.",
}

# The narrowest 80% credible interval this system can honestly produce, in
# minutes. The posterior is convolved with a model-adequacy term of ~33 min SD
# (see particle_filter.MODEL_ADEQUACY_SD_HOURS), so an 80% interval spans at
# least 2 * 1.2816 * 33 ~= 85 min however much data arrives. Scoring against
# zero width would peg q_posterior near 0.35 forever and make the tier system
# inert - the score has to be relative to what is achievable, not to perfection.
CI_WIDTH_FLOOR_MINUTES = 85.0
# Additional width beyond the floor at which the posterior term falls to 1/e.
CI_WIDTH_SCALE_MINUTES = 75.0


@dataclass
class Confidence:
    tier: int
    tier_name: str
    note: str
    q: float                       # 0-1 overall quality
    q_posterior: float
    q_coverage: float
    q_agreement: float
    q_ood: float
    n_nights: int
    ood_flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def is_low(self) -> bool:
        return self.tier == 0 or self.q < 0.45


def assess(
    n_nights: int,
    ci80_width_minutes: float,
    data_coverage: float,
    channel_agreement_minutes: float | None,
    ood_flags: list[str] | None = None,
) -> Confidence:
    """Combine the components into a tier and a 0-1 quality score."""
    ood_flags = ood_flags or []

    # Wide credible intervals mean low confidence, measured against the
    # narrowest interval the model can honestly produce rather than against zero.
    excess = max(ci80_width_minutes - CI_WIDTH_FLOOR_MINUTES, 0.0)
    q_posterior = float(np.exp(-excess / CI_WIDTH_SCALE_MINUTES))

    # Sensor completeness. Below ~70% coverage the HR channel is largely
    # interpolation.
    q_coverage = float(np.clip(data_coverage, 0.0, 1.0))

    # Independent channels agreeing is real evidence; disagreeing by hours means
    # at least one is wrong and we do not know which.
    if channel_agreement_minutes is None:
        # Only one channel available - not disagreement, but not corroboration.
        q_agreement = 0.7
    else:
        q_agreement = float(np.exp(-channel_agreement_minutes / 120.0))

    # Each OOD flag is a multiplicative penalty. These are exactly the
    # conditions under which published models fail badly.
    q_ood = float(0.6 ** len(ood_flags))

    q = q_posterior * q_coverage * q_agreement * q_ood

    # Nights set a ceiling; quality can pull the tier back down below it.
    tier = 0
    for candidate in (1, 2, 3):
        if n_nights >= TIER_NIGHTS[candidate]:
            tier = candidate
    if q < 0.25:
        tier = 0
    elif q < 0.45:
        tier = min(tier, 1)
    elif q < 0.6:
        tier = min(tier, 2)

    return Confidence(
        tier=tier,
        tier_name=TIER_NAMES[tier],
        note=TIER_NOTES[tier],
        q=round(q, 4),
        q_posterior=round(q_posterior, 4),
        q_coverage=round(q_coverage, 4),
        q_agreement=round(q_agreement, 4),
        q_ood=round(q_ood, 4),
        n_nights=n_nights,
        ood_flags=ood_flags,
        detail={
            "ci80_width_minutes": round(ci80_width_minutes, 1),
            "channel_agreement_minutes": (
                round(channel_agreement_minutes, 1)
                if channel_agreement_minutes is not None
                else None
            ),
            "tier_ceiling_from_nights": max(
                (t for t in (0, 1, 2, 3) if n_nights >= TIER_NIGHTS[t]), default=0
            ),
        },
    )


def detect_ood(
    session,
    as_of,
    sleep_sd_hours: float | None = None,
    illness: bool = False,
    travel_mode: bool = False,
    hr_coverage: float | None = None,
    last_sleep_hours: float | None = None,
) -> list[str]:
    """Flag conditions where the model is outside its comfortable regime."""
    flags: list[str] = []
    if travel_mode:
        flags.append("travel")
    if illness:
        flags.append("illness")
    if sleep_sd_hours is not None and sleep_sd_hours > 2.0:
        flags.append("irregular_sleep")
    if hr_coverage is not None and hr_coverage < 0.5:
        flags.append("low_hr_coverage")
    if last_sleep_hours is not None:
        if last_sleep_hours < 3.0:
            flags.append("all_nighter")
        elif last_sleep_hours > 13.0:
            flags.append("unusually_long_sleep")
    return flags

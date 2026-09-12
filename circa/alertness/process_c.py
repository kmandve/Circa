"""Process C — circadian drive on alertness.

Anchored to CBTmin rather than to clock time, because that is what the phase
posterior actually estimates. A two-harmonic waveform is used because the
alertness rhythm is visibly non-sinusoidal: a single cosine cannot produce both
the early-afternoon dip and the evening wake-maintenance zone, and those are the
two features the calendar most needs to get right.

Qualitative features the parameters are chosen to reproduce (all standard
findings in the circadian literature):

  * minimum at CBTmin (deep biological night)
  * a secondary dip in the early afternoon - the "post-lunch" dip, which occurs
    without lunch and is circadian rather than digestive
  * a pronounced evening peak, the wake-maintenance zone, in the hours before
    DLMO - the window in which it is hardest to fall asleep
"""

from __future__ import annotations

import numpy as np

OMEGA = 2.0 * np.pi / 24.0

# Second-harmonic weight and phase, fitted so the waveform's turning points land
# on the documented landmarks rather than merely near them. Targets, in hours
# after CBTmin, from the alertness literature for a conventional sleeper:
#
#   morning peak   CBTmin +  6.8 h   (late morning)
#   afternoon dip  CBTmin + 10.4 h   (the post-lunch dip, which is circadian
#                                      rather than digestive - it happens
#                                      without lunch)
#   evening peak   CBTmin + 15.7 h   (the wake-maintenance zone)
#   nadir          CBTmin            (deep biological night)
#
# The previous values (0.40, pi/8) put the morning peak at CBTmin + 8.5 h and
# the dip at + 9.9 h - only 1.4 h apart. The two turning points merged into a
# plateau, and once the homeostat was subtracted the afternoon "dip" came out
# as a 2.7-point wobble on a 0-100 scale. An afternoon dip is one of the two
# things this whole calendar exists to tell you about, and it was effectively
# absent.
SECOND_HARMONIC_WEIGHT = 0.62
SECOND_HARMONIC_PHASE = 0.4909


def drive(hours_since_cbtmin: np.ndarray) -> np.ndarray:
    """Circadian alertness drive, normalised to roughly [-1, 1].

    `hours_since_cbtmin` may be any real number; only its value mod 24 matters.
    """
    theta = OMEGA * np.asarray(hours_since_cbtmin, dtype=float)
    fundamental = -np.cos(theta)
    harmonic = -SECOND_HARMONIC_WEIGHT * np.cos(2 * theta + SECOND_HARMONIC_PHASE)
    raw = fundamental + harmonic
    # Normalise by the theoretical maximum so the amplitude is interpretable
    # regardless of the harmonic weight.
    return raw / (1.0 + SECOND_HARMONIC_WEIGHT)


def hours_since(reference_hours: float, times_local_hours: np.ndarray) -> np.ndarray:
    """Signed hours from a reference clock hour to each local clock hour."""
    return np.mod(np.asarray(times_local_hours, dtype=float) - reference_hours, 24.0)


def wake_maintenance_zone(cbtmin_hours: float) -> tuple[float, float]:
    """The evening window where sleep onset is hardest, in local clock hours.

    Sits roughly 13-17 h after CBTmin, i.e. the few hours before DLMO, and
    brackets the waveform's evening peak at CBTmin + 15.7 h. Reported so the
    calendar never suggests winding down inside it - that is the window in which
    trying to sleep is least likely to work.
    """
    return (np.mod(cbtmin_hours + 13.0, 24.0), np.mod(cbtmin_hours + 17.0, 24.0))


def circadian_dip(cbtmin_hours: float) -> tuple[float, float]:
    """The afternoon trough, in local clock hours.

    Brackets the waveform's secondary minimum at CBTmin + 10.4 h. This is the
    "post-lunch dip", which is circadian rather than digestive - it occurs
    without lunch.
    """
    return (np.mod(cbtmin_hours + 8.5, 24.0), np.mod(cbtmin_hours + 11.5, 24.0))

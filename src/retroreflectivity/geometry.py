"""
geometry.py — IRC:35-2015 observation geometry and correction factors.

The IRC:35-2015 standard specifies retroreflectivity measurement at a 30-metre
observation distance with:
    - Observation angle (α) = 2.29°   (angle subtended at the marking by the
      sensor and the illuminator, viewed from the marking centre)
    - Illumination angle (β) = 1.24°  (angle between the illumination beam
      and the road surface)

When the actual sensor-to-marking distance differs from 30 m, the observation
and illumination angles change.  This module derives actual angles from the
measured distance (via ultrasonic sensor) and the known vehicle/sensor height,
and computes a multiplicative correction factor so that the RL reading can be
normalised to the standard 30-metre geometry.

Usage:
    from src.retroreflectivity.geometry import (
        calculate_angles, geometry_correction_factor, distance_to_geometry_valid,
    )
"""

from __future__ import annotations

import math
from typing import Dict

from config import (
    OBSERVATION_ANGLE_DEG,
    ILLUMINATION_ANGLE_DEG,
    VALID_DISTANCE_RANGE_CM,
    VEHICLE_HEIGHT_CM,
)


def calculate_angles(
    distance_cm: float,
    vehicle_height_cm: float = VEHICLE_HEIGHT_CM,
) -> Dict[str, float]:
    """Compute actual observation and illumination angles.

    For a sensor mounted at height *h* measuring a marking at ground-level
    distance *d*:

        observation_angle  = arctan(h / d)
        illumination_angle ≈ observation_angle × (1.24 / 2.29)

    The illumination angle is scaled proportionally because the headlamp
    (illuminator) is co-located near the sensor on the vehicle.

    Parameters
    ----------
    distance_cm : float
        Horizontal distance from vehicle to marking in centimetres.
    vehicle_height_cm : float
        Height of the sensor/camera above the road surface in centimetres.

    Returns
    -------
    Dict[str, float]
        Keys: ``observation_angle_deg``, ``illumination_angle_deg``.

    Raises
    ------
    ValueError
        If *distance_cm* ≤ 0.
    """
    if distance_cm <= 0:
        raise ValueError(f"distance_cm must be positive, got {distance_cm}")

    obs_angle = math.degrees(math.atan(vehicle_height_cm / distance_cm))
    # Scale illumination angle proportionally
    illum_angle = obs_angle * (ILLUMINATION_ANGLE_DEG / OBSERVATION_ANGLE_DEG)

    return {
        "observation_angle_deg": round(obs_angle, 4),
        "illumination_angle_deg": round(illum_angle, 4),
    }


def geometry_correction_factor(
    actual_obs_angle: float,
    standard_obs_angle: float = OBSERVATION_ANGLE_DEG,
) -> float:
    """Compute the multiplicative geometry correction.

    The retroreflected luminance coefficient (RL) is proportional to the
    cosine of the observation angle.  When measuring at an angle that
    differs from the standard 2.29°, a cosine-ratio correction normalises
    the result:

        correction = cos(actual) / cos(standard)

    Parameters
    ----------
    actual_obs_angle : float
        Actual observation angle in degrees (derived from distance + height).
    standard_obs_angle : float
        IRC:35-2015 standard observation angle (default 2.29°).

    Returns
    -------
    float
        Multiplicative correction factor (close to 1.0 when distance ≈ 30 m
        and sensor height ≈ 1.2 m).
    """
    return math.cos(math.radians(actual_obs_angle)) / math.cos(
        math.radians(standard_obs_angle)
    )


def distance_to_geometry_valid(distance_cm: float) -> bool:
    """Check whether the sensor-to-marking distance is within range.

    Valid range is 2 m – 50 m (200–5000 cm) per engineering practice.

    Parameters
    ----------
    distance_cm : float
        Measured distance in centimetres.

    Returns
    -------
    bool
        ``True`` if the measurement geometry is reliable.
    """
    lo, hi = VALID_DISTANCE_RANGE_CM
    return lo <= distance_cm <= hi

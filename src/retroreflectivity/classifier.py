"""
classifier.py — IRC:35-2015 Green / Amber / Red threshold classifier.

Maps corrected RL values to compliance status based on object type.

Usage:
    from src.retroreflectivity.classifier import classify_rl, get_status_color_bgr
    status = classify_rl(280.0, "white_lane_marking")   # "AMBER"
    colour = get_status_color_bgr("AMBER")               # (0, 165, 255)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import CLASS_TO_IRC_KEY, IRC_THRESHOLDS


def classify_rl(rl_value: float, object_type: str) -> str:
    """Classify a retroreflectivity value per IRC:35-2015.

    Parameters
    ----------
    rl_value : float
        Corrected RL in mcd/m²/lx.
    object_type : str
        Unified class name (e.g. ``"white_lane_marking"``).

    Returns
    -------
    str
        ``"GREEN"``, ``"AMBER"``, or ``"RED"``.
    """
    irc_key = CLASS_TO_IRC_KEY.get(object_type, "white_marking")
    thresholds = IRC_THRESHOLDS.get(irc_key, {"green": 300, "amber": 150})

    if rl_value >= thresholds["green"]:
        return "GREEN"
    elif rl_value >= thresholds["amber"]:
        return "AMBER"
    else:
        return "RED"


def get_status_color_bgr(status: str) -> Tuple[int, int, int]:
    """Return a BGR colour tuple for OpenCV drawing.

    Parameters
    ----------
    status : str
        ``"GREEN"``, ``"AMBER"``, or ``"RED"``.

    Returns
    -------
    Tuple[int, int, int]
        BGR colour.
    """
    mapping = {
        "GREEN": (0, 255, 0),
        "AMBER": (0, 165, 255),
        "RED": (0, 0, 255),
    }
    return mapping.get(status.upper(), (255, 255, 255))


def generate_summary_stats(measurements: List[Dict]) -> Dict:
    """Compute aggregate statistics from a list of measurement dicts.

    Parameters
    ----------
    measurements : List[Dict]
        Each dict must have at least ``rl_mcd`` and ``status`` keys.

    Returns
    -------
    Dict
        Keys: ``total``, ``green_count``, ``amber_count``, ``red_count``,
        ``avg_rl``, ``min_rl``, ``max_rl``, ``compliance_pct``.
    """
    if not measurements:
        return {
            "total": 0,
            "green_count": 0,
            "amber_count": 0,
            "red_count": 0,
            "avg_rl": 0.0,
            "min_rl": 0.0,
            "max_rl": 0.0,
            "compliance_pct": 0.0,
        }

    rl_vals = [m.get("rl_mcd", 0.0) for m in measurements]
    statuses = [m.get("status", "RED").upper() for m in measurements]

    green = statuses.count("GREEN")
    amber = statuses.count("AMBER")
    red = statuses.count("RED")
    total = len(measurements)

    return {
        "total": total,
        "green_count": green,
        "amber_count": amber,
        "red_count": red,
        "avg_rl": round(sum(rl_vals) / total, 2),
        "min_rl": round(min(rl_vals), 2),
        "max_rl": round(max(rl_vals), 2),
        "compliance_pct": round(green / total * 100, 1) if total else 0.0,
    }

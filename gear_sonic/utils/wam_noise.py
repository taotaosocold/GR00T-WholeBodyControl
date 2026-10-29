"""One sensor-noise definition for WAM collection, imagination and evaluation."""

import math


# SuperMimic OBS_NOISE, expressed as uniform half-widths in Sonic term order.
WAM_OBSERVATION_NOISE = {
    "gravity_dir": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "actions": 0.0,
}


def wam_observation_noise(settings=None):
    """Return a validated copy, including optional WAM-only overrides."""
    values = WAM_OBSERVATION_NOISE.copy()
    overrides = (settings or {}).get("observation_noise", {}) or {}
    if set(overrides) - set(values):
        raise ValueError("WAM observation_noise contains unknown proprioceptive terms")
    for name, value in overrides.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("WAM observation noise half-widths must be finite and nonnegative")
        values[name] = value
    return values

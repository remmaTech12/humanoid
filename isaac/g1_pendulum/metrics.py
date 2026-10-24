"""Pure-Python pendulum attitude metrics (no Isaac Sim dependency)."""

import math


def rod_tilt_deg(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return (unsigned tilt, signed tilt toward +X, signed tilt toward +Y).

    The rod direction points from pivot to tip in world coordinates, with +Z up.
    """
    x, y, z = direction
    magnitude = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Rod direction must be finite and nonzero.")
    x, y, z = x / magnitude, y / magnitude, z / magnitude
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, z))))
    return tilt, math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, z))

"""
cow/curves.py
=============
Lactation curves evaluated at REAL calendar days (no model-step units here;
cow/tables.py does the step averaging). Two selectable methods, chosen by
cow.parameters.MILK_CURVE_METHOD ("wood" default | "milkbot"):

- wood:    MP(t) = a * t^b * exp(-c*t)   (Wood 1967; parameters:
           cow.parameters.WOOD1967 -- a/peak from Evangelista et al. 2024,
           b and c derived; see that dict for provenance and caveats)
- milkbot: MP(t) = a * (1 - exp((c - t)/b)/2) * exp(-d*t)   (Ehrlich 2011/2013 Eq. 5, printed;
           parameters Ehrlich 2011 Table 1, Holstein median -- cow.parameters.MILKBOT_EHRLICH2011)

`lactation` is 1..MAX_LACTATION; each lactation uses its SOURCE lactation's parameters
(cow.parameters.MILK_CURVE_SOURCE_LACTATION) and the evidence status of that choice is in
cow.parameters.MILK_CURVE_BASIS ('source' / 'repeated' / 'placeholder').
"""

import numpy as np
from cow.parameters import (
    MILKBOT_EHRLICH2011,
    WOOD1967, MILK_CURVE_METHOD, MILK_CURVE_SOURCE_LACTATION, MAX_LACTATION,
)


def milkbot_kg(days, lac_bucket):
    """MilkBot (Ehrlich 2011/2013): Y = a * (1 - exp((c - t)/b) / 2) * exp(-d t); a = scale (NOT the peak), b = ramp,
    c = offset, d = decay. Parameters: cow.parameters.MILKBOT_EHRLICH2011 (Holstein median, parities 1-3)."""
    p = MILKBOT_EHRLICH2011[lac_bucket]
    return np.maximum(0.0, p["a"] * (1 - np.exp((p["c"] - days) / p["b"]) / 2) * np.exp(-p["d"] * days))


def wood_kg(days, lac_bucket):
    p = WOOD1967["parameters_by_parity"][lac_bucket]
    if WOOD1967["calibration_required"] or any(v is None for v in p.values()):
        raise ValueError(
            f"MILK_CURVE_METHOD='wood' but WOOD1967['parameters_by_parity'][{lac_bucket}] "
            f"is not populated (cow/parameters.py). This project never guesses a parameter."
        )
    t = np.maximum(days, 0.0)
    return np.maximum(0.0, p["a"] * t ** p["b"] * np.exp(-p["c"] * t))


def curve_kg(days, lactation, method=None):
    """Potential milk (kg/day) at real day(s) `days` for modelled lactation 1..MAX_LACTATION."""
    method = method or MILK_CURVE_METHOD
    src = MILK_CURVE_SOURCE_LACTATION[min(max(int(lactation), 1), MAX_LACTATION)]
    if method == "wood":
        return wood_kg(days, src)
    if method == "milkbot":
        return milkbot_kg(days, src)
    raise ValueError(f"Unknown MILK_CURVE_METHOD={method!r} -- expected 'wood' or 'milkbot'.")

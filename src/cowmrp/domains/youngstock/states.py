"""
youngstock/states.py
====================
The youngstock state table. Its coordinates are AGE-BASED, deliberately NOT
the cow model's (lactation, DIM, DP, health): a calf has no lactation/DIM,
so those columns would be meaningless for it (see the design discussion).

Columns
-------
line   : DAIRY_F   female dairy calf on the replacement pathway
         DAIRY_M   male dairy calf            (terminal: sold)
         BEEF_F    beef-cross female calf     (terminal: sold)
         BEEF_M    beef-cross male calf       (terminal: sold)
         ALL       only for the STILLBORN outcome
stage  : CALF | HEIFER | BREEDING | PREGNANT            (alive, transient)
         DIED | CULLED | GRADUATED | SOLD | STILLBORN   (absorbing outcomes)
         SOLD is a terminal-line sale OR the birth-time sale of a surplus female
         dairy calf (FEMALE_DAIRY_RETENTION < 1); DIED on a terminal line is a
         pre-sale death.
age    : model steps completed since birth (CALF/HEIFER, terminal calves)
dp     : model steps pregnant (PREGNANT only, 1..DP_CALVE)
age_days : age * UNIT_DAYS -- the real-day age at the START of the step, the
           one column probability specs read (never a re-derived value).
absorbing/alive : convenience flags.

Absorbing outcomes are real states (each with a self-loop of 1) so the
youngstock transition matrix Q is fully row-stochastic and is validated by
the SAME mrp_core.validate_row_stochasticity as the cow matrix -- the check
is never weakened. An absorbing state's mass is CUMULATIVE; per-step flows
are the change between steps.
"""

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.domains.youngstock.parameters import K_BREED, K_SALE, DP_CALVE, WEANING_AGE_DAYS

TERMINAL_LINES = ("DAIRY_M", "BEEF_F", "BEEF_M")
ABSORBING_STAGES = ("DIED", "CULLED", "GRADUATED", "SOLD", "STILLBORN")


def build_youngstock_states():
    rows = []
    # --- DAIRY_F replacement pathway ---
    for k in range(K_BREED):
        stage = "CALF" if k * UNIT_DAYS < WEANING_AGE_DAYS else "HEIFER"
        rows.append(("DAIRY_F", stage, k, 0))
    rows.append(("DAIRY_F", "BREEDING", 0, 0))
    for dp in range(1, DP_CALVE + 1):
        rows.append(("DAIRY_F", "PREGNANT", 0, dp))
    for stage in ("DIED", "CULLED", "GRADUATED", "SOLD"):
        rows.append(("DAIRY_F", stage, 0, 0))
    # --- terminal lines: calf ages 0..K_SALE (sold DURING step K_SALE) ---
    for line in TERMINAL_LINES:
        for k in range(K_SALE[line] + 1):
            rows.append((line, "CALF", k, 0))
        rows.append((line, "SOLD", 0, 0))
        rows.append((line, "DIED", 0, 0))          # pre-sale death of a terminal calf (booked at birth)
    rows.append(("ALL", "STILLBORN", 0, 0))

    states = pd.DataFrame(rows, columns=["line", "stage", "age", "dp"])
    states["age_days"] = states["age"] * UNIT_DAYS
    states["absorbing"] = states["stage"].isin(ABSORBING_STAGES)
    states["alive"] = ~states["absorbing"]
    states["state_id"] = states.index
    if states.duplicated(subset=["line", "stage", "age", "dp"]).any():
        raise ValueError("youngstock state table has duplicate (line, stage, age, dp) rows")
    return states

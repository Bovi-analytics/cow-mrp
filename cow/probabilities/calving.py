"""
cow/probabilities/calving.py
=============================
Calving probability: p_calve (successful pregnancy completion and birth).
"""

import numpy as np
import pandas as pd
from cow.parameters import scale, UNIT_DAYS, GESTATION_LENGTH_DAYS, GESTATION_LENGTH_SD_DAYS, MAX_DP
from cow.probabilities.base import lookup, resolve_method, dp_bucket_calv, age_offset_slots


calving_table = pd.DataFrame([
    {'dp_bucket': '0-270',  'p': 0.0},
    {'dp_bucket': '271-290','p': 1.0},
])


def p_calving_table_legacy(states, context):
    """PRIOR implementation of the 'table' method -- RETAINED UNCHANGED for
    direct comparison, no longer wired into calving_spec['methods']
    (2026-08-27 resolution-consistency fix; see p_calving_table below).
    Looks up ONE calving_table bucket for the whole model step, using a
    hardcoded 270-day cutoff that was never reconciled with
    GESTATION_LENGTH_DAYS (used everywhere else in this model,
    including p_calving_formula's own midpoint below and
    cow.parameters.DRY_OFF_DP_DAYS = GESTATION_LENGTH_DAYS -
    DRY_OFF_DAYS_BEFORE_CALVING) -- confirmed present, unreconciled, since
    this repo's very first commit (2026-08-27 calving audit). At MONTH
    resolution this bucket lookup's own coarseness additionally delayed
    the actual calving trigger to real day 300 (not even 270) -- confirmed
    the dominant driver of an inflated, resolution-INCONSISTENT Dry-flag
    sojourn (~82 days at MONTH vs ~55 days at WEEK, both compared against
    a ~55-58 day literature benchmark). See cow/probabilities/
    conception.py's analogous p_conception_table_legacy."""
    dp = states['dp'].values
    return scale(lookup(calving_table, {'dp_bucket': dp_bucket_calv(dp)}))


def _calving_day_exact(dp_step):
    """Day-exact calving trigger. Calving is a deterministic event at real
    gestation day GESTATION_LENGTH_DAYS, not a compounding hazard --
    so unlike conception.py/abortion.py's survival-product integration,
    this only needs to ask whether the step's own real-day span
    [dp_step*UNIT_DAYS, dp_step*UNIT_DAYS + UNIT_DAYS) actually CONTAINS
    day 280, rather than looking up which coarse (to_units-rounded) bucket
    the step's own index falls into. At DAY resolution (UNIT_DAYS=1) this
    triggers at exactly dp=280 with zero discretization error; at WEEK it
    triggers in the 7-day window containing day 280; at MONTH it triggers
    in the 30-day window containing day 280 (real day 270 -- 30 days
    earlier than the legacy table's own day-300 trigger)."""
    starts = dp_step * UNIT_DAYS
    ends = starts + UNIT_DAYS - 1
    return np.where(ends >= GESTATION_LENGTH_DAYS, 1.0, 0.0)


def calving_step_mass(sd_days=0.0):
    """Fraction of a conception cohort that calves in pregnant step k, {k: mass}.

    Conception happens uniformly inside the step before dp=1, so at the first boundary the
    gestation age is u ~ Uniform(0, UNIT_DAYS) and step k spans ages (k-1)*U+u .. k*U+u. With sd_days = 0
    (the default 'table' method) a pregnancy calves in the step whose span contains GESTATION_LENGTH_DAYS,
    i.e. k = floor((G-u)/U) + 1: at DAY that is one step for everyone; at WEEK/MONTH the mass may spread
    across 2 adjacent steps depending on whether G divides U evenly (verified numerically whenever
    GESTATION_LENGTH_DAYS changes, not hardcoded here). With sd_days > 0 (the 'model' method) gestation
    length is Normal(GESTATION_LENGTH_DAYS, sd_days) and the mass of step k is the drop of the entry-averaged
    survival across its two boundaries (masses below 1e-6 are dropped; the last kept step resolves the rest).
    Slot midpoints discretize u."""
    slot = age_offset_slots()
    if sd_days <= 0:
        k_u = np.floor((GESTATION_LENGTH_DAYS - slot) / UNIT_DAYS).astype(int) + 1
        mass = {}
        for k in k_u:
            mass[int(k)] = mass.get(int(k), 0.0) + 1.0 / len(slot)
        return mass
    from scipy.special import ndtr
    k_hi = int(np.ceil((GESTATION_LENGTH_DAYS + 8 * sd_days) / UNIT_DAYS)) + 2
    Sbar = np.array([np.mean(1.0 - ndtr(((k - 1) * UNIT_DAYS + slot - GESTATION_LENGTH_DAYS) / sd_days)) for k in range(k_hi + 2)])
    mass = {k: float(Sbar[k] - Sbar[k + 1]) for k in range(1, k_hi + 1) if Sbar[k] - Sbar[k + 1] > 1e-6}
    return mass


def active_calving_mass():
    """calving_step_mass() of the ACTIVE calving method (used by cow/tables.py to size the state space)."""
    return calving_step_mass(GESTATION_LENGTH_SD_DAYS if calving_spec['method'] == 'model' else 0.0)


def calving_step_probability(dp, sd_days=0.0):
    """P(calve in pregnant step dp | still pregnant at its start), vectorized over dp (0 for dp = 0).
    The last step with any calving mass has probability exactly 1 (no pregnancy is left unresolved)."""
    mass = calving_step_mass(sd_days)
    k_last = max(mass)
    if k_last > MAX_DP:
        if sd_days <= 0:
            raise ValueError(f"calving reaches pregnant step {k_last} > MAX_DP={MAX_DP}: raise COW_MAX_DP_DAYS")
        k_last = MAX_DP        # normal tail beyond the state space (< 1e-5 of calvings) is resolved in the last step
    p = np.zeros(MAX_DP + 1)
    left = 1.0
    for k in range(1, MAX_DP + 1):
        m = mass.get(k, 0.0)
        p[k] = 1.0 if k >= k_last else (m / left if left > 1e-12 else 1.0)
        left -= m
    return p[np.clip(np.asarray(dp), 0, MAX_DP)]


def p_calving_table(states, context):
    """ACTIVE 'table' method (TIME-CONVENTION FIX, 2026-09-19): the per-step calving probability that
    follows from the SAME uniform-conception-time convention as the corrected abortion/conception/DNB
    functions (see calving_step_mass). Replaces the deterministic 'step containing day 280' trigger
    (p_calving_table_deterministic_legacy below): at MONTH that trigger fired at dp=9 for every cow, an
    effective gestation of 270 d once DIM 0 is read as 'calved somewhere in the step'; the averaged form
    puts 2/3 of calvings at dp 9 and 1/3 at dp 10 (mean gestation 280 d). Identical at WEEK and DAY."""
    return calving_step_probability(states['dp'].values)


def p_calving_model(states, context):
    """'model' method: gestation length ~ Normal(GESTATION_LENGTH_DAYS, GESTATION_LENGTH_SD_DAYS = 5.9 d)
    (Mwangi et al. 2025, class A for the SD; the mean is the model's own GESTATION_LENGTH_DAYS), with the same
    uniform-conception-time convention as the table. Pregnant cows calve over a ~+/-12 d spread instead of all on
    one day, so the dry period and the next lactation start vary. Select with calving_spec['method'] = 'model'
    (cow/tables.py widens the state space to the spread automatically)."""
    return calving_step_probability(states['dp'].values, GESTATION_LENGTH_SD_DAYS)


def p_calving_table_deterministic_legacy(states, context):
    """
    LEGACY (RESOLUTION-CONSISTENCY FIX of 2026-08-27, retained unchanged, unwired since 2026-09-19). See
    _calving_day_exact's docstring for the day-exact trigger this now uses
    instead of the coarse, unreconciled 270-day bucket lookup
    (p_calving_table_legacy above, retained for comparison). Uses
    GESTATION_LENGTH_DAYS -- the SAME reference point already used
    by p_calving_formula's own midpoint and by cow.parameters.
    DRY_OFF_DP_DAYS's "55 days before calving" definition -- resolving the
    confirmed internal inconsistency (270 vs 280) rather than introducing
    a new threshold.

    Verified (2026-08-27 calving resolution-consistency diagnosis):
    Open/DNB/Pregnant sojourn is mathematically UNAFFECTED by this change
    -- calving only governs the Dry-to-Open transition timing, with no
    path back into Open/DNB/Pregnant. Dry sojourn moves from ~82d
    (MONTH)/~55d (WEEK) -- a 49% MONTH-vs-WEEK disagreement -- to ~55d/
    ~55d (close agreement, matching the ~55-58 day literature benchmark).
    """
    dp = states['dp'].values
    return _calving_day_exact(dp)


def p_calving_formula(states, context):
    dp = states['dp'].values
    dp_days = dp * UNIT_DAYS
    midpoint, width = GESTATION_LENGTH_DAYS, 5
    return scale(1 / (1 + np.exp(-(dp_days - midpoint) / width)))


calving_spec = {
    'name': 'calving',
    'produces': 'p_calve',
    'depends_on': ['dp'],
    'method': 'table',
    'methods': {'table': p_calving_table, 'formula': p_calving_formula, 'model': p_calving_model},
}

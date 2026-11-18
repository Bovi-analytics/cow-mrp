"""
cow/probabilities/calving.py
=============================
Calving probability: p_calve (successful pregnancy completion and birth).
"""

import numpy as np
import pandas as pd
from cowmrp.domains.cow.parameters import scale, UNIT_DAYS, GESTATION_LENGTH_DAYS, GESTATION_LENGTH_SD_DAYS, MAX_DP
from cowmrp.domains.cow.probabilities.base import lookup, resolve_method, dp_bucket_calv, age_offset_slots


calving_table = pd.DataFrame([
    {'dp_bucket': '0-270',  'p': 0.0},
    {'dp_bucket': '271-290','p': 1.0},
])


def p_calving_table_legacy(states, context):
    """Earlier implementation of the 'table' method, not wired into
    calving_spec['methods']; kept for direct comparison with p_calving_table.
    It looks up ONE calving_table bucket for the whole model step, using a
    hardcoded 270-day cutoff that does not match GESTATION_LENGTH_DAYS (used
    everywhere else in this model, including p_calving_formula's own midpoint
    below and cow.parameters.DRY_OFF_DP_DAYS = GESTATION_LENGTH_DAYS -
    DRY_OFF_DAYS_BEFORE_CALVING). At MONTH resolution the coarse bucket
    lookup additionally delays the calving trigger to real day 300, which
    inflates the Dry-flag sojourn (~82 days at MONTH versus ~55 days at WEEK,
    against a literature benchmark of ~55-58 days). See also
    p_conception_table_legacy in conception.py."""
    dp = states['dp'].values
    return scale(lookup(calving_table, {'dp_bucket': dp_bucket_calv(dp)}))


def _calving_day_exact(dp_step):
    """Day-exact calving trigger. Calving is a deterministic event at real
    gestation day GESTATION_LENGTH_DAYS, not a compounding hazard, so unlike
    conception.py and abortion.py's survival-product integration it only
    needs to ask whether the step's own real-day span
    [dp_step*UNIT_DAYS, dp_step*UNIT_DAYS + UNIT_DAYS) contains day
    GESTATION_LENGTH_DAYS, rather than looking up which coarse
    (to_units-rounded) bucket the step's own index falls into. At DAY
    resolution (UNIT_DAYS=1) it triggers exactly at dp = GESTATION_LENGTH_DAYS
    with no discretization error; at WEEK it triggers in the 7-day window
    containing that day, and at MONTH in the 30-day window containing it
    (30 days earlier than the legacy table's day-300 trigger)."""
    starts = dp_step * UNIT_DAYS
    ends = starts + UNIT_DAYS - 1
    return np.where(ends >= GESTATION_LENGTH_DAYS, 1.0, 0.0)


def calving_step_mass(sd_days=0.0):
    """Fraction of a conception cohort that calves in pregnant step k, {k: mass}.

    Conception happens uniformly inside the step before dp=1, so at the first boundary the
    gestation age is u ~ Uniform(0, UNIT_DAYS) and step k spans ages (k-1)*U+u .. k*U+u. With sd_days = 0
    (the default 'table' method) a pregnancy calves in the step whose span contains GESTATION_LENGTH_DAYS,
    i.e. k = floor((G-u)/U) + 1: at DAY that is one step for everyone; at WEEK/MONTH the mass may spread
    across 2 adjacent steps depending on whether G divides U evenly (computed from
    GESTATION_LENGTH_DAYS, not hardcoded). With sd_days > 0 (the 'model' method) gestation
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
    """calving_step_mass() of the active calving method (used by cow/tables.py to size the state space)."""
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
    """Default 'table' method: the per-step calving probability that follows from the same
    uniform-conception-time convention as the abortion, conception and DNB functions (see
    calving_step_mass). It replaces the deterministic 'step containing the gestation day' trigger
    (p_calving_table_deterministic_legacy below): at MONTH that trigger fired at dp=9 for every cow, an
    effective gestation of 270 d once DIM 0 is read as 'calved somewhere in the step'; the averaged form
    puts 2/3 of calvings at dp 9 and 1/3 at dp 10 (mean gestation 280 d). Identical at WEEK and DAY."""
    return calving_step_probability(states['dp'].values)


def p_calving_model(states, context):
    """'model' method: gestation length ~ Normal(GESTATION_LENGTH_DAYS, GESTATION_LENGTH_SD_DAYS = 5.9 d)
    (Mwangi et al. 2025 for the SD; the mean is the model's own GESTATION_LENGTH_DAYS), with the same
    uniform-conception-time convention as the table. Pregnant cows calve over a ~+/-12 d spread instead of all on
    one day, so the dry period and the next lactation start vary. Select with calving_spec['method'] = 'model'
    (cow/tables.py widens the state space to the spread automatically)."""
    return calving_step_probability(states['dp'].values, GESTATION_LENGTH_SD_DAYS)


def p_calving_table_deterministic_legacy(states, context):
    """
    Earlier deterministic trigger, not wired into calving_spec; kept for comparison. It uses
    the day-exact trigger of _calving_day_exact instead of the coarse 270-day bucket lookup
    (p_calving_table_legacy above), with the same GESTATION_LENGTH_DAYS reference point as
    p_calving_formula's midpoint and cow.parameters.DRY_OFF_DP_DAYS's "55 days before calving"
    definition.

    The Open/DNB/Pregnant sojourn is unaffected by this trigger, since calving only governs the
    Dry-to-Open transition timing, with no path back into Open/DNB/Pregnant. The Dry sojourn is
    ~82 d at MONTH and ~55 d at WEEK with the bucket lookup (a 49 % disagreement) and ~55 d at
    both with this trigger, matching the ~55-58 day literature benchmark.
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

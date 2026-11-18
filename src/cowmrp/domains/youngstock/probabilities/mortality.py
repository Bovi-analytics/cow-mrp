"""
youngstock/probabilities/mortality.py
======================================
p_die: probability a DAIRY_F calf/heifer dies during a model step
(CALF and HEIFER stages, i.e. birth -> first breeding).

Every probability here is per MODEL STEP with the cow module's ONE time convention: an animal
born inside a step has a true age of k*UNIT_DAYS + u at boundary k (u uniform), so the per-step
probability is the ratio of the u-averaged source survival at the two boundaries
(cow/probabilities/base.mean_step_probability). Rows not in scope get 0.

Methods (choose in mortality_spec['method']):
  'table'   : ACTIVE. Three age windows (days) with window risks taken from
              the literature (see youngstock/parameters.py); constant hazard
              inside each window.
  'formula' : one constant daily hazard for the whole preweaning period
              (matches the 5.0% total but NOT the age-at-death shape) and the
              annual weaned rate afterwards. Simpler; kept for comparison.
  'model'   : None (no mechanistic calf-mortality model).
"""

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.domains.cow.lookup_tables import mean_step_probability, age_offset_slots
from cowmrp.domains.youngstock.parameters import (
    WEANING_AGE_DAYS, FIRST_BREEDING_AGE_DAYS, K_BREED as YP_K_BREED,
    PREWEANING_EARLY_WINDOW_END_DAYS, PREWEANING_MORTALITY_TOTAL,
    WEANED_HEIFER_ANNUAL_MORTALITY, _preweaning_window_risks,
)
from cowmrp.mrp.engine import validate_lookup_coverage

_d1, _d2 = _preweaning_window_risks()

# Window risk = probability of dying within the window (conditional on being
# alive at its start). Hazard inside a window is constant.
calf_mortality_table = pd.DataFrame([
    {'age_from_days': 0,
     'age_to_days': PREWEANING_EARLY_WINDOW_END_DAYS,
     'p': _d1},
    {'age_from_days': PREWEANING_EARLY_WINDOW_END_DAYS,
     'age_to_days': WEANING_AGE_DAYS,
     'p': _d2},
    {'age_from_days': WEANING_AGE_DAYS,
     'age_to_days': FIRST_BREEDING_AGE_DAYS,
     'p': 1 - (1 - WEANED_HEIFER_ANNUAL_MORTALITY) ** ((FIRST_BREEDING_AGE_DAYS - WEANING_AGE_DAYS) / 365.0)},
])

def _survival_from_windows(windows):
    """S(age_days) from windows of (from_day, to_day, window_risk): constant hazard inside each window,
    the last window extended past first-breeding age. Linear interpolation between whole days."""
    top = (YP_K_BREED + 3) * UNIT_DAYS + 2
    h = np.zeros(top + 1)
    windows = list(windows)
    for i, (lo, hi, p) in enumerate(windows):
        span = max(hi - lo, 1)                      # hazard uses the window's TRUE span...
        end = top + 1 if i == len(windows) - 1 else min(hi, top + 1)   # ...only the slice extends
        h[lo:end] = 1 - (1 - p) ** (1.0 / span)
    S = np.concatenate(([1.0], np.cumprod(1 - h)))
    grid = np.arange(len(S))
    return lambda age: np.interp(age, grid, S)


def _step_death_probability(states, survival_fn):
    """Per-step death probability for DAIRY_F CALF/HEIFER rows. The age at boundary k is the TRUE age
    k*UNIT_DAYS + u (u uniform: born inside the previous step), so the probability is the ratio of the
    u-averaged survival at boundaries k and k+1 (cow/probabilities/base.mean_step_probability)."""
    pk = mean_step_probability(survival_fn, YP_K_BREED + 1)
    age = np.clip(states['age'].values, 0, len(pk) - 1)
    in_scope = (states['line'].values == 'DAIRY_F') & np.isin(states['stage'].values, ['CALF', 'HEIFER'])
    return np.where(in_scope, pk[age], 0.0)


def p_mortality_table(states, context):
    validate_lookup_coverage(calf_mortality_table, ['age_from_days', 'age_to_days', 'p'], 'calf_heifer_mortality')
    windows = zip(calf_mortality_table['age_from_days'], calf_mortality_table['age_to_days'], calf_mortality_table['p'])
    return _step_death_probability(states, _survival_from_windows(windows))


def p_mortality_formula(states, context):
    windows = [
        (0, WEANING_AGE_DAYS, PREWEANING_MORTALITY_TOTAL),
        (WEANING_AGE_DAYS, FIRST_BREEDING_AGE_DAYS,
         1 - (1 - WEANED_HEIFER_ANNUAL_MORTALITY) ** ((FIRST_BREEDING_AGE_DAYS - WEANING_AGE_DAYS) / 365.0)),
    ]
    return _step_death_probability(states, _survival_from_windows(windows))


def birth_step_survival():
    """Share of a birth cohort still alive at the FIRST boundary: calves born inside a step are already
    u ~ Uniform(0, UNIT_DAYS) days old there, so Sbar(0) = E_u S(u) < 1 (about 0.98 at MONTH, 0.995 at
    WEEK, ~1 at DAY). The remaining deaths happen inside the birth step and are booked at birth by
    birth_outcomes (youngstock/probabilities/birth_outcomes.py) so no early mortality is lost at coarse steps."""
    windows = (list(zip(calf_mortality_table['age_from_days'], calf_mortality_table['age_to_days'], calf_mortality_table['p']))
               if mortality_spec['method'] == 'table' else
               [(0, WEANING_AGE_DAYS, PREWEANING_MORTALITY_TOTAL),
                (WEANING_AGE_DAYS, FIRST_BREEDING_AGE_DAYS,
                 1 - (1 - WEANED_HEIFER_ANNUAL_MORTALITY) ** ((FIRST_BREEDING_AGE_DAYS - WEANING_AGE_DAYS) / 365.0))])
    return float(np.mean(_survival_from_windows(windows)(age_offset_slots())))


mortality_spec = {
    'name': 'calf_heifer_mortality',
    'produces': 'p_die',
    'depends_on': ['line', 'stage', 'age'],
    'method': 'table',      # <-- change to 'formula' for the constant-hazard variant
    'methods': {'table': p_mortality_table, 'formula': p_mortality_formula, 'model': None},
}

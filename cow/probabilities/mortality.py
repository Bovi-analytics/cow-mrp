"""
cow/probabilities/mortality.py
===============================
Mortality probability: p_dead (cow dies this period, unrelated to culling).

Three methods, most-confident-usable as default (method='formula'):
- table:   the ORIGINAL uncited placeholder table (kept byte-unchanged,
           still selectable -- see mortality_table below).
- formula: literature-anchored LEVEL calibration, ACTIVE DEFAULT -- see
           p_mortality_anchor_calibrated and cow.parameters.
           MORTALITY_ANNUAL_ANCHOR for the full published-vs-our-adaptation
           breakdown. The table's relative shape (by lactation/DIM/status)
           is UNCHANGED; only its overall level is rescaled to match a
           literature-corroborated ~5.7%/year mortality rate. The prior
           uncited exponential-decay formula (p_mortality_formula) is kept
           unchanged below but no longer wired into mortality_spec.
- model:   None (unchanged).
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from cow.parameters import MAX_LACTATION, MAX_DIM, DRY_OFF_DP
from cow.lookup_tables import (
    N_DIM_MONTHS, STATUSES, full_grid, register_full_table, get_full_table, table_to_array,
    step_probability_matrix, status_index,
)
from cow.parameters import UNIT_DAYS, scale, MORTALITY_ANNUAL_ANCHOR
from cow.probabilities.base import lookup, resolve_method, lactation_bucket, status_bucket, dim_bucket_risk


mortality_table = pd.DataFrame([
    {'lactation_bucket': '1', 'dim_bucket': '0-60',  'status': 'Open',     'p': 0.006},
    {'lactation_bucket': '1', 'dim_bucket': '61-150', 'status': 'Open',     'p': 0.002},
    {'lactation_bucket': '1', 'dim_bucket': '151-300','status': 'Open',     'p': 0.003},
    {'lactation_bucket': '1', 'dim_bucket': '300+',   'status': 'Open',     'p': 0.005},
    {'lactation_bucket': '1', 'dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.001},
    {'lactation_bucket': '2', 'dim_bucket': '0-60',  'status': 'Open',     'p': 0.012},
    {'lactation_bucket': '2', 'dim_bucket': '61-150', 'status': 'Open',     'p': 0.003},
    {'lactation_bucket': '2', 'dim_bucket': '151-300','status': 'Open',     'p': 0.005},
    {'lactation_bucket': '2', 'dim_bucket': '300+',   'status': 'Open',     'p': 0.008},
    {'lactation_bucket': '2', 'dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.002},
    {'lactation_bucket': '3+','dim_bucket': '0-60',  'status': 'Open',     'p': 0.025},
    {'lactation_bucket': '3+','dim_bucket': '61-150', 'status': 'Open',     'p': 0.006},
    {'lactation_bucket': '3+','dim_bucket': '151-300','status': 'Open',     'p': 0.010},
    {'lactation_bucket': '3+','dim_bucket': '300+',   'status': 'Open',     'p': 0.016},
    {'lactation_bucket': '3+','dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.003},
])


def p_mortality_table(states, context):
    """Original uncited placeholder table -- UNCHANGED, still selectable via
    method='table'. No citation (see docs/LIMITATIONS_REGISTER.md L-006)."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    base = lookup(mortality_table, {
        'lactation_bucket': lactation_bucket(lac),
        'dim_bucket': dim_bucket_risk(dim),
        'status': status_bucket(dp),
    })
    return scale(base)


def _solve_mortality_calibration_scale():
    """Solve for k (see cow.parameters.MORTALITY_ANNUAL_ANCHOR
    ['calibration_method'] for the full derivation) such that scaling every
    mortality_table monthly rate by k makes their EQUAL-WEIGHTED mean
    annualized rate -- mean_i(1-(1-k*p_i)**12) -- equal
    MORTALITY_ANNUAL_ANCHOR['anchor_value']. Each per-cell term
    1-(1-k*p_i)**12 is monotonic increasing in k only while k*p_i <= 1 (past
    that, (1-k*p_i) goes negative and the even power turns the term sharply
    DECREASING, breaking monotonicity) -- so the bisection's upper bound is
    capped at just under 1/max(p_i) to guarantee it never leaves the safe,
    monotonic region. No scipy dependency needed. Solved once at import
    time."""
    p = mortality_table['p'].values.astype(float)
    target = MORTALITY_ANNUAL_ANCHOR['anchor_value']

    def mean_annual(k):
        return np.mean(1 - (1 - k * p) ** 12)

    lo, hi = 0.0, 0.999 / p.max()
    assert mean_annual(hi) > target, "anchor_value unreachable within the safe monotonic range"
    for _ in range(100):
        mid = (lo + hi) / 2
        if mean_annual(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


MORTALITY_CALIBRATION_SCALE = _solve_mortality_calibration_scale()


def p_mortality_anchor_calibrated(states, context):
    """ACTIVE DEFAULT. Same lookup/shape as p_mortality_table, rescaled by
    MORTALITY_CALIBRATION_SCALE so the table's equal-weighted mean annual
    rate matches the literature-corroborated ~5.7%/year anchor -- see
    cow.parameters.MORTALITY_ANNUAL_ANCHOR for the full published-vs-our-
    adaptation breakdown. The relative shape (by lactation/DIM/status) is
    UNCHANGED from the original uncited table; only the overall level
    moves."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    base = lookup(mortality_table, {
        'lactation_bucket': lactation_bucket(lac),
        'dim_bucket': dim_bucket_risk(dim),
        'status': status_bucket(dp),
    }) * MORTALITY_CALIBRATION_SCALE
    return scale(base)


def p_mortality_formula(states, context):
    """Original uncited exponential-decay hazard formula -- UNCHANGED, but
    no longer the active default (superseded in that role by
    p_mortality_anchor_calibrated above). Retained for reference/
    comparison; not wired into mortality_spec['methods'] any more since a
    spec has exactly one 'formula' slot -- call this function directly if
    you need it."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    dim_days = dim * UNIT_DAYS
    lac_factor = 1 + 1.6 * (lac - 1)
    transition_excess = 0.006 * np.exp(-dim_days / 40)
    chronic = 0.0008 * (np.exp(dim_days / 500) - 1)
    open_hazard = (0.0015 + transition_excess + chronic) * lac_factor
    pregnant_hazard = 0.001 * lac_factor
    daily = np.where(dp > 0, pregnant_hazard, open_hazard)
    return scale(1 - (1 - np.clip(daily, 0, 1)) ** UNIT_DAYS)


# ============================================================
# FULL LOOKUP TABLE (lactation 1..MAX_LACTATION x month x Open/Pregnant/Dry), TWO CLOCKS
# ============================================================
# ACTIVE table (audit P3, 2026-10-08): hazards read from Pinedo, De Vries & Webb (2010) Figs 4 and 6 for the
# disposal code "died": OPEN cows against days after calving (DIM clock), PREGNANT cows against days of
# pregnancy (DP clock; Dry = late gestation on the same clock), parity 1 / 2 / 3 / >=4. The '>=4' curve is spread
# over lactations 4..10 with the ratio of Miller et al. (2008) Table 2 death-per-lactation values (4.38 ... 6.40 %).
# Evidence class B (graph values read by eye; see cow/evidence/pinedo2010.py). Replaces the uncited placeholder
# buckets, whose lactation effect stopped at lactation 3 and which were scaled to one annual anchor.
# Override by placing cow/data/mortality_full.csv (same layout; Pregnant/Dry `month` = month of pregnancy).

from cow.evidence.miller2008 import death_relative_to_pinedo_parity4plus
from cow.probabilities.hazard_tables import pinedo_status_table, status_step_arrays, lookup_status_arrays


def _build_mortality_full():
    return pinedo_status_table(
        ("died",), lactation_factor=death_relative_to_pinedo_parity4plus(),
        factor_note="; lactations 4-10 = Pinedo '>=4' curve x Miller 2008 Table 2 ratio (adapted)")


register_full_table('mortality_full', _build_mortality_full, N_DIM_MONTHS, STATUSES)


@lru_cache(maxsize=1)
def _mortality_step_arrays():
    """(open[lac, dim], pregnant[lac, dp], dry[lac, dp]) per-step probabilities, one time-discretization rule."""
    return status_step_arrays(get_full_table('mortality_full'))


def p_mortality_full_table(states, context):
    """ACTIVE 'table' method: full lactation x month x status table (cow/data/mortality_full.csv overrides),
    Open on the DIM clock, Pregnant/Dry on the DP clock, per-step probabilities by cow/lookup_tables.py."""
    return lookup_status_arrays(_mortality_step_arrays(), states)


mortality_spec = {
    'name': 'mortality',
    'produces': 'p_dead',
    'depends_on': ['lactation', 'dim', 'dp'],
    'method': 'table',
    # table   : full lactation x month x status table, Open on the DIM clock / Pregnant+Dry on the DP clock (ACTIVE DEFAULT)
    # formula : previous uncited bucket lookup x calibration scale (kept for comparison)
    # model   : not available yet (e.g. a fitted hazard model) -- selecting it raises a clear error
    'methods': {'table': p_mortality_full_table, 'formula': p_mortality_anchor_calibrated, 'model': None},
}

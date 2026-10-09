"""
cow/probabilities/culling.py
=============================
Competing-risk culling probability. Two modules:
1. culling_age_stage -- age/lactation/DIM/status baseline risk (literature-anchored
                        residual level; see p_cull_age_stage_anchor_calibrated and
                        cow.parameters.CULLING_AGE_STAGE_ANNUAL_ANCHOR)
2. culling_repro     -- the milk-floor reproductive cull: an OPEN cow (dp == 0) whose
                        DIM has reached her lactation's cull DIM (cow/tables.py) is
                        replaced with probability 1. The cull DIM is precomputed from
                        the lactation curve and the farmer's MILK_FLOOR_KG, evaluated
                        only for DIM past LAST_BREEDING_DIM. Pregnant cows are never
                        checked.

Combine as: p_cull = 1 - (1 - p_age_stage) * (1 - p_repro)
(cow/probabilities/__init__.py, ENABLE_CULL_* switches in cow/parameters.py).

Health-driven and milk-level-policy culling were removed with the health / milk-bucket
state variables (CowMRP fork); the milk-floor cull replaces the milk cause. Total
involuntary culling is therefore LOWER than before -- a disclosed, intended change.
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from cow.parameters import MAX_LACTATION, MAX_DIM, DRY_OFF_DP
from cow.lookup_tables import (
    N_DIM_MONTHS, STATUSES, full_grid, register_full_table, get_full_table, table_to_array,
    step_probability_matrix, status_index,
)
from cow.parameters import UNIT_DAYS, scale, CULLING_AGE_STAGE_ANNUAL_ANCHOR
from cow.probabilities.base import lookup, resolve_method, lactation_bucket, status_bucket, dim_bucket_risk


culling_table = pd.DataFrame([
    {'lactation_bucket': '1', 'dim_bucket': '0-60',  'status': 'Open',     'p': 0.015},
    {'lactation_bucket': '1', 'dim_bucket': '61-150', 'status': 'Open',     'p': 0.010},
    {'lactation_bucket': '1', 'dim_bucket': '151-300','status': 'Open',     'p': 0.035},
    {'lactation_bucket': '1', 'dim_bucket': '300+',   'status': 'Open',     'p': 0.050},
    {'lactation_bucket': '1', 'dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.002},
    {'lactation_bucket': '2', 'dim_bucket': '0-60',  'status': 'Open',     'p': 0.025},
    {'lactation_bucket': '2', 'dim_bucket': '61-150', 'status': 'Open',     'p': 0.015},
    {'lactation_bucket': '2', 'dim_bucket': '151-300','status': 'Open',     'p': 0.050},
    {'lactation_bucket': '2', 'dim_bucket': '300+',   'status': 'Open',     'p': 0.075},
    {'lactation_bucket': '2', 'dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.004},
    {'lactation_bucket': '3+','dim_bucket': '0-60',  'status': 'Open',     'p': 0.050},
    {'lactation_bucket': '3+','dim_bucket': '61-150', 'status': 'Open',     'p': 0.030},
    {'lactation_bucket': '3+','dim_bucket': '151-300','status': 'Open',     'p': 0.085},
    {'lactation_bucket': '3+','dim_bucket': '300+',   'status': 'Open',     'p': 0.120},
    {'lactation_bucket': '3+','dim_bucket': 'ANY',    'status': 'Pregnant', 'p': 0.010},
])


# ============================================================
# MODULE 1: Age/Stage/Lactation culling
# ============================================================

def p_cull_age_stage_table(states, context):
    """Original uncited placeholder table -- UNCHANGED, still selectable via
    method='table'. No citation (see docs/LIMITATIONS_REGISTER.md L-006)."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    base = lookup(culling_table, {
        'lactation_bucket': lactation_bucket(lac),
        'dim_bucket': dim_bucket_risk(dim),
        'status': status_bucket(dp),
    })
    return scale(base)


def _solve_culling_age_stage_calibration_scale():
    """Solve for k (see cow.parameters.CULLING_AGE_STAGE_ANNUAL_ANCHOR
    ['calibration_method'] for the full derivation) such that scaling every
    culling_table monthly rate by k makes their EQUAL-WEIGHTED mean
    annualized rate -- mean_i(1-(1-k*p_i)**12) -- equal
    CULLING_AGE_STAGE_ANNUAL_ANCHOR['anchor_value']. Same bisection approach
    as cow/probabilities/mortality.py::_solve_mortality_calibration_scale,
    including the same safe-upper-bound cap (each per-cell term stops being
    monotonic once k*p_i > 1 -- culling_table's larger max p makes this
    bound bite much sooner than mortality's, so it must be computed
    dynamically here, not copied as a fixed constant)."""
    p = culling_table['p'].values.astype(float)
    target = CULLING_AGE_STAGE_ANNUAL_ANCHOR['anchor_value']

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


CULLING_AGE_STAGE_CALIBRATION_SCALE = _solve_culling_age_stage_calibration_scale()


def p_cull_age_stage_anchor_calibrated(states, context):
    """ACTIVE DEFAULT. Same lookup/shape as p_cull_age_stage_table, rescaled
    by CULLING_AGE_STAGE_CALIBRATION_SCALE so the table's equal-weighted
    mean annual rate matches a literature-derived RESIDUAL/unattributed-
    cause share (~6.47%/year) -- RE-SCOPED this pass from the prior full
    ~29.2%/year population-total anchor (see cow.parameters.
    CULLING_AGE_STAGE_ANNUAL_ANCHOR for the full published-vs-our-
    adaptation breakdown and docs/LIMITATIONS_REGISTER.md L-020 for why).
    'Age/stage' is kept as this module's name for continuity, but it no
    longer represents a biological age hazard -- it represents whatever
    disposal-code reasons other than the milk-floor reproductive cull
    (culling_repro; the former health / milk-policy / DNB modules were removed with their
    state variables). The relative shape (by lactation/DIM/status) is UNCHANGED
    from the original uncited table -- only the LEVEL changed."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    base = lookup(culling_table, {
        'lactation_bucket': lactation_bucket(lac),
        'dim_bucket': dim_bucket_risk(dim),
        'status': status_bucket(dp),
    }) * CULLING_AGE_STAGE_CALIBRATION_SCALE
    return scale(base)


def p_cull_age_stage_formula(states, context):
    """Original uncited exponential-decay hazard formula -- UNCHANGED, but
    no longer the active default (superseded in that role by
    p_cull_age_stage_anchor_calibrated above). Retained for reference/
    comparison; not wired into culling_age_stage_spec['methods'] any more
    since a spec has exactly one 'formula' slot -- call this function
    directly if you need it."""
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    dim_days = dim * UNIT_DAYS
    lac_factor = 1 + 1.6 * (lac - 1)
    transition_excess = 0.010 * np.exp(-dim_days / 40)
    chronic = 0.004 * (np.exp(dim_days / 400) - 1)
    open_hazard = (0.010 + transition_excess + chronic) * lac_factor
    pregnant_hazard = 0.002 * lac_factor
    daily = np.where(dp > 0, pregnant_hazard, open_hazard)
    return scale(1 - (1 - np.clip(daily, 0, 1)) ** UNIT_DAYS)


# ============================================================
# FULL LOOKUP TABLE (lactation 1..MAX_LACTATION x month x Open/Pregnant/Dry), TWO CLOCKS
# ============================================================
# ACTIVE table (audit P2/P3, 2026-10-08): "other-reason" involuntary culling = the sum of the Pinedo, De Vries &
# Webb (2010) Figs 4-7 hazards for feet and legs, injury/other, mastitis, disease and udder problems -- the pathways
# that disappeared when the health and milk-level states were removed, now represented as a STATE-INDEPENDENT hazard
# that depends only on (lactation, DIM for open cows | DP for pregnant cows). Low production and reproduction
# are deliberately excluded (the explicit milk-floor policy, culling_repro, represents them); "reason not
# reported" (5 % of removals) has no curve. Parity 1/2/3/>=4; the '>=4' curve repeats for lactations 4-10.
# Evidence class B (graph values read by eye; cow/evidence/pinedo2010.py). Replaces the uncited placeholder
# buckets scaled to a residual anchor. Override with cow/data/culling_age_stage_full.csv.

from cow.evidence.pinedo2010 import OTHER_CULL_CODES
from cow.probabilities.hazard_tables import pinedo_status_table, status_step_arrays, lookup_status_arrays


def _build_culling_age_stage_full():
    return pinedo_status_table(OTHER_CULL_CODES)


register_full_table('culling_age_stage_full', _build_culling_age_stage_full, N_DIM_MONTHS, STATUSES)


@lru_cache(maxsize=1)
def _culling_age_stage_step_arrays():
    """(open[lac, dim], pregnant[lac, dp], dry[lac, dp]) per-step probabilities, one time-discretization rule."""
    return status_step_arrays(get_full_table('culling_age_stage_full'))


def p_culling_age_stage_full_table(states, context):
    """ACTIVE 'table' method: full lactation x month x status table (cow/data/culling_age_stage_full.csv overrides),
    Open on the DIM clock, Pregnant/Dry on the DP clock, per-step probabilities by cow/lookup_tables.py."""
    return lookup_status_arrays(_culling_age_stage_step_arrays(), states)


culling_age_stage_spec = {
    'name': 'culling_age_stage',
    'produces': 'p_cull_age_stage',
    'depends_on': ['lactation', 'dim', 'dp'],
    'method': 'table',
    # table: full lactation x month x status table (ACTIVE DEFAULT) | formula: previous bucket x scale | model: n/a
    'methods': {'table': p_culling_age_stage_full_table, 'formula': p_cull_age_stage_anchor_calibrated, 'model': None},
}


# ============================================================
# MODULE 2: Milk-floor reproductive cull
# ============================================================

def p_cull_repro(states, context):
    """Milk-floor reproductive cull: for open cows (dp == 0), the precomputed probability that she is
    culled in this DIM step (cow.tables cull_p, spread around the floor-crossing age with the calving
    entry-offset rule, 1.0 from the last step on); 0 for pregnant/dry cows. Pure array lookup."""
    from cow.tables import get_tables, lactation_index
    from cow.parameters import MAX_DIM
    cull_p = get_tables()['cull_p']
    lac, dim, dp = states['lactation'].values, states['dim'].values, states['dp'].values
    return np.where(dp == 0, cull_p[lactation_index(lac), np.clip(dim, 0, MAX_DIM)], 0.0)


culling_repro_spec = {
    'name': 'culling_repro',
    'produces': 'p_cull_repro',
    'depends_on': ['lactation', 'dim', 'dp'],
    'method': 'formula',
    'methods': {'table': None, 'formula': p_cull_repro, 'model': None},
}

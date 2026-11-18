"""
cow/probabilities/base.py
=========================
Shared architecture for probability specs: helper functions, lookup tables,
bucket functions, health multipliers. Everything used by multiple probability
modules lives here.
"""

import numpy as np
import pandas as pd
from cowmrp.domains.cow.parameters import UNIT_DAYS, to_units, scale


def lookup(table, keys):
    """
    Generic wildcard table lookup.
    table: DataFrame with key columns + 'p' column
    keys: dict of {column: array_of_values}
    'ANY' in a table row matches regardless of that key.
    Most specific row (fewest 'ANY') wins.
    """
    from cowmrp.mrp.core import lookup as mrp_lookup
    return mrp_lookup(table, keys)


def resolve_method(spec):
    """Check that a spec's active method actually exists."""
    from cowmrp.mrp.engine import resolve_method as mrp_resolve
    return mrp_resolve(spec)


# ============================================================
# TIME-CONVENTION HELPER
# ============================================================
# ONE convention for every age-dependent per-step probability: an animal that
# ENTERS a phase (calving -> DIM 0, conception -> dp 1) does so somewhere INSIDE
# the transition step, so at the first step boundary its true age in that phase
# is uniform on (0, UNIT_DAYS). The state index k therefore stands for a true
# age of  k*UNIT_DAYS + u  (DIM steps) or  (k-1)*UNIT_DAYS + u  (gestation steps,
# whose dp=1 is the first boundary), u ~ Uniform(0, UNIT_DAYS) -- never for the
# exact age k*UNIT_DAYS. The per-step probability that keeps the chain's
# marginal survival equal to the source curve is the ratio of the AVERAGE
# survival at the step's two boundaries:  p_k = 1 - Sbar(k+1)/Sbar(k),
# Sbar(k) = E_u S(age_k + u).  At DAY resolution (U=1) this is the plain
# day-exact value; the coarser the step, the more it matters.

from cowmrp.domains.cow.lookup_tables import age_offset_slots, mean_step_probability  # noqa: F401  (re-exported)


from cowmrp.domains.cow.lookup_tables import threshold_crossing_mass, crossing_conditional  # noqa: F401  (re-exported)


# ============================================================
# BUCKET HELPERS
# ============================================================

def lactation_bucket(lac):
    return np.select([lac == 1, lac == 2], ['1', '2'], default='3+')

def lactation_bucket_breeding(lac):
    return np.where(lac == 1, '1', '2+')

def status_bucket(dp):
    """'Open' when dp == 0, else 'Pregnant' (dry cows count as Pregnant)."""
    return np.where(np.asarray(dp) == 0, 'Open', 'Pregnant')

# DIM bucket boundaries (in time units)
DIM_RISK_LOW = to_units(60)
DIM_RISK_MID = to_units(150)
DIM_RISK_HIGH = to_units(300)

DIM_BREEDING_LOW = to_units(80)
DIM_BREEDING_MID = to_units(150)
DIM_BREEDING_HIGH = to_units(250)

DIM_CONCEPTION_VWP = to_units(50)
DIM_CONCEPTION_MID = to_units(100)
DIM_CONCEPTION_HIGH = to_units(200)
DIM_CONCEPTION_VHIGH = to_units(350)

DP_ABORT_LOW = to_units(60)
DP_ABORT_MID = to_units(150)
DP_ABORT_HIGH = to_units(270)
DP_CALV_CUTOFF = to_units(270)


def dim_bucket_risk(dim):
    return np.select(
        [dim <= DIM_RISK_LOW, dim <= DIM_RISK_MID, dim <= DIM_RISK_HIGH],
        ['0-60', '61-150', '151-300'],
        default='300+'
    )

def dim_bucket_breeding(dim):
    return np.select(
        [dim <= DIM_BREEDING_LOW, dim <= DIM_BREEDING_MID, dim <= DIM_BREEDING_HIGH],
        ['0-80', '81-150', '151-250'],
        default='251-449'
    )

def dim_bucket_conception(dim):
    return np.select(
        [dim <= DIM_CONCEPTION_VWP, dim <= DIM_CONCEPTION_MID,
         dim <= DIM_CONCEPTION_HIGH, dim <= DIM_CONCEPTION_VHIGH],
        ['0-50', '51-100', '101-200', '201-350'],
        default='351-449'
    )

def dp_bucket_abort(dp):
    return np.select(
        [dp <= DP_ABORT_LOW, dp <= DP_ABORT_MID, dp <= DP_ABORT_HIGH],
        ['1-60', '61-150', '151-270'],
        default='271+'
    )

def dp_bucket_calv(dp):
    return np.select([dp <= DP_CALV_CUTOFF], ['0-270'], default='271-290')

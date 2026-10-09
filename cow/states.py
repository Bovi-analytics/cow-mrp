"""
cow/states.py
=============
State variable definitions.

The state is exactly three variables, all in model-step units (TIME_UNIT):
    lactation : 1 .. MAX_LACTATION
    dim       : days (steps) in milk, 0 .. dim_max
    dp        : days (steps) pregnant, 0 .. MAX_DP

Reproductive status is READ OFF dp, never stored: dp == 0 is open, 0 < dp <
DRY_OFF_DP pregnant, dp >= DRY_OFF_DP dry. "Do not breed" is not a state either:
it is the region of open cows past LAST_BREEDING_DIM (see cow/parameters.py).

dim_max comes from cow/tables.py (computed before any state exists): no cow can be
open past her lactation's cull DIM or pregnant past LAST_BREEDING_DIM + MAX_DP, so
the DIM axis stops there instead of at MAX_DIM_DAYS.

The MRP engine (mrp_engine.py) is "blind" to cows -- it just takes
variable_domains and derived_variables as inputs.
"""

import numpy as np
from cow.parameters import MAX_LACTATION, MAX_DP, DRY_OFF_DP
from cow.tables import get_tables, state_milk_kg


def repro_status(dp):
    """'Open' / 'Pregnant' / 'Dry' from dp (arrays or scalars)."""
    dp = np.asarray(dp)
    return np.select([dp == 0, dp < DRY_OFF_DP], ['Open', 'Pregnant'], default='Dry')


def milk_yield_kg(dim, lactation, dp):
    """Realized milk (kg/day): precomputed curve x pregnancy factor (0 once dry)."""
    return state_milk_kg(dim, lactation, dp)


# Independent variables: these define the whole state space
variable_domains = {
    'lactation': list(range(1, MAX_LACTATION + 1)),
    'dim': list(range(0, get_tables()['dim_max'] + 1)),
    'dp': list(range(0, get_tables()['dp_max'] + 1)),
}

# No derived variables: flag / milk / health no longer exist.
derived_variables = []

def build_cow_states():
    """The CowMRP state table: the Cartesian product of variable_domains minus states that can never be
    reached. For a pregnant cow dim - dp is constant (both advance by 1 per step) and equals the DIM step in
    which she conceived, so a pregnant state (dim, dp > 0) exists only if conception is possible in step
    dim - dp (cow.probabilities.conception.conception_entry_steps). Open states (dp == 0) are all kept.
    Behavior-preserving: V on the kept states is identical (verify_state_space.py checks it). The generic
    Engine build_states is untouched -- this is a Domain-level filter applied after it."""
    from mrp import engine as mrp_engine
    states = mrp_engine.build_states(variable_domains, derived_variables)
    entry = np.isin(states['dim'].values - states['dp'].values, get_tables()['entry_steps'])
    states = states[(states['dp'].values == 0) | entry].reset_index(drop=True)
    states['state_id'] = states.index
    return states


print(f"{len(variable_domains)} independent variables, {len(derived_variables)} derived variables declared.")

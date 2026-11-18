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

dim_max comes from domains.cow/tables.py (computed before any state exists): the last pregnant
DIM of a conception at STATE_DIM_CAP_DAYS, plus the re-opening step. The table holds
states no cow reaches under the default settings (conception after the last breeding
day has probability 0; open cows past the cull DIM leave with probability 1).

The MRP engine (mrp_engine.py) is "blind" to cows -- it just takes
variable_domains and derived_variables as inputs.
"""

import numpy as np
from cowmrp.domains.cow.parameters import MAX_LACTATION, MAX_DP, DRY_OFF_DP
from cowmrp.domains.cow.tables import get_tables, state_milk_kg


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
    """The CowMRP state table: the Cartesian product of variable_domains minus infeasible combinations. For a
    pregnant cow dim - dp is constant (both advance by 1 per step) and equals the DIM step in which she conceived;
    cow/state_constraints.py declares, as named rules with a rationale and a kind (biological / management /
    numerical), which combinations are possible (conception after calving, after the voluntary waiting period,
    before the last breeding day; the finite dp/dim axes). Open states (dp == 0) are
    all kept. The rules reproduce exactly the former filter on the conception steps with positive probability
    (verify_state_semantics.py checks both the equality and the probability closure). The generic Engine
    build_states is untouched -- this is a Domain-level filter applied after it."""
    from cowmrp.mrp import engine as mrp_engine
    from cowmrp.domains.cow.state_constraints import RuleParams, feasible_mask
    states = mrp_engine.build_states(variable_domains, derived_variables)
    t = get_tables()
    rules = RuleParams(dp_max=t['dp_max'], dim_max=t['dim_max'])
    ok = feasible_mask(states['dim'].values, states['dp'].values, rules, states['lactation'].values)
    states = states[ok].reset_index(drop=True)
    states['state_id'] = states.index
    return states


print(f"{len(variable_domains)} independent variables, {len(derived_variables)} derived variables declared.")

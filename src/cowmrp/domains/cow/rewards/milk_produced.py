"""
cow/rewards/milk_produced.py
============================
Tracked metric (non-monetary): milk produced in kg per step.
This rides along in every export but is NOT summed into total reward.
"""

import numpy as np
from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.domains.cow.states import milk_yield_kg


def milk_produced_formula(states, probs):
    """Actual realized milk yield (kg/day-equivalent for this time step)."""
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    milk_actual = milk_yield_kg(dim, lac, dp)
    return milk_actual * UNIT_DAYS


# This is a tracked metric, not a monetary reward. 
# It gets added to rewards dict but NOT included in the 'total' sum.
milk_produced_spec = {
    'name': 'milk_produced_kg',
    'depends_on': {
        'state': ['dim', 'lactation', 'dp'],
        'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': milk_produced_formula, 'model': None},
}

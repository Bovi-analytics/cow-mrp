"""cow/rewards/mortality_cost.py - disposal cost for dead cows."""
import numpy as np
from cow.rewards.base import DISPOSAL_COST

def mortality_cost_formula(states, probs):
    return probs['p_dead'] * DISPOSAL_COST

mortality_cost_spec = {
    'name': 'mortality_cost', 'sign': -1,
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': ['p_dead'],
        'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': mortality_cost_formula, 'model': None},
}

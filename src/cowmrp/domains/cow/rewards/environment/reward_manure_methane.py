"""
cow/rewards/environment/reward_manure_methane.py
===================================================
$ cost from manure methane. sign=-1, PRICE_PER_KG_CH4 defaults to 0.0
(shared with reward_enteric_methane.py -- both price the same gas, just
from different biological pathways; see reward_nitrogen.py for the full
pricing rationale). Only included when ENABLE_REWARD_MANURE_METHANE=True.

Manure CH4 itself (the bio quantity, computed in
cow/rewards/environment/manure_methane.py) is only present when
COW_MANURE_SYSTEM/COW_MANURE_CLIMATE are configured -- if this reward is
enabled without that configuration, validate_model_completeness raises a
clear MissingDependencyError rather than silently producing a zero
reward, so a misconfiguration is never mistaken for "no manure methane".
"""

from cowmrp.domains.cow.parameters import PRICE_PER_KG_CH4


def reward_manure_methane_formula(states, context):
    manure_ch4_kg = context['manure_ch4_kg']
    return manure_ch4_kg * PRICE_PER_KG_CH4


reward_manure_methane_spec = {
    'name': 'reward_manure_methane', 'sign': -1,
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['PRICE_PER_KG_CH4'], 'external_input': [],
        'model_output': ['manure_ch4_kg'], 'time': [],
    },
    'method': 'formula', 'methods': {'table': None, 'formula': reward_manure_methane_formula, 'model': None},
}

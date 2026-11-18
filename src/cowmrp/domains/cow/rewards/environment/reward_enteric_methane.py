"""
cow/rewards/environment/reward_enteric_methane.py
====================================================
$ cost from enteric methane. sign=-1, PRICE_PER_KG_CH4 defaults to 0.0
(see reward_nitrogen.py for the full rationale -- same pattern here).
Only included when ENABLE_REWARD_ENTERIC_METHANE=True.
"""

from cowmrp.domains.cow.parameters import PRICE_PER_KG_CH4


def reward_enteric_methane_formula(states, context):
    enteric_ch4_g = context['enteric_ch4_g']
    enteric_ch4_kg = enteric_ch4_g / 1000.0
    return enteric_ch4_kg * PRICE_PER_KG_CH4


reward_enteric_methane_spec = {
    'name': 'reward_enteric_methane', 'sign': -1,
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['PRICE_PER_KG_CH4'], 'external_input': [],
        'model_output': ['enteric_ch4_g'], 'time': [],
    },
    'method': 'formula', 'methods': {'table': None, 'formula': reward_enteric_methane_formula, 'model': None},
}

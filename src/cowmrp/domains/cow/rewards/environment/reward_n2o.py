"""
cow/rewards/environment/reward_n2o.py
========================================
$ cost from total N2O (direct + indirect volatilization + indirect
leaching -- already summed as n2o_total_kg by compute_environment()).
sign=-1, PRICE_PER_KG_N2O defaults to 0.0 (see reward_nitrogen.py for
the full pricing rationale). Only included when ENABLE_REWARD_N2O=True.

Prices the TOTAL only, not each pathway separately, to avoid triple-
pricing the same underlying manure-N source through three near-identical
reward terms.
The three pathways remain independently inspectable as raw quantities
(n2o_direct_kg, n2o_indirect_volatilization_kg, n2o_indirect_leaching_kg)
in the CSV even though only their sum is priced.
"""

from cowmrp.domains.cow.parameters import PRICE_PER_KG_N2O


def reward_n2o_formula(states, context):
    n2o_total_kg = context['n2o_total_kg']
    return n2o_total_kg * PRICE_PER_KG_N2O


reward_n2o_spec = {
    'name': 'reward_n2o', 'sign': -1,
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['PRICE_PER_KG_N2O'], 'external_input': [],
        'model_output': ['n2o_total_kg'], 'time': [],
    },
    'method': 'formula', 'methods': {'table': None, 'formula': reward_n2o_formula, 'model': None},
}

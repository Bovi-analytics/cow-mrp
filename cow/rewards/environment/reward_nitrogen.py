"""
cow/rewards/environment/reward_nitrogen.py
=============================================
$ cost from manure nitrogen. sign=-1 (cost), like every other cost
reward in this repo (feed_cost, health_cost, ...): higher emissions ->
lower reward.

PRICE_PER_KG_N_MANURE defaults to 0.0 in cow/parameters.py -- an
explicit, user-set economic assumption (a nutrient/carbon-price-style
policy choice, not a biological constant). With the default price, this
reward is always 0 even when ENABLE_REWARD_NITROGEN=True; you must set a
price intentionally for it to have any effect. No arbitrary
normalization constant is assumed on your behalf (see the scientific
audit's normalization discussion).

Only enabled/included in reward_specs (see cow/rewards/__init__.py) when
ENABLE_REWARD_NITROGEN=True -- default OFF, so reward_total is unchanged
unless you opt in.
"""

from cow.parameters import PRICE_PER_KG_N_MANURE


def reward_nitrogen_formula(states, context):
    manure_n_g = context['manure_n_g']
    manure_n_kg = manure_n_g / 1000.0
    return manure_n_kg * PRICE_PER_KG_N_MANURE


reward_nitrogen_spec = {
    'name': 'reward_nitrogen', 'sign': -1,
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['PRICE_PER_KG_N_MANURE'], 'external_input': [],
        'model_output': ['manure_n_g'], 'time': [],
    },
    'method': 'formula', 'methods': {'table': None, 'formula': reward_nitrogen_formula, 'model': None},
}

"""
cow/rewards/cull_reward.py
===========================
Salvage value from culling.

F3 fix (2026-08, see docs/REWARD_ECONOMICS_REVIEW_2026-08.md):
BEEF_PRICE_PER_KG (cow/rewards/base.py) is a DRESSED-basis cull-cow
price (USDA AMS Livestock Market News quotes cull cows this way, plant-
delivered/dressed). The prior formula applied this dressed-basis price
directly to LIVE body weight (bw) with no conversion -- a confirmed
~2.15x (1/0.463) overstatement of salvage value. Fixed by converting
live weight to a dressed-weight-equivalent via DRESSING_PCT (~46.3%,
the typical dairy cull-cow dressing percentage) before pricing.
"""
import numpy as np
from cow.parameters import UNIT_DAYS
from cow.rewards.base import max_lactation_exit_probability, body_weight_kg, age_days_of, mid_step_dim_days, BEEF_PRICE_PER_KG, CULLING_COST, DRESSING_PCT

def cull_reward_formula(states, probs):
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    age_days = age_days_of(lac, dim)
    bw = body_weight_kg(age_days, mid_step_dim_days(dim), dp * UNIT_DAYS)
    p_dead = probs['p_dead']
    p_cull = probs['p_cull']
    # culled cows plus cows leaving by calving at the maximum modelled lactation (sold, then replaced)
    p_cull_specifically = (1 - p_dead) * p_cull + max_lactation_exit_probability(states, probs)
    salvage_value = bw * DRESSING_PCT * BEEF_PRICE_PER_KG
    return p_cull_specifically * (salvage_value - CULLING_COST)

cull_reward_spec = {
    'name': 'cull_reward', 'sign': +1,
    'depends_on': {
        'state': ['dim', 'lactation', 'dp'],
        'derived_state': [], 'probability_output': ['p_dead', 'p_cull', 'p_abort', 'p_calve'], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': cull_reward_formula, 'model': None},
}

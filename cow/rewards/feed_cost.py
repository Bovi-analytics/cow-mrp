"""cow/rewards/feed_cost.py - feed (DMI x price) + dry period cost."""
import numpy as np
import pandas as pd
from cow.parameters import UNIT_DAYS, DRY_OFF_DP
from cow.states import milk_yield_kg
from cow.rewards.base import dry_matter_intake_kg, body_weight_kg, age_days_of, mid_step_dim_days, FEED_PRICE_PER_KG_DM, DRY_PERIOD_DAILY_COST
from mrp.core import lookup

feed_cost_table_df = pd.DataFrame([
    {'bw_bucket': 'low',    'cost_per_day': 4.5},
    {'bw_bucket': 'medium', 'cost_per_day': 5.5},
    {'bw_bucket': 'high',   'cost_per_day': 6.5},
])

def bw_bucket(bw):
    return np.select([bw < 600, bw < 680], ['low', 'medium'], default='high')

def feed_cost_table(states, probs):
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    dim_days = mid_step_dim_days(dim)
    age_days = age_days_of(lac, dim)
    bw = body_weight_kg(age_days, dim_days, dp * UNIT_DAYS)
    cost_per_day = lookup(feed_cost_table_df, {'bw_bucket': bw_bucket(bw)})
    return cost_per_day * UNIT_DAYS

def feed_cost_formula(states, probs):
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    dim_days = mid_step_dim_days(dim)
    age_days = age_days_of(lac, dim)
    milk_actual = milk_yield_kg(dim, lac, dp)
    bw = body_weight_kg(age_days, dim_days, dp * UNIT_DAYS)
    dmi = dry_matter_intake_kg(milk_actual, bw, np.maximum(dim_days, 1))
    cost = dmi * FEED_PRICE_PER_KG_DM * UNIT_DAYS
    return cost + np.where(dp >= DRY_OFF_DP, DRY_PERIOD_DAILY_COST * UNIT_DAYS, 0.0)

feed_cost_spec = {
    'name': 'feed_cost', 'sign': -1,
    'depends_on': {
        'state': ['dim', 'lactation', 'dp'],
        'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': feed_cost_table, 'formula': feed_cost_formula, 'model': None},
}

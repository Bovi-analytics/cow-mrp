"""
cow/rewards/milk_income.py
==========================
Milk production revenue (table/formula/model-swappable).
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from cow.parameters import MAX_DIM, MAX_LACTATION, DRY_OFF_DP
from cow.lookup_tables import (
    N_DIM_MONTHS, STATUSES, full_grid, register_full_table, get_full_table, table_to_array,
    step_mean_matrix, status_index,
)
from cow.parameters import UNIT_DAYS
from cow.states import milk_yield_kg
from cow.rewards.base import MILK_PRICE_PER_KG
from mrp.core import lookup
from mrp.engine import resolve_method


milk_income_table_df = pd.DataFrame([
    {'lactation_bucket': '1', 'dim_bucket': '0-60',   'income_per_day': 9.9},
    {'lactation_bucket': '1', 'dim_bucket': '61-150', 'income_per_day': 11.7},
    {'lactation_bucket': '1', 'dim_bucket': '151-300','income_per_day': 9.0},
    {'lactation_bucket': '1', 'dim_bucket': '300+',   'income_per_day': 6.3},
    {'lactation_bucket': '2', 'dim_bucket': '0-60',   'income_per_day': 12.2},
    {'lactation_bucket': '2', 'dim_bucket': '61-150', 'income_per_day': 14.4},
    {'lactation_bucket': '2', 'dim_bucket': '151-300','income_per_day': 11.3},
    {'lactation_bucket': '2', 'dim_bucket': '300+',   'income_per_day': 7.7},
    {'lactation_bucket': '3+', 'dim_bucket': '0-60',  'income_per_day': 13.5},
    {'lactation_bucket': '3+', 'dim_bucket': '61-150', 'income_per_day': 16.2},
    {'lactation_bucket': '3+', 'dim_bucket': '151-300','income_per_day': 12.6},
    {'lactation_bucket': '3+', 'dim_bucket': '300+',  'income_per_day': 8.6},
])

def milk_lactation_bucket(lac):
    return np.select([lac == 1, lac == 2], ['1', '2'], default='3+')

def milk_dim_bucket(dim):
    from cow.parameters import to_units
    low, mid, high = to_units(60), to_units(150), to_units(300)
    return np.select([dim <= low, dim <= mid, dim <= high], ['0-60', '61-150', '151-300'], default='300+')

def milk_income_table(states, probs):
    lac = states['lactation'].values
    dim = states['dim'].values
    income_per_day = lookup(milk_income_table_df, {
        'lactation_bucket': milk_lactation_bucket(lac),
        'dim_bucket': milk_dim_bucket(dim),
    })
    return income_per_day * UNIT_DAYS

def milk_income_formula(states, probs):
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    milk_actual = milk_yield_kg(dim, lac, dp)
    return milk_actual * MILK_PRICE_PER_KG * UNIT_DAYS

# ============================================================
# FULL LOOKUP TABLE (lactation 1..MAX_LACTATION x every native month x Open/Pregnant/Dry)
# ============================================================
# DAILY milk income ($/day) by lactation, month in milk and status. Default values = the
# uncited placeholder bucket table above (milk_income_table_df) replicated across lactations
# and months; Dry rows are 0 (no milk). Replace with real per-lactation values by placing
# cow/data/milk_income_full.csv. Time discretization: the mean daily value over the real days
# of the step, times UNIT_DAYS (cow/lookup_tables.step_mean_matrix).

def _build_milk_income_full():
    g = full_grid(N_DIM_MONTHS, STATUSES)
    day_mid = g['month'].values * 30 + 15
    lac = g['lactation'].values
    inc = lookup(milk_income_table_df.rename(columns={'income_per_day': 'p'}), {
        'lactation_bucket': np.select([lac == 1, lac == 2], ['1', '2'], default='3+'),
        'dim_bucket': np.select([day_mid <= 60, day_mid <= 150, day_mid <= 300],
                                ['0-60', '61-150', '151-300'], default='300+'),
    })
    g['income_per_day'] = np.where(g['status'].values == 'Dry', 0.0, inc)
    g['basis'] = 'placeholder'
    g['note'] = 'uncited placeholder bucket value copied across lactations/months; Dry rows = 0 (no milk). The ACTIVE milk-income method is the Wood/MilkBot formula; this table is a template.'
    return g


register_full_table('milk_income_full', _build_milk_income_full, N_DIM_MONTHS, STATUSES, value_col='income_per_day')


@lru_cache(maxsize=1)
def _milk_income_step_arrays():
    V = table_to_array(get_full_table('milk_income_full'), N_DIM_MONTHS, STATUSES, value_col='income_per_day')
    return np.stack([step_mean_matrix(V[s], MAX_DIM) for s in range(len(STATUSES))]) * UNIT_DAYS


def milk_income_full_table(states, probs):
    """'table' method: full lactation x month x status table -> $ per step."""
    lac = np.clip(states['lactation'].values, 1, MAX_LACTATION) - 1
    dim = np.clip(states['dim'].values, 0, MAX_DIM)
    s = status_index(states['dp'].values, DRY_OFF_DP)
    return _milk_income_step_arrays()[s, lac, dim]


milk_income_spec = {
    'name': 'milk_income',
    'sign': +1,
    'depends_on': {
        'state': ['dim', 'lactation', 'dp'],
        'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': milk_income_full_table, 'formula': milk_income_formula, 'model': None},
}

"""
cow/rewards/replacement_cost.py
================================
Cost of replacing a dead or culled cow.

PURCHASE_COST and REARING_COST (cow/rewards/base.py) are alternative
replacement pathways (buy vs. raise), not additive -- a farm pays one or
the other for a given replacement, never both. REPLACEMENT_STRATEGY
(cow/rewards/base.py) selects exactly one; "rear" is the only
implemented value. "buy" raises NotImplementedError naming what's
missing, the same convention as mrp_engine.resolve_method /
cow.parameters.WOOD1967's uncalibrated-parameter guard, rather than
silently falling back to the old (confirmed double-counting) additive
behavior. See docs/REWARD_ECONOMICS_REVIEW_2026-08.md for history.
"""
import numpy as np
from cow.rewards.base import PURCHASE_COST, REARING_COST, REPLACEMENT_STRATEGY, max_lactation_exit_probability

def replacement_cost_formula(states, probs):
    p_dead = probs['p_dead']
    p_cull = probs['p_cull']
    # death + culling + the forced exit of a cow that calves at the maximum modelled lactation (same replacement flow as the transition)
    p_replacement = p_dead + (1 - p_dead) * p_cull + max_lactation_exit_probability(states, probs)

    if REPLACEMENT_STRATEGY == 'rear':
        cost_per_replacement = REARING_COST
    else:
        raise NotImplementedError(
            f"REPLACEMENT_STRATEGY={REPLACEMENT_STRATEGY!r} is not "
            "implemented -- only 'rear' is currently supported. 'buy' "
            "(charge PURCHASE_COST alone) is a documented future "
            "extension point; the prior additive behavior (PURCHASE_COST "
            "+ REARING_COST) is not offered, as it was found to likely "
            "double-count -- see docs/REWARD_ECONOMICS_REVIEW_2026-08.md."
        )

    return p_replacement * cost_per_replacement

replacement_cost_spec = {
    'name': 'replacement_cost', 'sign': -1,
    'depends_on': {
        'state': ['lactation'], 'derived_state': [], 'probability_output': ['p_dead', 'p_cull', 'p_abort', 'p_calve'],
        'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': replacement_cost_formula, 'model': None},
}

"""
cow/rewards/replacement_cost.py
================================
Cost of replacing a dead or culled cow.

Not part of the state reward vector. `replacement_cost_spec` is no longer in
cow/rewards/__init__.py's BASE_REWARD_SPECS: `rewards['total']` and the reward vector exclude it, because the
transition matrix already routes every exit (death, culling, calving at the last lactation) to the replacement
state, and the cost of obtaining the replacement is charged where the cow and a replacement heifer are COMPARED
(cow/value_comparisons.py::replacement_comparison), over the same horizon. `replacement_cost_vector()` below gives
the per-state expected cost (probability of exit x REARING_COST) for that comparison; it is never folded into a
state reward.

PURCHASE_COST and REARING_COST (cow/rewards/base.py) are alternative
replacement pathways (buy vs. raise), not additive -- a farm pays one or
the other for a given replacement, never both. REPLACEMENT_STRATEGY
(cow/rewards/base.py) selects exactly one; "rear" is the only
implemented value. "buy" raises NotImplementedError naming what's
missing, the same convention as mrp_engine.resolve_method /
cow.parameters.WOOD1967's uncalibrated-parameter guard, rather than
silently falling back to an additive behaviour (both costs summed),
which would count the replacement twice.
"""
import numpy as np
from cowmrp.domains.cow.rewards.base import PURCHASE_COST, REARING_COST, REPLACEMENT_STRATEGY, max_lactation_exit_probability

def cost_per_replacement():
    """Cost of ONE replacement under REPLACEMENT_STRATEGY (cow/rewards/base.py). Only 'rear' is implemented."""
    if REPLACEMENT_STRATEGY == 'rear':
        return REARING_COST
    raise NotImplementedError(
        f"REPLACEMENT_STRATEGY={REPLACEMENT_STRATEGY!r} is not "
        "implemented -- only 'rear' is currently supported. 'buy' "
        "(charge PURCHASE_COST alone) is a possible future extension; "
        "summing PURCHASE_COST and REARING_COST is not offered, as it "
        "would count the replacement twice."
    )


def replacement_cost_formula(states, probs):
    p_dead = probs['p_dead']
    p_cull = probs['p_cull']
    # death + culling + the forced exit of a cow that calves at the maximum modelled lactation (same replacement flow as the transition)
    p_replacement = p_dead + (1 - p_dead) * p_cull + max_lactation_exit_probability(states, probs)
    return p_replacement * cost_per_replacement()

def replacement_cost_vector(states, probs):
    """Expected replacement cost per state and step (>= 0): P(exit to the replacement state) x cost of one replacement.
    Same event definition as the transition matrix (cow/transitions.py). Used only by the separate replacement comparison."""
    return np.asarray(replacement_cost_formula(states, probs), dtype=float)


replacement_cost_spec = {
    'name': 'replacement_cost', 'sign': -1,
    'depends_on': {
        'state': ['lactation'], 'derived_state': [], 'probability_output': ['p_dead', 'p_cull', 'p_abort', 'p_calve'],
        'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': replacement_cost_formula, 'model': None},
}

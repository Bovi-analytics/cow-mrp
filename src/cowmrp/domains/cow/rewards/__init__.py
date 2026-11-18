"""
cow/rewards/__init__.py
=======================
Reward orchestrator: imports all reward specs, sums them, returns reward vector.

STATE REWARD VECTOR:
    total = milk_income + pregnancy_reward + cull_reward - mortality_cost - feed_cost - reproduction_cost
    (+ any environmental reward whose ENABLE_REWARD_* flag is True)
`replacement_cost` is NOT part of it: exits are modelled by the transition matrix (every death, cull and calving at
the last lactation moves the cow to the replacement state) and the cost of the replacement is charged in the
separate cow-versus-replacement-heifer comparison (cow/value_comparisons.py::replacement_comparison). The
probability-weighted event components (cull_reward, mortality_cost) use exactly the exit probabilities of the
transition matrix: mortality p_dead; cull (1-p_dead)*p_cull + calving at the last lactation.

To add a new (non-environmental) reward component:
1. Create cow/rewards/your_reward.py with a spec
2. Import it here
3. Add to BASE_REWARD_SPECS
4. That's it — compute_rewards() handles the rest

To add a new environmental reward component, see
cow/rewards/environment/__init__.py's module docstring instead.

To swap methods (table/formula/model):
Edit the 'method' field in each spec in its .py file.
"""

import numpy as np

from cowmrp.domains.cow import parameters as cow_parameters
from cowmrp.domains.cow.parameters import (
    ENABLE_REWARD_NITROGEN, ENABLE_REWARD_ENTERIC_METHANE,
    ENABLE_REWARD_MANURE_METHANE, ENABLE_REWARD_N2O,
    MANURE_SYSTEM, MANURE_CLIMATE,
)
from cowmrp.mrp.engine import run_specs, validate_model_completeness

# Import all reward modules
from cowmrp.domains.cow.rewards.milk_income import milk_income_spec
from cowmrp.domains.cow.rewards.pregnancy_reward import pregnancy_reward_spec
from cowmrp.domains.cow.rewards.cull_reward import cull_reward_spec
from cowmrp.domains.cow.rewards.mortality_cost import mortality_cost_spec
from cowmrp.domains.cow.rewards.replacement_cost import replacement_cost_vector   # memo only, never in BASE_REWARD_SPECS
from cowmrp.domains.cow.rewards.feed_cost import feed_cost_spec
from cowmrp.domains.cow.rewards.reproduction_cost import reproduction_cost_spec
from cowmrp.domains.cow.rewards.milk_produced import milk_produced_spec

# Environmental rewards. Biological quantities (NI, CH4, CO2, N2O, ...)
# are ALWAYS computed by compute_environment() and always CSV-observable,
# exactly like milk_produced_kg. The reward specs below are likewise
# ALWAYS computed and exposed as separately-named outputs (never hidden
# inside another reward) -- see compute_rewards() -- but are folded into
# 'total' only when their ENABLE_REWARD_* flag is True (all default
# False), so reward_total's VALUE is unaffected unless a user both
# enables a component AND sets a nonzero PRICE_PER_KG_* (default 0.0).
# reward_manure_methane/reward_n2o additionally require MANURE_SYSTEM/
# MANURE_CLIMATE to be configured -- their biological inputs are simply
# undefined otherwise, so (like the bio quantities themselves) they are
# absent from the output rather than silently defaulted when
# unconfigured. No reward_co2 exists -- see cow/rewards/environment/co2.py
# for why (IPCC/EPA GHG-accounting convention).
from cowmrp.domains.cow.rewards.environment import compute_environment
from cowmrp.domains.cow.rewards.environment.reward_nitrogen import reward_nitrogen_spec
from cowmrp.domains.cow.rewards.environment.reward_enteric_methane import reward_enteric_methane_spec
from cowmrp.domains.cow.rewards.environment.reward_manure_methane import reward_manure_methane_spec
from cowmrp.domains.cow.rewards.environment.reward_n2o import reward_n2o_spec


BASE_REWARD_SPECS = [
    milk_income_spec,
    pregnancy_reward_spec,
    cull_reward_spec,
    mortality_cost_spec,
    feed_cost_spec,
    reproduction_cost_spec,
]

_MANURE_CONFIGURED = MANURE_SYSTEM is not None and MANURE_CLIMATE is not None

# Always computable (nitrogen/enteric methane never depend on manure
# config); manure methane/N2O only computable -- and therefore only
# ever produced/exposed -- when manure config is set.
ENVIRONMENTAL_REWARD_SPECS = [reward_nitrogen_spec, reward_enteric_methane_spec]
if _MANURE_CONFIGURED:
    ENVIRONMENTAL_REWARD_SPECS += [reward_manure_methane_spec, reward_n2o_spec]

_ENV_REWARD_ENABLED = {
    'reward_nitrogen': ENABLE_REWARD_NITROGEN,
    'reward_enteric_methane': ENABLE_REWARD_ENTERIC_METHANE,
    'reward_manure_methane': ENABLE_REWARD_MANURE_METHANE,
    'reward_n2o': ENABLE_REWARD_N2O,
}

# Full spec list, kept for documentation/validation purposes (matches
# the pre-existing "add to reward_specs list" convention referenced
# elsewhere) -- compute_rewards() below computes BASE_REWARD_SPECS and
# ENVIRONMENTAL_REWARD_SPECS separately so environmental rewards can be
# exposed unconditionally while only conditionally counted in 'total'.
reward_specs = BASE_REWARD_SPECS + ENVIRONMENTAL_REWARD_SPECS


def _available_for_rewards(states, probs, bio):
    """
    Build the Engine's 'available' dict for validating reward specs.
    probability_output uses the actual computed `probs` dict keys --
    the real runtime availability, including the Domain-composed 'p_cull'
    -- rather than a separately maintained static list. model_output is
    the environmental bio-quantity dict's keys, populated the same way.
    """
    return {
        "state": set(states.columns),
        "derived_state": set(states.columns),
        "probability_output": set(probs.keys()),
        "reward_output": {spec['name'] for spec in reward_specs}
                          | {milk_produced_spec['name'], 'reward_environment_total'},
        "parameter": {n for n in dir(cow_parameters) if n.isupper()},
        "external_input": set(),
        "model_output": set(bio.keys()),
        "time": {"time_unit"},
    }


def compute_rewards(states, probs):
    """
    Compute all reward components and total reward vector.

    Also computes: tracked_metric milk_produced_kg (not in total); the
    environmental biological quantities (never in total, see
    compute_environment()); and every environmental reward component,
    ALWAYS as a separately-named output (reward_nitrogen,
    reward_enteric_methane, and -- only when manure config is set --
    reward_manure_methane, reward_n2o), each folded into 'total' only if
    its ENABLE_REWARD_* flag is True. reward_environment_total is a pure
    aggregation of whichever environmental components are currently
    folded into 'total' -- not an independently-priced quantity, so it
    can never double-count against the individual components or 'total'
    itself.
    """
    # Biological calculation layer (NI, N excretion, CH4, CO2, N2O, ...)
    # -- physical quantities, never reward points. Always computed
    # (subject to manure CH4/N2O being skipped without manure config --
    # see cow/rewards/environment/__init__.py). This is the sole source
    # of truth for environmental equations; nothing below duplicates them.
    bio = compute_environment(states, probs)
    context = {**probs, **bio}

    # Pre-flight: every declared reward spec's dependencies exist and its
    # selected method resolves, before any numeric computation runs.
    validate_model_completeness(
        states, probability_specs=[], reward_specs=reward_specs + [milk_produced_spec],
        available=_available_for_rewards(states, probs, bio),
    )

    # Base economic/biological rewards -> 'total'.
    results = run_specs(BASE_REWARD_SPECS, states, context=context, has_sign=True)

    # Environmental rewards: always compute + expose as separately-named
    # outputs; fold into 'total' (and the derived environment-only
    # subtotal) only for currently-enabled components.
    env_total = np.zeros(len(states))
    for spec in ENVIRONMENTAL_REWARD_SPECS:
        value = run_specs([spec], states, context=context, has_sign=False)[spec['name']]
        results[spec['name']] = value
        if _ENV_REWARD_ENABLED[spec['name']]:
            results['total'] = results['total'] + spec['sign'] * value
            env_total = env_total + spec['sign'] * value
    results['reward_environment_total'] = env_total

    # Add tracked metric (NOT part of total reward sum)
    results['milk_produced_kg'] = run_specs(
        [milk_produced_spec], states, context=context
    )['milk_produced_kg']

    # Environmental biological quantities ride along in every export,
    # exactly like milk_produced_kg -- CSV-observable, NOT part of
    # reward_total (only the priced reward_* components above are).
    results.update(bio)

    return results

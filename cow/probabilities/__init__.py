"""
cow/probabilities/__init__.py
==============================
Probability orchestrator: imports all specs, combines them, computes full
probability vector for all states.

To add a new probability module:
1. Create cow/probabilities/your_module.py with a spec
2. Import it here
3. Add to probability_specs list
4. That's it — compute_probabilities() handles the rest

To swap methods (table/formula/model):
Edit the 'method' field in each spec in its .py file, or change ENABLE_CULL_*
flags in parameters.py to toggle modules.
"""

import numpy as np
import pandas as pd

from cow import parameters as cow_parameters
from cow.parameters import ENABLE_CULL_AGE_STAGE, ENABLE_CULL_REPRO, RESUME_BREEDING_AFTER_LOSS
from mrp.engine import run_specs, validate_model_completeness

# Import all probability modules
from cow.probabilities.mortality import mortality_spec
from cow.probabilities.culling import culling_age_stage_spec, culling_repro_spec
from cow.probabilities.conception import conception_spec
from cow.probabilities.abortion import abortion_spec
from cow.probabilities.calving import calving_spec


# Register every probability spec (culling_repro is a formal spec like the other culling module).
probability_specs = [
    mortality_spec,
    culling_age_stage_spec,
    culling_repro_spec,
    conception_spec,
    abortion_spec,
    calving_spec,
]

# p_cull is not any single spec's raw `produces` output -- it's the
# Domain-composed competing-risk combination of the culling
# submodules above (see compute_probabilities() below). It is registered
# here as an additional available probability_output for dependency
# validation purposes, alongside each spec's own `produces`.
COMPOSED_PROBABILITY_OUTPUTS = {'p_cull'}


def _produced_names(spec):
    produces = spec.get('produces')
    if produces is None:
        return []
    return [produces] if isinstance(produces, str) else list(produces)


def _available_for_probabilities(states):
    """
    Build the Engine's 'available' dict for validating probability specs.
    parameter/external_input availability is a Domain concern (mrp_engine.py
    never imports cow.parameters itself) -- introspecting the parameters
    module namespace directly, no separate parameter registry.
    """
    probability_outputs = COMPOSED_PROBABILITY_OUTPUTS.copy()
    for spec in probability_specs:
        probability_outputs.update(_produced_names(spec))

    return {
        "state": set(states.columns),
        "derived_state": set(states.columns),
        "probability_output": probability_outputs,
        "reward_output": set(),
        "parameter": {n for n in dir(cow_parameters) if n.isupper()},
        "external_input": set(),
        "model_output": set(),
        "time": {"time_unit"},
    }


def compute_probabilities(states):
    """
    Compute every probability for every state. Array lookups only -- the milk
    curve / cull DIM were precomputed in cow/tables.py before states existed.
    Returns p_dead, p_cull, p_preg, p_abort, p_calve, p_resume (one value per state).
    """
    dp = states['dp'].values

    # Pre-flight: every declared probability spec's dependencies exist and
    # its selected method resolves, before any numeric computation runs.
    validate_model_completeness(
        states, probability_specs, reward_specs=[],
        available=_available_for_probabilities(states),
    )

    results = run_specs(probability_specs, states, context={})

    p_dead = results['mortality']

    # ---- Modular, competing-risk culling ----
    p_cull = np.zeros(len(states))
    if ENABLE_CULL_AGE_STAGE:
        p_cull = 1 - (1 - p_cull) * (1 - results['culling_age_stage'])
    if ENABLE_CULL_REPRO:
        p_cull = 1 - (1 - p_cull) * (1 - results['culling_repro'])

    # Conception: only open cows (dp == 0). The farmer's last breeding day is applied by day inside the
    # conception hazard ("do not breed" past it: probability 0, no breeding cost).
    p_preg = np.where(dp == 0, results['conception'], 0.0)   # cutoff (LAST_BREEDING_DIM_DAYS) is inside the hazard

    # Pregnancy loss: P(a cow that loses her pregnancy in this step conceives again in the remaining half step).
    # Same conception hazard as the open cow's step probability q, halved in time: 1 - sqrt(1 - q). 0 when disabled.
    q = np.clip(np.asarray(results['conception'], dtype=float), 0.0, 1.0)
    p_resume = np.where((dp > 0) & RESUME_BREEDING_AFTER_LOSS, 1.0 - np.sqrt(1.0 - q), 0.0)

    return {
        'p_resume': p_resume,
        'p_dead': p_dead,
        'p_cull': p_cull,
        'p_preg': p_preg,
        'p_abort': results['abortion'],
        'p_calve': results['calving'],
    }

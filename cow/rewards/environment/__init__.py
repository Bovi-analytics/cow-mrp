"""
cow/rewards/environment/__init__.py
====================================
Environmental-emissions biological-quantity orchestrator.

This layer sits BETWEEN probabilities and rewards:
    states + probs -> compute_environment() -> bio quantities (g/kg, NOT $)
    bio quantities -> environment reward specs -> $ reward components

Biological calculation is kept strictly separate from reward presentation
(see PRIMARY.md architecture and the environmental scientific audit).
Every quantity here is a physical unit (g N/step, g CH4/step, kg N2O/step,
...), never a reward point -- reward transformation happens only in
cow/rewards/environment/reward_*.py.

Quantities are computed in STAGES so each stage's outputs are available
as `model_output` context to the next (BW/DMI/milk yield -> nitrogen
intake -> N excretion pathways -> manure N -> manure CH4/N2O), exactly
mirroring how compute_probabilities()'s output is threaded into
compute_rewards() as context. No changes to mrp_core.py/mrp_engine.py
were needed: `run_specs` already executes any spec shaped
{name, depends_on, method, methods} identically regardless of what the
method names are, and the typed depends_on schema already reserves a
`model_output` kind for exactly this.

BW/DMI/milk yield are NOT reinvented here -- stage 0 wraps the repo's
EXISTING pipeline functions (cow.rewards.base.body_weight_kg/
dry_matter_intake_kg, cow.states.milk_yield_kg), the same ones
feed_cost.py already uses, so a cow's environmental calculations and her
feed cost are driven by the same BW/DMI/yield -- not two disagreeing
values for the same cow (see the scientific audit).

To add a new environmental biological quantity:
1. Create a spec (produces=..., depends_on=[...], method, methods={...})
   in the relevant module (nitrogen.py / enteric_methane.py / ...).
2. Add it to the appropriate _STAGE_* list below (or a new stage, if it
   depends on something not yet computed at that point).
3. That's it -- it becomes a bio dict key, a CSV column (via the
   existing milk_produced_kg-style tracked-metric mechanism in
   cow/rewards/__init__.py), and is available as context['your_name']
   to every later stage and to reward specs.
"""

import numpy as np

from cow.parameters import UNIT_DAYS, MANURE_SYSTEM, MANURE_CLIMATE
from cow.states import milk_yield_kg
from cow.rewards.base import age_days_of, body_weight_kg, dry_matter_intake_kg, mid_step_dim_days

from cow.rewards.environment.nitrogen import (
    nitrogen_intake_spec, fecal_n_spec, urinary_n_spec, milk_n_spec, manure_n_spec,
)
from cow.rewards.environment.enteric_methane import enteric_methane_spec
from cow.rewards.environment.manure_methane import manure_methane_spec
from cow.rewards.environment.co2 import respiratory_co2_spec
from cow.rewards.environment.n2o import (
    n2o_direct_spec, n2o_indirect_volatilization_spec, n2o_indirect_leaching_spec,
)

from mrp.engine import run_specs, validate_model_completeness


# ============================================================
# Stage 0: reused pipeline values (BW, DMI, milk yield), wrapped as
# specs purely so downstream environment specs can declare them as
# ordinary model_output dependencies instead of re-importing/re-calling
# the underlying functions themselves.
# ============================================================

def _bw_formula(states, context):
    """BW (kg) -- a snapshot/stock quantity, NOT a flow: no UNIT_DAYS scaling."""
    lac = states['lactation'].values
    dim = states['dim'].values
    dp = states['dp'].values
    age_days = age_days_of(lac, dim)
    return body_weight_kg(age_days, mid_step_dim_days(dim), dp * UNIT_DAYS)

bw_spec = {
    'name': 'bw_kg', 'produces': 'bw_kg',
    'depends_on': {'state': ['lactation', 'dim', 'dp'], 'derived_state': [], 'probability_output': [],
                    'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': []},
    'method': 'existing_pipeline', 'methods': {'existing_pipeline': _bw_formula},
}


def _milk_yield_formula(states, context):
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    return milk_yield_kg(dim, lac, dp) * UNIT_DAYS

milk_yield_wrapper_spec = {
    'name': 'milk_yield_kg', 'produces': 'milk_yield_kg',
    'depends_on': {'state': ['dim', 'lactation', 'dp'], 'derived_state': [], 'probability_output': [],
                    'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': []},
    'method': 'existing_pipeline', 'methods': {'existing_pipeline': _milk_yield_formula},
}


def _dmi_formula(states, context):
    """
    NOTE: recomputes BW directly (via body_weight_kg(), same helper
    bw_spec above wraps) rather than reading context['bw_kg'] -- bw_spec
    and dmi_spec run in the SAME run_specs() batch (stage 0), and
    run_specs does not thread one spec's output to another within a
    single batch (only *between* the staged compute_environment() calls
    below does that happen). Recomputing BW here (cheap, pure function,
    identical to feed_cost.py's own usage) avoids a same-batch ordering
    bug rather than working around it.
    """
    dim = states['dim'].values
    lac = states['lactation'].values
    dp = states['dp'].values
    milk_actual = milk_yield_kg(dim, lac, dp)
    age_days = age_days_of(lac, dim)
    bw = body_weight_kg(age_days, mid_step_dim_days(dim), dp * UNIT_DAYS)
    dim_days = np.maximum(mid_step_dim_days(dim), 1)
    dmi = dry_matter_intake_kg(milk_actual, bw, dim_days)
    return dmi * UNIT_DAYS

dmi_spec = {
    'name': 'dmi_kg', 'produces': 'dmi_kg',
    'depends_on': {'state': ['dim', 'lactation', 'dp'], 'derived_state': [],
                    'probability_output': [], 'reward_output': [], 'parameter': [],
                    'external_input': [], 'model_output': [], 'time': []},
    'method': 'existing_pipeline', 'methods': {'existing_pipeline': _dmi_formula},
}


_STAGE_0 = [bw_spec, milk_yield_wrapper_spec, dmi_spec]
_STAGE_1 = [nitrogen_intake_spec, enteric_methane_spec, respiratory_co2_spec]
_STAGE_2 = [fecal_n_spec, urinary_n_spec, milk_n_spec]
_STAGE_3 = [manure_n_spec]
_STAGE_MANURE = [manure_methane_spec]
_STAGE_N2O = [n2o_direct_spec, n2o_indirect_volatilization_spec, n2o_indirect_leaching_spec]

_ALL_ENV_SPECS = _STAGE_0 + _STAGE_1 + _STAGE_2 + _STAGE_3 + _STAGE_MANURE + _STAGE_N2O


def _manure_system_configured():
    return MANURE_SYSTEM is not None and MANURE_CLIMATE is not None


def _produced_names(spec):
    produces = spec.get('produces')
    return [produces] if isinstance(produces, str) else list(produces or [])


def _available_for_environment(states, probs):
    model_outputs = set()
    for spec in _ALL_ENV_SPECS:
        model_outputs.update(_produced_names(spec))
    from cow import parameters as cow_parameters
    return {
        "state": set(states.columns), "derived_state": set(states.columns),
        "probability_output": set(probs.keys()), "reward_output": set(),
        "parameter": {n for n in dir(cow_parameters) if n.isupper()},
        "external_input": set(), "model_output": model_outputs, "time": {"time_unit"},
    }


def _run_stage(specs, states, context):
    """Run one stage of environment specs, tag each with its active method, merge into context."""
    out = run_specs(specs, states, context=context)
    for spec in specs:
        out[f"{spec['name']}_method"] = np.full(len(states), spec['method'], dtype=object)
    return out


def compute_environment(states, probs):
    """
    Compute every environmental biological quantity, staged so later
    stages can consume earlier ones as `model_output` context. Returns a
    flat dict (quantity_name -> array, plus "<name>_method" -> array of
    the method-name string used) suitable for merging into the rewards
    dict (CSV-observable, matching the milk_produced_kg tracked-metric
    pattern) and for use as reward-spec context.

    Manure CH4 and all N2O pathways are SKIPPED (not silently defaulted)
    unless COW_MANURE_SYSTEM and COW_MANURE_CLIMATE are both set -- IPCC
    MCF/EF values are climate- and system-dependent, so no single default
    is scientifically defensible (see manure_methane.py).
    """
    validate_model_completeness(
        states, probability_specs=_ALL_ENV_SPECS if _manure_system_configured()
                else [s for s in _ALL_ENV_SPECS if s not in _STAGE_MANURE + _STAGE_N2O],
        reward_specs=[], available=_available_for_environment(states, probs),
    )

    context = dict(probs)
    context.update(_run_stage(_STAGE_0, states, context))
    context.update(_run_stage(_STAGE_1, states, context))
    context.update(_run_stage(_STAGE_2, states, context))
    context.update(_run_stage(_STAGE_3, states, context))

    if _manure_system_configured():
        context.update(_run_stage(_STAGE_MANURE, states, context))
        context.update(_run_stage(_STAGE_N2O, states, context))
        context['n2o_total_kg'] = (
            context['n2o_direct_kg'] + context['n2o_indirect_volatilization_kg']
            + context['n2o_indirect_leaching_kg']
        )
        context['n2o_total_method'] = np.full(len(states), 'ipcc_2019', dtype=object)

    return {k: v for k, v in context.items() if k not in probs}


ENVIRONMENT_METHOD_REGISTRY = {
    'nitrogen': nitrogen_intake_spec['methods'],
    'enteric_methane': enteric_methane_spec['methods'],
    'manure_methane': manure_methane_spec['methods'],
    'n2o': n2o_direct_spec['methods'],
}

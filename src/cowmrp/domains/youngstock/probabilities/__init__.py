"""
youngstock/probabilities/__init__.py
=====================================
Probability orchestrator for the youngstock chain -- same pattern as
cow/probabilities/__init__.py: import every spec, register it in
probability_specs, and compute_probabilities() runs them through the shared
mrp_engine.run_specs after the pre-flight dependency/method validation.

To add a spec: create it in a module here, import it, append to
probability_specs. To swap a method: edit the spec's 'method' field
(table / formula / model) in its own file.
"""

from cowmrp.domains.youngstock import parameters as youngstock_parameters
from cowmrp.mrp.engine import run_specs, validate_model_completeness

from cowmrp.domains.youngstock.probabilities.mortality import mortality_spec
from cowmrp.domains.youngstock.probabilities.heifer_repro import (
    conception_spec, breeding_exit_spec, pregnant_exit_spec,
    heifer_abortion_spec, heifer_calving_spec,
)
from cowmrp.domains.youngstock.probabilities.birth_outcomes import birth_outcomes_spec

probability_specs = [
    mortality_spec,
    conception_spec,
    breeding_exit_spec,
    pregnant_exit_spec,
    heifer_abortion_spec,
    heifer_calving_spec,
    birth_outcomes_spec,
]


def _available(states):
    return {
        "state": set(states.columns),
        "derived_state": set(states.columns),
        "probability_output": set(),
        "reward_output": set(),
        "parameter": {n for n in dir(youngstock_parameters) if n.isupper()},
        "external_input": set(),
        "model_output": set(),
        "time": {"time_unit"},
    }


def compute_probabilities(states):
    """Per-step probability vectors for every youngstock state, plus the
    birth-outcome mixes ('birth_mix': {'1': vec, '2+': vec})."""
    validate_model_completeness(states, probability_specs, reward_specs=[], available=_available(states))
    results = run_specs(probability_specs, states, context={})
    return {
        'p_die': results['calf_heifer_mortality'],
        'p_conceive': results['heifer_conception'],
        'p_exit_breeding': results['breeding_exit'],
        'p_exit_pregnant': results['pregnant_exit'],
        'p_abort': results['heifer_abortion'],
        'p_calve': results['heifer_calving'],
        'birth_mix': results['birth_outcomes'],
    }

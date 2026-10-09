"""
mrp_engine.py
=============
Generic MRP orchestration layer (domain-independent).

This module knows the *shape* of a spec (name / depends_on / method /
methods / produces / sign) and orchestrates the declare -> validate ->
resolve -> construct workflow around it. It never knows what any
particular state, dependency, or spec means -- that meaning belongs to
the domain (e.g. cow/).

mrp_core.py is the layer below this one: pure math/data-structure
primitives (matrices, Bellman/exact-solve, distribution evolution,
simulation, discounting) that have no notion of "spec" at all.

Dependency direction: Domain -> mrp_engine -> mrp_core. This file may
import from mrp_core; mrp_core must never import from this file or from
any domain package.
"""

import numpy as np
import pandas as pd


class MissingDependencyError(ValueError):
    """A spec's declared depends_on requires something that isn't available."""
    pass


class LookupCoverageError(ValueError):
    """A lookup table is missing required key columns, or a spec produced
    NaN values -- likely an uncovered lookup combination."""
    pass


# ============================================================
# STATE SPACE CONSTRUCTION
# ============================================================

def build_states(variable_domains, derived_variables):
    """
    Build the full state table from independent variables and derived rules.

    variable_domains: dict of {name: list_of_values}
    derived_variables: list of (name, [(value, condition_fn), ...])
    """
    names = list(variable_domains.keys())
    mesh = np.meshgrid(*[variable_domains[n] for n in names], indexing='ij')
    grid = pd.DataFrame({n: m.ravel() for n, m in zip(names, mesh)})

    frames = [grid]
    for var_name, rules in derived_variables:
        next_frames = []
        for frame in frames:
            row = {col: frame[col].values for col in frame.columns}
            matches = [(condition(row), value) for value, condition in rules]
            match_count = sum(m.astype(int) for m, _ in matches)

            if (match_count == 0).any():
                raise ValueError(f"No rule matches some row(s) for derived variable '{var_name}'")

            for mask, value in matches:
                if mask.any():
                    part = frame.loc[mask].copy()
                    part[var_name] = value
                    next_frames.append(part)
        frames = next_frames

    all_names = list(variable_domains) + [d[0] for d in derived_variables]
    states = pd.concat(frames, ignore_index=True).drop_duplicates(subset=all_names)
    states = states.reset_index(drop=True)
    states['state_id'] = states.index
    return states


# ============================================================
# METHOD RESOLUTION
# ============================================================

def resolve_method(spec):
    """
    Check that a spec's active method actually exists.
    Works for probability specs AND reward specs.
    """
    method = spec['method']
    implementations = spec.get('methods', {})
    if implementations.get(method) is None:
        available = [name for name, fn in implementations.items() if fn is not None]
        raise ValueError(
            f"'{spec['name']}' (depends on {spec['depends_on']}) "
            f"is set to use method '{method}', but no implementation was provided. "
            f"Available methods: {available or 'none'}."
        )
    return implementations[method]


# ============================================================
# SHARED SPEC EXECUTION (used identically by probability and
# reward orchestration -- no Engine class, just a function)
# ============================================================

def run_specs(specs, states, context=None, has_sign=False):
    """
    Generic executor for a list of specs (probability or reward).

    specs: list of spec dicts, each with name/method/methods (and,
        for reward-style specs, 'sign').
    states: the state DataFrame every spec function receives as its
        first argument.
    context: passed as each spec function's second argument verbatim
        (e.g. the already-computed `probs` dict for reward specs, or
        {} for probability specs that don't yet need cross-spec inputs).
    has_sign: if True, sums spec['sign'] * value into results['total']
        (reward-style aggregation). If False, no 'total' key is added.

    A spec function may return either a numeric array (the normal case,
    NaN-checked as a lookup-coverage safety net) or a dict of named
    sub-arrays (e.g. a health-transition spec producing several outputs
    at once) -- dict results are stored as-is, not NaN-checked or summed
    into 'total'.
    """
    context = context if context is not None else {}
    results = {}
    total = np.zeros(len(states)) if has_sign else None
    for spec in specs:
        fn = resolve_method(spec)
        value = fn(states, context)
        if isinstance(value, dict):
            results[spec['name']] = value
            continue
        value = np.asarray(value, dtype=float)
        if np.isnan(value).any():
            raise LookupCoverageError(
                f"'{spec['name']}' produced {int(np.isnan(value).sum())} NaN value(s) "
                f"-- likely an uncovered lookup combination. Inspect the table against "
                f"the states that hit it."
            )
        results[spec['name']] = value
        if has_sign:
            total += spec['sign'] * value
    if has_sign:
        results['total'] = total
    return results


# ============================================================
# DEPENDENCY / COMPLETENESS VALIDATION
# ============================================================

def validate_dependencies(spec, available):
    """
    Check that a spec's declared depends_on is satisfiable given what's
    available. Distinct from method availability (resolve_method) and
    from lookup coverage (validate_lookup_coverage) -- those are checked
    separately and must not be conflated with this.

    spec['depends_on'] may be the typed dict shape:
        {"state": [...], "derived_state": [...], "probability_output": [...],
         "reward_output": [...], "parameter": [...], "external_input": [...],
         "model_output": [...], "time": [...]}
    or, for specs not yet migrated to the typed shape, a flat list -- a
    flat list is treated as an all-"state" shorthand for backward
    compatibility during the migration.

    available: dict of {dependency_kind: set_of_available_names}.
    """
    depends_on = spec.get('depends_on', {})
    if isinstance(depends_on, list):
        depends_on = {'state': depends_on}

    missing = {}
    for kind, required in depends_on.items():
        have = available.get(kind, set())
        gap = [n for n in required if n not in have]
        if gap:
            missing[kind] = gap

    if missing:
        lines = [f"Missing dependencies for '{spec['name']}':"]
        for kind, names in missing.items():
            lines.append(
                f"  {kind}: required {depends_on[kind]}, "
                f"available {sorted(available.get(kind, []))}, missing {names}"
            )
        raise MissingDependencyError("\n".join(lines))


def validate_lookup_coverage(table, key_columns, spec_name):
    """
    Structural check: does the table have the columns the spec's
    dependency contract says it's keyed on? Full combination coverage is
    enforced separately, by the post-computation NaN check in run_specs
    -- not by eager enumeration here.
    """
    missing_cols = [c for c in key_columns if c not in table.columns]
    if missing_cols:
        raise LookupCoverageError(
            f"Table for '{spec_name}' is missing required key columns: {missing_cols}"
        )


def validate_model_completeness(states, probability_specs, reward_specs, available):
    """
    Pre-flight validation sweep: for every given spec, check its declared
    dependencies exist (validate_dependencies) and its selected method
    resolves (resolve_method) -- fails on the first hard error rather than
    collecting every error (fail early). Either spec list may be empty;
    callers validate whichever specs they have direct knowledge of.

    Returns a report dict on success.
    """
    report = {"probabilities": {}, "rewards": {}}
    for spec in probability_specs:
        validate_dependencies(spec, available)
        resolve_method(spec)
        report["probabilities"][spec['name']] = "OK"
    for spec in reward_specs:
        validate_dependencies(spec, available)
        resolve_method(spec)
        report["rewards"][spec['name']] = "OK"
    return report

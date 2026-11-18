#!/usr/bin/env python3
"""
verify_migration_parity.py
===========================
Small, dependency-free parity check (no pytest/test framework; the
pytest suite lives in tests/). Runs the Cow pipeline with
cow/parameters.py's own defaults (no COW_* env overrides -- this exactly
reproduces benchmark.py's "Model A" configuration) and compares a handful
of deterministic outputs against a saved baseline JSON within tight
numerical tolerance.

USAGE
-----
    python verify_migration_parity.py [path/to/baseline/worker_result.json]

Default baseline path: runners/cow/outputs/baseline/Model_A/worker_result.json
(produced by running benchmark.py once, before a change, and keeping the
Model_A subfolder as a reference).

Exits with status 0 and prints PARITY_OK on success; exits with status 1
and prints a diff on any mismatch.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import sys
import json

from cowmrp.mrp import core as mrp_core
from cowmrp.mrp import engine as mrp_engine
from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.domains.cow.states import variable_domains, derived_variables
from cowmrp.domains.cow.probabilities import compute_probabilities
from cowmrp.domains.cow.rewards import compute_rewards
from cowmrp.domains.cow.transitions import compute_transitions

DEFAULT_BASELINE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), "runners", "cow", "outputs", "baseline", "Model_A", "worker_result.json")
GAMMA_ANNUAL = 0.95
NUM_YEARS = 5
TOLERANCE = 1e-6  # relative tolerance for float comparisons


def run_pipeline():
    from cowmrp.domains.cow.states import build_cow_states
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = mrp_core.assemble_matrix(states, rows, cols, vals)
    mrp_core.validate_row_stochasticity(matrix, states)
    rewards = compute_rewards(states, probs)

    gamma = mrp_core.annual_to_per_step_gamma(GAMMA_ANNUAL, UNIT_DAYS)
    v_finite = mrp_core.evaluate_finite_horizon(matrix, rewards['total'], gamma, NUM_YEARS, UNIT_DAYS)
    v_infinite = mrp_core.evaluate_infinite_horizon(matrix, rewards['total'], gamma)

    return {
        "num_states": len(states),
        "nnz_transitions": int(matrix.nnz),
        "V_finite_mean": float(v_finite.mean()),
        "V_infinite_mean": float(v_infinite.mean()),
    }


def close_enough(a, b, tol=TOLERANCE):
    return abs(a - b) < tol * max(1.0, abs(b))


def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE
    with open(baseline_path) as f:
        baseline = json.load(f)

    current = run_pipeline()

    checks = [
        ("num_states", current["num_states"] == baseline["num_states"]),
        ("nnz_transitions", current["nnz_transitions"] == baseline["nnz_transitions"]),
        ("V_finite_mean", close_enough(current["V_finite_mean"], baseline["V_finite_mean"])),
        ("V_infinite_mean", close_enough(current["V_infinite_mean"], baseline["V_infinite_mean"])),
    ]

    failed = [name for name, ok in checks if not ok]
    if failed:
        print("PARITY_FAILED:", failed)
        print("current: ", current)
        print("baseline:", {k: baseline[k] for k, _ in checks})
        sys.exit(1)

    print("PARITY_OK:", current)
    sys.exit(0)


if __name__ == "__main__":
    main()

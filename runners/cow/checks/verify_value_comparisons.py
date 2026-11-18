#!/usr/bin/env python3
"""
verify_value_comparisons.py
============================
Dependency-free verification of the generic state-value comparison
primitives (mrp_core.state_values / state_value_diff) and the two Cow
Domain demonstrations (cow/value_comparisons.py::pregnancy_value,
replacement_value), plus the modular registry both are dispatched
through (VALUE_COMPARISON_FUNCTIONS).

State = (lactation, dim, dp) -- health / flag / milk no longer exist.

Checks:
1. state_values/state_value_diff match manual V[id_a]/V[id_b] indexing and agree.
2. state_value_diff raises ValueError for a structurally impossible query.
3. pregnancy_value: documented shape, value == V_state - V_comparison, varies by state.
4. replacement_value's target is EXACTLY the state cow/transitions.py routes
   replacement to, (lactation=1, dim=0, dp=0).
5. replacement_value varies across representative current states.
6. Both functions raise ValueError for out-of-range queries.
7. VALUE_COMPARISON_FUNCTIONS registry dispatches both names.
8. Reward-component decomposition of pregnancy_value sums to the direct difference
   (linearity of the infinite-horizon solve); the pregnancy_value baseline below is
   the CowMRP post-reformulation value, NOT comparable with the
   original repo's historical baselines (different state space and reward model).
9. rewards['total'] is bit-identical before/after calling the comparison functions.
10. dp generalization of pregnancy_value (gestation-stage curve).

Exits 0 and prints VALUE_COMPARISON_CHECK_OK on success.

USAGE
-----
    python verify_value_comparisons.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import os
import sys

os.environ.setdefault("COW_TIME_UNIT", "month")
os.environ.setdefault("COW_MAX_LACTATION", "10")
os.environ.setdefault("COW_MAX_DIM_DAYS", "1000")
os.environ.setdefault("COW_MAX_DP_DAYS", "300")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
from cowmrp.mrp.engine import build_states
from cowmrp.mrp.core import (
    evaluate_infinite_horizon, annual_to_per_step_gamma, assemble_matrix,
    state_value_diff, state_values, find_state_id,
)
from cowmrp.domains.cow.states import variable_domains, derived_variables
from cowmrp.domains.cow.probabilities import compute_probabilities
from cowmrp.domains.cow.rewards import compute_rewards
from cowmrp.domains.cow.transitions import compute_transitions
from cowmrp.domains.cow.value_comparisons import (
    pregnancy_value, replacement_value, decompose_pregnancy_value, VALUE_COMPARISON_FUNCTIONS,
)
from cowmrp.domains.cow.parameters import DRY_OFF_DP
from cowmrp.domains.cow.rewards.base import CALF_VALUE

GAMMA_ANNUAL, UNIT_DAYS = 0.95, 30

# CowMRP baseline for the intended model (Model A, MONTH, gamma 0.95/yr, lac=2, dim=5, dp=1):
# CALF_VALUE 1150 (intended) and a state reward vector WITHOUT replacement_cost.
# Derivation (all verified independently in main(), check #8, by a Neumann-series solve that uses no linear solver):
#   262.51  old pin: CALF_VALUE 250, replacement cost inside the reward (reproduced below from the decomposition)
#   410.37  CALF_VALUE 1150, replacement cost still inside the reward (pregnancy component x 1150/250)
#    80.88  CALF_VALUE 1150, replacement cost NOT in the reward = 410.37 - 329.49 (the avoided replacement spending, which the
#           replacement comparison now carries: cow/value_comparisons.py::replacement_comparison)
EXPECTED_PREGNANCY_VALUE = 80.88
OLD_PIN_CALF250_REPLACEMENT_IN_REWARD = 262.51
CALF_VALUE_OLD = 250.0


def main():
    gamma = annual_to_per_step_gamma(GAMMA_ANNUAL, UNIT_DAYS)

    print("Building Model A state space (fresh, in-memory)...")
    from cowmrp.domains.cow.states import build_cow_states
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = assemble_matrix(states, rows, cols, vals)
    rewards = compute_rewards(states, probs)
    total_before = rewards["total"].copy()
    V = evaluate_infinite_horizon(matrix, rewards["total"], gamma)
    print(f"  {len(states):,} states built.\n")

    # ---- 1 ---
    qA = {"lactation": 2, "dim": 5, "dp": 1}
    qB = {"lactation": 2, "dim": 5, "dp": 0}
    id_a, id_b = find_state_id(states, **qA), find_state_id(states, **qB)
    V_a, V_b = state_values(states, V, qA, qB)
    diff = state_value_diff(states, V, qA, qB)
    assert V_a == V[id_a] and V_b == V[id_b], "state_values doesn't match manual indexing"
    assert diff == V_a - V_b, "state_value_diff inconsistent with state_values"
    print(f"[OK] state_values/state_value_diff match manual indexing (V_a={V_a:.2f}, V_b={V_b:.2f}, diff={diff:.2f})")

    # ---- 2 ---
    try:
        state_value_diff(states, V, query_A={"lactation": 2, "dim": 99999, "dp": 0},
                         query_B={"lactation": 2, "dim": 5, "dp": 0})
        raise AssertionError("expected ValueError for a nonexistent state")
    except ValueError as e:
        assert "No state matches" in str(e)
        print("[OK] state_value_diff raises ValueError for a structurally impossible state")

    # ---- 3 ---
    pv_results = {}
    for lac in [1, 2, 3]:
        for dim in [2, 5, 8]:
            r = pregnancy_value(states, V, lactation=lac, dim=dim)
            assert set(r.keys()) == {"value", "V_state", "V_comparison"}
            assert np.isclose(r["value"], r["V_state"] - r["V_comparison"]) and np.isfinite(r["value"])
            pv_results[(lac, dim)] = r["value"]
    assert len(set(round(v, 2) for v in pv_results.values())) > 1, "pregnancy_value is constant everywhere"
    print(f"[OK] pregnancy_value: correct shape, state-dependent ({len(pv_results)} combinations)")

    # a pregnancy cannot start in a step the table does not hold (here: after the table's day cap, STATE_DIM_CAP_DAYS)
    from cowmrp.domains.cow.tables import get_tables
    dim_bad = get_tables()['e_max'] + 2        # landing state (dim, dp 1) has conception step dim - 1
    try:
        pregnancy_value(states, V, lactation=2, dim=dim_bad)
        raise AssertionError(f"expected ValueError: no state exists for (dim {dim_bad}, dp 1): conception step {dim_bad - 1} is past the cap")
    except ValueError as e:
        assert "No state matches" in str(e)
        print("[OK] pregnancy_value raises ValueError where the pregnant state cannot exist (conception impossible)")

    # ---- 4 ---
    replacement_query = {"lactation": 1, "dim": 0, "dp": 0}
    r = replacement_value(states, V, lactation=2, dim=5, dp=0)
    expected_diff = state_value_diff(states, V, {"lactation": 2, "dim": 5, "dp": 0}, replacement_query)
    assert np.isclose(r["value"], expected_diff), "replacement target != cow/transitions.py replacement state"
    print(f"[OK] replacement_value target is (lactation=1, dim=0, dp=0) (value={r['value']:.2f})")

    # ---- 5 ---
    rv = {}
    for lac, dim, dp in [(1, 0, 0), (2, 5, 0), (2, 5, 1), (3, 11, 0), (5, 12, 5)]:
        r = replacement_value(states, V, lactation=lac, dim=dim, dp=dp)
        assert np.isclose(r["value"], r["V_state"] - r["V_comparison"])
        rv[(lac, dim, dp)] = r["value"]
        print(f"  lac={lac} dim={dim:2d} dp={dp}: replacement_value = {r['value']:9.2f}")
    assert len(set(round(v, 2) for v in rv.values())) > 1
    print("[OK] replacement_value varies across representative states")

    # ---- 6 ---
    for fn, kwargs in [(pregnancy_value, dict(lactation=2, dim=99999)),
                       (replacement_value, dict(lactation=2, dim=99999, dp=0))]:
        try:
            fn(states, V, **kwargs)
            raise AssertionError(f"expected ValueError from {fn.__name__}")
        except ValueError:
            pass
    print("[OK] both functions raise ValueError for an out-of-range DIM")

    # ---- 7 ---
    r1 = VALUE_COMPARISON_FUNCTIONS["pregnancy_value"](states, V, lactation=2, dim=5)
    r2 = VALUE_COMPARISON_FUNCTIONS["replacement_value"](states, V, lactation=2, dim=5, dp=0)
    assert set(VALUE_COMPARISON_FUNCTIONS.keys()) == {"pregnancy_value", "replacement_value"}
    from cowmrp.domains.cow.value_comparisons import replacement_comparison
    rcmp = replacement_comparison(states, matrix, rewards, probs, gamma, lactation=2, dim=5, dp=0)
    assert abs(rcmp["value"] - (rcmp["V_total_state"] - rcmp["V_total_heifer"]
                                - (rcmp["PV_replacement_cost_state"] - rcmp["PV_replacement_cost_heifer"]) + rcmp["replacement_cost_now"])) < 1e-6
    print(f"[OK] replacement_comparison (replacement cost carried separately) = {rcmp['value']:.2f} vs the state-reward difference {r2['value']:.2f}")
    print(f"[OK] registry dispatches both comparisons (pregnancy_value={r1['value']:.2f}, "
          f"replacement_value={r2['value']:.2f})")

    # ---- 8 ---
    components = [("milk_income", +1), ("pregnancy_reward", +1), ("cull_reward", +1),
                  ("mortality_cost", -1), ("feed_cost", -1), ("reproduction_cost", -1)]
    id_preg, id_open = find_state_id(states, lactation=2, dim=5, dp=1), find_state_id(states, lactation=2, dim=5, dp=0)
    assert "replacement_cost" not in rewards, "replacement_cost must not be a state reward"

    def neumann(reward):          # independent infinite-horizon solve: sum_t gamma^t P^t r (no linear solver)
        term = np.asarray(reward, float).copy()
        out = term.copy()
        for _ in range(400000):
            term = gamma * matrix.dot(term)
            out += term
            if np.abs(term).max() < 1e-11:
                break
        return out

    total_check = 0.0
    contrib_ind = {}
    print("\nReward-component decomposition at lac=2, dim=5, dp=1 (library solver | independent Neumann series):")
    for name, sign in components:
        V_c = evaluate_infinite_horizon(matrix, sign * rewards[name], gamma)
        d = V_c[id_preg] - V_c[id_open]
        V_n = neumann(sign * rewards[name])
        d_ind = V_n[id_preg] - V_n[id_open]
        contrib_ind[name] = d_ind
        assert abs(d - d_ind) < 1e-6, (name, d, d_ind)
        total_check += d
        print(f"  {name:20s} (sign {sign:+d}): {d:10.2f} | {d_ind:10.2f}")
    direct = V[id_preg] - V[id_open]
    assert abs(total_check - direct) < 1e-6
    print(f"  {'SUM':20s}            : {total_check:10.2f}  (matches direct diff)")
    # the avoided replacement spending, carried by the replacement comparison and not by the state reward
    from cowmrp.domains.cow.rewards import replacement_cost_vector
    V_rc = neumann(replacement_cost_vector(states, probs))
    rc_component = -(V_rc[id_preg] - V_rc[id_open])                    # what the old reward vector contributed
    with_rc = total_check + rc_component
    old_pin = with_rc - contrib_ind["pregnancy_reward"] * (1 - CALF_VALUE_OLD / CALF_VALUE)
    print(f"  replacement cost (no longer in the reward): {rc_component:10.2f};  with it: {with_rc:.2f};  "
          f"with CALF_VALUE {CALF_VALUE_OLD:g} as well: {old_pin:.2f}")
    assert abs(old_pin - OLD_PIN_CALF250_REPLACEMENT_IN_REWARD) < 0.01, \
        f"the old pin ({OLD_PIN_CALF250_REPLACEMENT_IN_REWARD}) is not reproduced by the independent decomposition: {old_pin:.3f}"
    print(f"[OK] the previous pin {OLD_PIN_CALF250_REPLACEMENT_IN_REWARD} is reproduced independently (calf value 250, replacement cost inside the reward)")
    assert abs(total_check - EXPECTED_PREGNANCY_VALUE) < 0.01, \
        f"pregnancy_value(lac=2, dim=5) changed: {total_check:.2f} (expected {EXPECTED_PREGNANCY_VALUE})"
    print(f"[OK] pregnancy_value(lac=2, dim=5) = {direct:.2f} matches the pinned value for the intended model "
          f"(independent solve gives {total_check:.2f})")

    # ---- 10 ---
    lac_ref, dim_ref = 2, 10      # dim - dp must be a possible conception step: dp 1..10 are all valid at DIM step 10
    assert pregnancy_value(states, V, lactation=lac_ref, dim=dim_ref) == \
        pregnancy_value(states, V, lactation=lac_ref, dim=dim_ref, dp=1)
    dp_curve = {dp: pregnancy_value(states, V, lactation=lac_ref, dim=dim_ref, dp=dp)["value"]
                for dp in [1, 2, 3, 5, 7, DRY_OFF_DP, DRY_OFF_DP + 1]}
    assert len(set(round(v, 2) for v in dp_curve.values())) > 1
    print(f"[OK] pregnancy_value varies across dp 1..{DRY_OFF_DP + 1} (crosses the dry-off boundary without error)")
    try:
        pregnancy_value(states, V, lactation=lac_ref, dim=dim_ref, dp=0)
        raise AssertionError("expected ValueError for dp=0")
    except ValueError as e:
        assert "dp must be >= 1" in str(e)
        print("[OK] pregnancy_value(dp=0) raises ValueError")

    for dp in [1, 3]:
        contributions = decompose_pregnancy_value(states, matrix, rewards, gamma,
                                                  lactation=lac_ref, dim=dim_ref, dp=dp)
        assert set(contributions.keys()) == {n for n, _ in components}
        summed = sum(contributions.values())
        d_direct = pregnancy_value(states, V, lactation=lac_ref, dim=dim_ref, dp=dp)["value"]
        assert abs(summed - d_direct) < 1e-6, (dp, summed, d_direct)
    print("[OK] decompose_pregnancy_value sums to pregnancy_value's direct diff at dp=1 and dp=3")

    # ---- 9 ---
    assert np.array_equal(rewards["total"], total_before), "rewards['total'] changed -- diagnostics leaked!"
    print("[OK] rewards['total'] bit-identical before/after calling the comparison functions")

    print("\nVALUE_COMPARISON_CHECK_OK")


if __name__ == "__main__":
    main()

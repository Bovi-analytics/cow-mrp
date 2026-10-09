"""
audit_metrics.py
================
Experiment harness for the CowMRP audit: run the COMPLETE model at one or more time
resolutions, record a fixed set of metrics, and compare two snapshots (BEFORE vs AFTER a change).

    python audit_metrics.py run <label> [--units month,week,day]
    python audit_metrics.py compare <labelA> <labelB> [--units month,week,day]
    python audit_metrics.py list

Each resolution runs in its own subprocess (cow.parameters reads COW_TIME_UNIT at import).
Snapshots: outputs/audit/snapshots/<label>_<unit>.json  (generated files).

Metrics: state count, nnz, reachable states from the replacement state, row-stochasticity,
V at the replacement state, per-reward-component V at the replacement state, V of a reference
open cow (lactation 2, DIM ~150 d), pregnancy_value / replacement_value at that reference,
and lifetime statistics of one replacement-to-replacement cycle (expected length, calvings,
open / pregnant / dry days, exits by cause, mean lactation).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # project root (mrp/, cow/)
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # project root
SNAP = os.path.join(HERE, "outputs", "audit", "snapshots")
UNITS_DEFAULT = ("month", "week", "day")


def _apply_pinedo_sensitivity():
    """COW_PINEDO_SENS="death=1.15,other=0.85,noise=0.10,seed=3": multiply the transcribed Pinedo 2010 hazards
    (died / the five other-reason codes) by a constant and / or an independent lognormal error per value (sd = noise),
    BEFORE any table is built. Sensitivity only: the evidence module on disk is never edited."""
    spec = os.environ.get("COW_PINEDO_SENS", "")
    if not spec:
        return
    import numpy as np
    from cow.evidence import pinedo2010 as E
    kv = dict(item.split("=") for item in spec.split(","))
    f_death, f_other = float(kv.get("death", 1)), float(kv.get("other", 1))
    noise, rng = float(kv.get("noise", 0)), np.random.default_rng(int(kv.get("seed", 0)))
    for store in (E.OPEN, E.PREGNANT):
        for code, by_parity in store.items():
            f = f_death if code == "died" else f_other
            for par, vals in by_parity.items():
                by_parity[par] = [v * f * (float(np.exp(rng.normal(0, noise))) if noise else 1.0) for v in vals]


def collect(print_result=True):
    _apply_pinedo_sensitivity()
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from scipy.sparse.csgraph import breadth_first_order
    import cow.parameters as P
    from mrp import core as mrp_core
    from mrp import engine as mrp_engine
    from cow.states import variable_domains, derived_variables
    from cow.probabilities import compute_probabilities
    from cow.rewards import compute_rewards, BASE_REWARD_SPECS
    from cow.transitions import compute_transitions, get_replacement_state_id
    from cow.value_comparisons import pregnancy_value, replacement_value
    import cow.probabilities.culling as C
    # optional sensitivity switch: COW_METHOD_OVERRIDES="abortion=model,calving=model" selects another method of a spec
    # (table | formula | model) for this run only; defaults are never changed on disk.
    import cow.probabilities as _pr
    # COW_DISABLE="repro,age_stage" switches off the milk-floor reproductive cull and / or the other-reason culling table
    # for this run only (ablation); the probability composer reads these module flags.
    _dis = set(filter(None, os.environ.get("COW_DISABLE", "").split(",")))
    _pr.ENABLE_CULL_REPRO = "repro" not in _dis
    _pr.ENABLE_CULL_AGE_STAGE = "age_stage" not in _dis
    for item in filter(None, os.environ.get("COW_METHOD_OVERRIDES", "").split(",")):
        spec_name, method = item.split("=")
        next(s for s in _pr.probability_specs if s["name"] == spec_name)["method"] = method

    try:
        from cow.states import build_cow_states as _build
        states = _build()
    except ImportError:
        states = mrp_engine.build_states(variable_domains, derived_variables)
    U = P.UNIT_DAYS
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = mrp_core.assemble_matrix(states, rows, cols, vals)
    mrp_core.validate_row_stochasticity(matrix, states)
    rewards = compute_rewards(states, probs)
    gamma = mrp_core.annual_to_per_step_gamma(0.95, U)
    rid = get_replacement_state_id(states)
    n = len(states)

    out = {"unit": P.TIME_UNIT, "num_states": n, "nnz": int(matrix.nnz), "row_stochastic": True}
    M = matrix.tocsr()
    out["reachable_states"] = int(len(breadth_first_order(M, rid, directed=True, return_predecessors=False)))
    out["rewards_finite"] = bool(all(np.isfinite(np.asarray(v, dtype=float)).all()
                                      for k, v in rewards.items() if np.asarray(v).dtype.kind in "fi"))

    V = mrp_core.evaluate_infinite_horizon(matrix, rewards["total"], gamma)
    out["values_finite"] = bool(np.isfinite(V).all())
    out["V_replacement"] = float(V[rid])
    out["V_mean"] = float(V.mean())
    k150 = max(0, int(round(150 / U)) - 1)                      # mid-step age ~150 d
    ref = {"lactation": 2, "dim": k150}
    out["V_open_lac2_d150"] = float(V[mrp_core.find_state_id(states, dp=0, **ref)])
    out["pregnancy_value_dp1_lac2_d150"] = float(pregnancy_value(states, V, **ref)["value"])   # NOT resolution-invariant:
    # the dp=1 cohort has been pregnant ~U/2 days at the boundary (0.5 d at DAY, 15 d at MONTH)
    import math
    k_m = max(0, math.floor((165 - U / 2) / U + 0.5))            # DIM step with mean START age ~165 d
    dp_m = math.floor((105 - U / 2) / U + 0.5) + 1               # DP step with mean START gestation age ~105 d
    out["pregnancy_value_matched_lac2_dim165_gest105"] = float(pregnancy_value(states, V, lactation=2, dim=k_m, dp=dp_m)["value"])
    out["replacement_value_lac2_d150"] = float(replacement_value(states, V, dp=0, **ref)["value"])
    comp = {}
    for sp_ in BASE_REWARD_SPECS:
        Vc = mrp_core.evaluate_infinite_horizon(matrix, sp_["sign"] * np.asarray(rewards[sp_["name"]]), gamma)
        comp[sp_["name"]] = float(Vc[rid])
    out["V_components_replacement"] = comp

    # lifetime of one replacement-to-replacement cycle (replacement made absorbing)
    Q = M.copy().tolil()
    Q[:, rid] = 0
    Q = Q.tocsr()
    e = np.zeros(n)
    e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)      # expected visits per cycle
    lac, dim, dp = (states[k].values for k in ("lactation", "dim", "dp"))
    survive = (1 - probs["p_dead"]) * (1 - probs["p_cull"])
    pr = C.p_cull_repro(states, {}) * (1.0 if _pr.ENABLE_CULL_REPRO else 0.0)
    pa = C.p_culling_age_stage_full_table(states, {}) * (1.0 if _pr.ENABLE_CULL_AGE_STAGE else 0.0)
    out["cycle_days"] = float(x.sum() * U)
    out["calvings_per_cycle"] = float((x * survive * (1 - probs["p_abort"]) * probs["p_calve"]).sum())
    out["open_days"] = float(x[dp == 0].sum() * U)
    out["pregnant_days"] = float(x[(dp > 0) & (dp < P.DRY_OFF_DP)].sum() * U)
    out["dry_days"] = float(x[dp >= P.DRY_OFF_DP].sum() * U)
    yrs = x.sum() * U / 365.0
    out["death_pct_per_year"] = float((x * probs["p_dead"]).sum() / yrs * 100)
    out["live_cull_pct_per_year"] = float((x * (1 - probs["p_dead"]) * probs["p_cull"]).sum() / yrs * 100)
    out["mean_lactation"] = float((x * lac).sum() / x.sum())
    out["exit_death"] = float((x * probs["p_dead"]).sum())
    out["exit_cull_age_stage"] = float((x * (1 - probs["p_dead"]) * pa * (1 - pr)).sum())
    out["exit_cull_repro"] = float((x * (1 - probs["p_dead"]) * pr).sum())
    out["exit_max_lactation_calving"] = float((x * survive * (lac >= P.MAX_LACTATION) *
                                                (1 - probs["p_abort"]) * probs["p_calve"]).sum())

    # one lactation in isolation (lactation 2, from calving): does she calve again, and after how many days?
    trans = np.where(lac == 2)[0]
    nxt = np.where(lac == 3)[0]
    Q2 = M[trans][:, trans].tocsc()
    R_calve = np.asarray(M[trans][:, nxt].sum(axis=1)).ravel()
    I2 = sp.eye(len(trans), format="csc")
    i0 = int(np.where((dim[trans] == 0) & (dp[trans] == 0))[0][0])
    e0 = np.zeros(len(trans))
    e0[i0] = 1.0
    visits = spl.spsolve((I2 - Q2).T.tocsc(), e0)                  # expected visits from the start of lactation 2
    h = spl.spsolve((I2 - Q2).tocsc(), R_calve)                     # P(calve eventually | state)
    out["lac2_P_calve_again"] = float(visits @ R_calve)
    out["lac2_calving_interval_days_if_calved"] = float(U * (visits @ h) / (visits @ R_calve))
    out["lac2_days_in_lactation_any_exit"] = float(visits.sum() * U)
    if print_result:
        print("RESULT" + json.dumps(out))
    return out


def run(label, units):
    os.makedirs(SNAP, exist_ok=True)
    for unit in units:
        env = dict(os.environ, COW_TIME_UNIT=unit)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--collect"], env=env, cwd=HERE,
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode != 0 or not line:
            print(f"[FAIL] {unit}:\n{p.stderr[-2000:]}")
            sys.exit(1)
        with open(os.path.join(SNAP, f"{label}_{unit}.json"), "w", encoding="utf-8") as f:
            f.write(line[0][6:])
        d = json.loads(line[0][6:])
        print(f"[{unit:5s}] {label}: states={d['num_states']:,} reachable={d['reachable_states']:,} "
              f"V(rep)={d['V_replacement']:,.1f} cycle={d['cycle_days']:.0f}d calvings={d['calvings_per_cycle']:.2f}")


def _flat(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def compare(a, b, units):
    for unit in units:
        fa = _flat(json.load(open(os.path.join(SNAP, f"{a}_{unit}.json"), encoding="utf-8")))
        fb = _flat(json.load(open(os.path.join(SNAP, f"{b}_{unit}.json"), encoding="utf-8")))
        print(f"\n=== {unit}: {a} -> {b}")
        print(f"{'metric':42s} {'before':>14s} {'after':>14s} {'change':>10s}")
        for k in fa:
            va, vb = fa[k], fb.get(k)
            if k == "unit" or not isinstance(va, (int, float)) or isinstance(va, bool):
                continue
            ch = "" if va == vb else (f"{(vb - va) / abs(va):+.2%}" if va else f"{vb - va:+.3g}")
            print(f"{k:42s} {va:14,.3f} {vb:14,.3f} {ch:>10s}")


def main():
    args = sys.argv[1:]
    if "--collect" in args:
        return collect()
    units = UNITS_DEFAULT
    if "--units" in args:
        units = tuple(args[args.index("--units") + 1].split(","))
        args = [a for i, a in enumerate(args) if a != "--units" and (i == 0 or args[i - 1] != "--units")]
    if args[0] == "run":
        run(args[1], units)
    elif args[0] == "compare":
        compare(args[1], args[2], units)
    elif args[0] == "list":
        print("\n".join(sorted(os.listdir(SNAP))) if os.path.isdir(SNAP) else "(no snapshots)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

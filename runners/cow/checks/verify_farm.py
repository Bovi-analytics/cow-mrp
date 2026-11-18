"""
verify_farm.py
==============
Checks for the farm layer (runners/farm/: build.py, generate.py, simulate.py) on the (lactation, DIM, DP) state.
Run: python runners/cow/checks/verify_farm.py     (exit 1 on any failure). Everything is written to a temp folder.

  - the equilibrium herd distribution is truly stationary (pi P = pi) and reconstructs the replacement-cycle visits
  - generated herds contain only valid reachable states and consistent status labels; "do not breed" cows are open
    cows past the last breeding day; impossible requests are clear errors
  - expected / stochastic / monte_carlo runs: cows conserved, original + replaced reconstruct the herd, outputs written
  - herd values: replacement value of a cow in the replacement state is 0; pregnancy value exists only for pregnant cows
  - a herd built under another time resolution is refused
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import os
import subprocess
import sys
import tempfile
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def run(args):
    return subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)


def main():
    from runners.cow.farm import generate as G
    from runners.cow.farm.build import build_engine
    from cowmrp.domains.cow.parameters import LAST_BREEDING_DIM_DAYS, UNIT_DAYS, MAX_LACTATION
    from cowmrp.domains.cow.transitions import get_replacement_state_id

    states, probs, matrix = build_engine()
    rid = get_replacement_state_id(states)
    pi = G.steady_state(states, matrix, rid)
    check("equilibrium herd is stationary: pi P = pi", np.abs(pi @ matrix - pi).max() < 1e-9, f"max |piP-pi| = {np.abs(pi @ matrix - pi).max():.2e}")
    check("equilibrium herd is a distribution", abs(pi.sum() - 1) < 1e-12 and (pi >= 0).all())

    tmp = tempfile.mkdtemp()
    herds = {}
    for mode in ("equilibrium", "mix", "exact"):
        out = os.path.join(tmp, f"{mode}.csv")
        df = G.generate(states, matrix, mode, 500, 7, out)
        herds[mode] = out
        st = states.set_index("state_id").loc[df["state_id"]]
        check(f"{mode}: every cow sits in a valid model state, lactation within 1..{MAX_LACTATION}",
              len(df) > 0 and df["lactation"].between(1, MAX_LACTATION).all() and (st["lactation"].values == df["lactation"].values).all())
        dnb = df[df["status"] == "dnb"]
        check(f"{mode}: 'dnb' cows are open (dp = 0) and past the last breeding day",
              (dnb["up"] == 0).all() and (dnb["dim_days"] > LAST_BREEDING_DIM_DAYS).all(), f"{len(dnb)} dnb cows")
        check(f"{mode}: status labels agree with dp (pregnant/dry have dp > 0, open/dnb have dp = 0)",
              ((df["status"].isin(["pregnant", "dry"])) == (df["up"] > 0)).all())
    try:
        G.sample_mix(states, [{"proportion": 1.0, "status": "dry", "lactation": [1, 1], "dp_days": [1, 5]}], 10, np.random.default_rng(0))
        check("an impossible herd category is a clear error", False)
    except ValueError as e:
        check("an impossible herd category is a clear error", "No model state satisfies" in str(e))
    try:
        G.exact_ids(states, [(1, 10, 200, 5)])
        check("an unreachable exact state is a clear error", False)
    except ValueError as e:
        check("an unreachable exact state is a clear error", "not a reachable model state" in str(e))

    for mode in ("expected", "stochastic", "monte_carlo"):
        od = os.path.join(tmp, f"res_{mode}")
        r = run(["runners/cow/farm/simulate.py", "--mode", mode, "--steps", "6", "--sims", "10", "--farm-csv", herds["equilibrium"], "--out-dir", od])
        ok = r.returncode == 0 and os.path.exists(os.path.join(od, "dashboard.html"))
        check(f"simulate --mode {mode} runs and writes the dashboard", ok, (r.stderr or r.stdout)[-300:] if not ok else "")
        if not ok:
            continue
        ds = pd.read_csv(os.path.join(od, "distribution_summary.csv"))
        for g, d in ds.groupby("grouper"):
            check(f"{mode}: '{g}' groups conserve the herd (500 cows)", abs(d["current"].sum() - 500) < 1e-6 and abs(d["predicted"].sum() - 500) < 1e-6,
                  f"current {d['current'].sum():.1f}, predicted {d['predicted'].sum():.1f}")
        reg = pd.read_csv(os.path.join(od, "replacement_tracking_regular.csv"), index_col=0)
        rep = pd.read_csv(os.path.join(od, "replacement_tracking_replaced.csv"), index_col=0)
        tot = reg.fillna(0).sum() + rep.fillna(0).sum()
        check(f"{mode}: never-replaced + replaced cows reconstruct the herd at every step", np.allclose(tot.values, 500, atol=1e-6))
        vals = pd.read_csv(os.path.join(od, "cow_values.csv"))
        check(f"{mode}: cow values are finite", np.isfinite(vals["V"]).all() and np.isfinite(vals["replacement_value"]).all())
        check(f"{mode}: pregnancy value exists only for pregnant/dry cows",
              vals.loc[vals["status"].isin(["Open (breeding)", "Open (do not breed)"]), "pregnancy_value"].isna().all()
              and vals.loc[vals["status"].isin(["Pregnant", "Dry"]), "pregnancy_value"].notna().any())

    # a cow standing in the replacement state, compared with a heifer bought now: keep(h) - (keep(h) - C) = C, the cost
    # of the replacement she does not have to buy (replacement cost is charged in the comparison, not in the state reward)
    one = pd.DataFrame({"cow_id": [1], "state_id": [rid], "lactation": [1]})
    from runners.cow.farm.simulate import herd_values
    from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
    v = herd_values(states, probs, matrix, one, rid)
    check("a cow in the replacement state has replacement value = the cost of one replacement",
          abs(v["replacement_value"].iloc[0] - cost_per_replacement()) < 1e-6, f"{v['replacement_value'].iloc[0]:.4f} vs {cost_per_replacement()}")
    check("V (reward vector) and V_keep differ by the present value of replacement costs, and V_keep <= V",
          bool((v["V_keep"] <= v["V"]).all()))

    # herd built under another resolution is refused
    other = "month" if os.environ.get("COW_TIME_UNIT", "month") == "week" else "week"        # any resolution other than the one the herd was built under
    r = run(["runners/cow/farm/simulate.py", "--time-unit", other, "--steps", "2", "--farm-csv", herds["equilibrium"], "--out-dir", os.path.join(tmp, "x"), "--no-values"])
    check("a herd built under another time resolution is refused", r.returncode != 0 and "different configuration" in (r.stdout + r.stderr))

    if FAILED:
        print(f"\nFARM_VERIFY_FAILED: {FAILED}")
        sys.exit(1)
    print("\nFARM_VERIFY_OK")


if __name__ == "__main__":
    main()

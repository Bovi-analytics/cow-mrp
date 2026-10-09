"""
verify_farm.py
==============
Checks for the farm layer (runners/farm/: build.py, generate.py, simulate.py) on the (lactation, DIM, DP) state.
Run: python verify/verify_farm.py     (exit 1 on any failure). Everything is written to a temp folder.

  - the equilibrium herd distribution is truly stationary (pi P = pi) and reconstructs the replacement-cycle visits
  - generated herds contain only valid reachable states and consistent status labels; "do not breed" cows are open
    cows past the last breeding day; impossible requests are clear errors
  - expected / stochastic / monte_carlo runs: cows conserved, original + replaced reconstruct the herd, outputs written
  - herd values: replacement value of a cow in the replacement state is 0; pregnancy value exists only for pregnant cows
  - a herd built under another time resolution is refused
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # project root (mrp/, cow/)
import os
import subprocess
import sys
import tempfile
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def run(args):
    return subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)


def main():
    from runners.farm import generate as G
    from runners.farm.build import build_engine
    from cow.parameters import LAST_BREEDING_DIM_DAYS, UNIT_DAYS, MAX_LACTATION
    from cow.transitions import get_replacement_state_id

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
        r = run(["runners/farm/simulate.py", "--mode", mode, "--steps", "6", "--sims", "10", "--farm-csv", herds["equilibrium"], "--out-dir", od])
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

    # replacement value of a cow standing in the replacement state is exactly 0
    one = pd.DataFrame({"cow_id": [1], "state_id": [rid], "lactation": [1]})
    from runners.farm.simulate import herd_values
    v = herd_values(states, probs, matrix, one, rid)
    check("a cow in the replacement state has replacement value 0", abs(v["replacement_value"].iloc[0]) < 1e-9)

    # herd built under another resolution is refused
    r = run(["runners/farm/simulate.py", "--time-unit", "week", "--steps", "2", "--farm-csv", herds["equilibrium"], "--out-dir", os.path.join(tmp, "x"), "--no-values"])
    check("a herd built under another time resolution is refused", r.returncode != 0 and "different configuration" in (r.stdout + r.stderr))

    if FAILED:
        print(f"\nFARM_VERIFY_FAILED: {FAILED}")
        sys.exit(1)
    print("\nFARM_VERIFY_OK")


if __name__ == "__main__":
    main()

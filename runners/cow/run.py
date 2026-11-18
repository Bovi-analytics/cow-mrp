"""
runners/cow/run.py
==================
The COW module in one command: build the cow MRP with the generic engine (mrp/) and write everything one needs to look at it.

    python runners/cow/run.py [--time-unit month|week|day] [--years 5] [--annual-discount 0.95]

OUTPUT  runners/cow/outputs/model/<unit>/
    states.csv                          one row per state: state_id, lactation, dim, dp  (dim is 0-based: MIM = dim + 1)
    transition_matrix_cow_long.csv      src_*/tgt_* state columns + probability   (also .npz and .parquet)
    reward_vector_cow.csv               the reward vector per state (total and its components)
    values.csv                          V_finite (<years> y) and V_infinite per state, discounted at --annual-discount
    manifest.json                       time unit, state/non-zero counts, replacement state, files

Plots for the cow module: python runners/cow/plots/plot_validation.py (and the other scripts in plots/).
Checks: python runners/cow/checks/verify_state_semantics.py, ... (see runners/README.md).
No model logic here: states, probabilities, rewards and transitions come from domains/cow, the numerics from mrp/.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))   # project root
from runners._shared import common
common.set_time_unit_from_argv()

import argparse
import os

import numpy as np

from cowmrp.mrp import core
from runners._shared import engine_run
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.cow.transitions import get_replacement_state_id


def main():
    if "--views" in _sys.argv:       # the local report folder views/ for this domain (runners/_shared/views.py); other options pass through, e.g. --skip validation
        from runners._shared import views
        _sys.exit(views.main(["--only", "cow"] + [x for x in _sys.argv[1:] if x != "--views"]))
    if "--scale" in _sys.argv:       # scale benchmark (runners/_shared/scale_benchmark.py); other options pass through, e.g. --quick, --sizes, --time-units
        from runners._shared import scale_benchmark
        scale_benchmark.main(["--subject", "cow"] + [x for x in _sys.argv[1:] if x != "--scale"])
        return
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) + chr(10))[0])
    ap.add_argument("--time-unit", choices=["month", "week", "day"], default=None, help="resolution (default: COW_TIME_UNIT or month)")
    ap.add_argument("--years", type=int, default=5, help="finite horizon in years")
    ap.add_argument("--annual-discount", type=float, default=0.95)
    ap.add_argument("--notebook", action="store_true", help="run the shared notebook (runners/_shared/notebook) for the cow instead of this script")
    ap.add_argument("--review", action="store_true", help="write the review pages for a domain expert (runners/_shared/review.py) instead of this script")
    a = ap.parse_args()
    if a.review:
        from runners._shared import review
        print(f"review pages: {review.build('cow')}")
        return
    if a.notebook:
        from runners._shared import run_notebook
        print(f"executed notebook: {run_notebook.run('cow')}")
        return

    model = engine_run.build("cow")
    out = common.module_outputs("cow", "model", TIME_UNIT)
    engine_run.export("cow", model, out)

    gamma = core.annual_to_per_step_gamma(a.annual_discount, UNIT_DAYS)
    V_fin = core.evaluate_finite_horizon(model.P, model.rewards["total"], gamma, a.years, UNIT_DAYS)
    V_inf = core.evaluate_infinite_horizon(model.P, model.rewards["total"], gamma)
    vals_df = model.states[["state_id", "lactation", "dim", "dp"]].copy()
    vals_df["V_finite"], vals_df["V_infinite"] = V_fin, V_inf
    vals_df.to_csv(os.path.join(out, "values.csv"), index=False)

    states = model.states
    mim = states["dim"] * UNIT_DAYS // 30 + 1                       # month in milk as a state label (MONTH steps: dim + 1)
    open_ = states["dp"] == 0
    x = "Month in milk (MIM)"
    v_open = vals_df[open_].assign(MIM=mim[open_]).pivot_table(index="MIM", columns="lactation", values="V_infinite", aggfunc="mean").add_prefix("Lactation ")
    r_open = model.rewards["total"]
    r_df = states[["lactation"]].assign(MIM=mim, total=np.asarray(r_open), open=open_)
    r_open_df = r_df[r_df["open"]].pivot_table(index="MIM", columns="lactation", values="total", aggfunc="mean").add_prefix("Lactation ")
    r_preg_df = r_df[~r_df["open"]].pivot_table(index="MIM", columns="lactation", values="total", aggfunc="mean").add_prefix("Lactation ")
    count = states.assign(MIM=mim, status=np.where(open_, "Open", "Pregnant")).pivot_table(index="MIM", columns="status", values="state_id", aggfunc="count").fillna(0)
    common.write_page(common.module_outputs("cow", "dashboard", f"cow_model_dashboard_{TIME_UNIT}.html"), "Cow Model Dashboard (Interactive)", [
        f"{TIME_UNIT.capitalize()} steps | States: <strong>{len(states):,}</strong> | Non-zero transitions: <strong>{model.P.nnz:,}</strong> | "
        f"Discount {a.annual_discount:g} per year | Horizon {a.years} years",
        "The state is (lactation, dim, dp); here dim is shown as the month in milk. Per-state values, not a herd projection (see the farm dashboard for that)."], [
        common.section("Value of an open cow by month in milk", "V (infinite horizon, discounted), open states (dp = 0), one line per lactation.",
                       common.area_chart(v_open, "Value of an open cow by month in milk", x, "USD per cow", stacked=False)),
        common.section("Reward per step by month in milk", "Total reward of one step, average over the states of that month; left open cows, right pregnant cows.",
                       common.area_chart(r_open_df, "Reward per step, open cows", x, "USD per step", stacked=False),
                       common.area_chart(r_preg_df, "Reward per step, pregnant cows", x, "USD per step", stacked=False)),
        common.section("States per month in milk", "How many (lactation, dim, dp) states exist at each month in milk (the full state table).",
                       common.area_chart(count, "States per month in milk", x, "Number of states")),
    ])

    rid = get_replacement_state_id(model.states)
    manifest = engine_run.write_model_manifest(
        "cow", model, out, unit_days=UNIT_DAYS, replacement_state=model.states.loc[rid, ["lactation", "dim", "dp"]].astype(int).tolist(),
        annual_discount=a.annual_discount, finite_years=a.years,
        V_finite_mean=float(np.mean(V_fin)), V_infinite_mean=float(np.mean(V_inf)))
    print(f"cow module ({TIME_UNIT}): {len(model.states):,} states, {model.P.nnz:,} transitions -> {out}\nmanifest: {manifest}")


if __name__ == "__main__":
    main()

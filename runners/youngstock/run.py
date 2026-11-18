"""
runners/youngstock/run.py
=========================
The YOUNGSTOCK module on its own (no cow chain, no coupling): build the calf -> heifer -> first-calving chain with the
generic engine (mrp/), check the matrix Q, and follow one birth cohort through it.

    python runners/youngstock/run.py [--time-unit month|week|day] [--dam-parity 1|2|3+] [--years 3]

OUTPUT  runners/youngstock/outputs/model/<unit>/
    states.csv                          line, stage, age, dp, age_days, absorbing, alive, state_id
    transition_matrix_youngstock_long.csv   (+ .parquet, .npz)   source/target state columns + probability (rows sum to 1)
    cohort_fates.csv                    per step: share of a birth cohort in each outcome (alive, GRADUATED, SOLD, DIED, CULLED, STILLBORN)
    births_by_state.csv                 the birth mix used as the cohort start (per birth event of a dam of the given parity)
    fates.png                           the cohort curves
    manifest.json                       time unit, counts, parameters used

The cohort start is the birth mix (twins and stillbirths included), so the curves read "per birth event".
The coupling with the cow chain (calving events, replacement heifers, economics) is the HERD module: runners/herd/run.py.
"""
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root
from runners._shared import common
common.set_time_unit_from_argv()

import argparse

import numpy as np
import pandas as pd
from cowmrp.mrp import core
from runners._shared import engine_run
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.youngstock import parameters as YP

OUTCOMES = ["GRADUATED", "SOLD", "DIED", "CULLED", "STILLBORN"]


def main():
    if "--views" in _sys.argv:       # the local report folder views/ for this domain (runners/_shared/views.py); other options pass through, e.g. --skip validation
        from runners._shared import views
        _sys.exit(views.main(["--only", "youngstock"] + [x for x in _sys.argv[1:] if x != "--views"]))
    if "--scale" in _sys.argv:       # scale benchmark (runners/_shared/scale_benchmark.py); other options pass through, e.g. --quick, --sizes, --time-units
        from runners._shared import scale_benchmark
        scale_benchmark.main(["--subject", "youngstock"] + [x for x in _sys.argv[1:] if x != "--scale"])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--time-unit", choices=["month", "week", "day"], default=None, help="resolution (default: COW_TIME_UNIT or month)")
    ap.add_argument("--dam-parity", choices=["1", "2", "3+"], default="2", help="parity of the dam whose birth mix starts the cohort")
    ap.add_argument("--years", type=float, default=3.0, help="how long to follow the cohort")
    ap.add_argument("--notebook", action="store_true", help="run the shared notebook (runners/_shared/notebook) for youngstock instead of this script")
    ap.add_argument("--review", action="store_true", help="write the review pages for a domain expert (runners/_shared/review.py) instead of this script")
    a = ap.parse_args()
    if a.review:
        from runners._shared import review
        print(f"review pages: {review.build('youngstock')}")
        return
    if a.notebook:
        from runners._shared import run_notebook
        print(f"executed notebook: {run_notebook.run('youngstock')}")
        return

    model = engine_run.build("youngstock")
    ys, yp, Q = model.states, model.probs, model.P
    out = common.module_outputs("youngstock", "model", TIME_UNIT)
    engine_run.export("youngstock", model, out, state_cols=["line", "stage", "age", "dp", "state_id"])

    births = np.asarray(yp["birth_mix"][a.dam_parity], float)
    pi0 = births / births.sum()
    pd.DataFrame({"state_id": ys["state_id"], "line": ys["line"], "stage": ys["stage"], "age": ys["age"],
                  "births_per_event": births}).query("births_per_event > 0").to_csv(os.path.join(out, "births_by_state.csv"), index=False)

    n_steps = int(round(a.years * 365 / UNIT_DAYS))
    stage = ys["stage"].values
    pi, hist, by_group = pi0.copy(), [], []
    alive_group = (ys["line"] + " " + ys["stage"]).values
    for t in range(n_steps + 1):
        row = {"step": t, "days": t * UNIT_DAYS, "alive": float(pi[ys["alive"].values].sum())}
        row.update({o: float(pi[stage == o].sum()) for o in OUTCOMES})
        hist.append(row)
        by_group.append(pd.Series(pi[ys["alive"].values]).groupby(alive_group[ys["alive"].values]).sum())
        pi = core.distribution_recursive(Q, pi, 1)
    curves = pd.DataFrame(hist)
    curves.to_csv(os.path.join(out, "cohort_fates.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col in ["alive"] + OUTCOMES:
        ax.plot(curves["days"], curves[col], label=col.lower())
    ax.set_xlabel("days since birth"); ax.set_ylabel("share of the birth cohort")
    ax.set_title(f"Youngstock cohort fates (dam parity {a.dam_parity}, {TIME_UNIT} steps)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fates.png"), dpi=150); plt.close(fig)

    days = curves.set_index("days")
    live = pd.DataFrame(by_group).fillna(0.0).set_axis(days.index)
    x, y = "Days since birth", "Share of the birth cohort"
    births = pd.read_csv(os.path.join(out, "births_by_state.csv"))
    births = births.groupby(["line", "stage"], as_index=False)["births_per_event"].sum()
    common.write_page(common.module_outputs("youngstock", "dashboard", "youngstock_dashboard.html"), "Youngstock Cohort Projection (Interactive)", [
        f"Dam parity: <strong>{a.dam_parity}</strong> | {TIME_UNIT.capitalize()} steps | Follow-up: <strong>{a.years:g}</strong> years | "
        f"States: <strong>{len(ys)}</strong> | Breeding program: <strong>{YP.BREEDING_PROGRAM}</strong>",
        "One birth event, twins and stillbirths included. Expected shares, not a validated forecast. Hover for details; click a legend entry to toggle it."], [
        common.section("Where the cohort is", "Share of the birth cohort alive in each line and stage, plus the outcomes already reached.",
                       common.area_chart(days[["alive"] + OUTCOMES], "Cohort fates over time", x, y)),
        common.section("Live animals by line and stage", "Calves, growing and breeding heifers, pregnant heifers, by dairy / beef line.",
                       common.area_chart(live, "Live cohort by line and stage", x, y)),
        common.section("Birth mix", "What starts the cohort (per birth event of a dam of this parity).", births.round(4).to_html(index=False)),
    ])

    engine_run.write_model_manifest("youngstock", model, out, unit_days=UNIT_DAYS,
                          dam_parity=a.dam_parity, cohort_years=a.years, breeding_program=YP.BREEDING_PROGRAM,
                          final_shares={k: round(float(v), 6) for k, v in curves.iloc[-1].items() if k in ["alive"] + OUTCOMES})
    print(f"youngstock module ({TIME_UNIT}): {len(ys)} states, {Q.nnz} transitions -> {out}")
    print(curves.iloc[-1].round(4).to_string())


if __name__ == "__main__":
    main()

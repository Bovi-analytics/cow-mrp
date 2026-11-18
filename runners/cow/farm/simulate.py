"""
runners/cow/farm/simulate.py
========================
Projects a herd forward with the MRP and reports where it goes, what it is worth, and how much of it has been
replaced.

    python runners/cow/farm/simulate.py [--mode expected|stochastic|monte_carlo] [--steps 24] [--seed 42] [--sims 100]
                                    [--farm-csv runners/cow/outputs/farm/farm_data.csv] [--groupers status,lactation,dim_stage]
                                    [--compare-methods] [--no-values] [--time-unit month|week|day] [--out-dir DIR]

WHAT IT DOES
  * reads the herd CSV (generate.py) and refuses one built under a different cow configuration;
  * projects it with the MRP's own transition matrix: `expected` (exact distribution evolution, with the herd split
    into cows never replaced vs replaced at least once -- the split reconstructs the whole distribution exactly),
    `stochastic` (one sampled path per cow) or `monte_carlo` (many sampled paths: mean / std / 5-95 %);
  * summarises by cow groupings (reproductive status Open / Pregnant / Dry, lactation, DIM stage);
  * links the herd to the model's VALUES (same infinite-horizon solve as the MRP): per cow the value V, the
    REPLACEMENT value V(cow) - V(replacement state), and for pregnant cows the PREGNANCY value
    V(cow) - V(same lactation and DIM, open). Read-only diagnostics -- never added to a reward;
  * writes CSVs and an HTML dashboard (matplotlib images embedded; + an interactive Plotly page if plotly is installed).

No model logic lives here: states, probabilities, transitions, rewards and values come from the package. The projection is
cowmrp.domains.cow.farm.project_herd and the dashboards are cowmrp.viz.farm; this script keeps the herd-file check
(`load_herd`), the arguments and the printed summary.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root
from runners.cow.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.cow.farm import (project_herd, GROUPERS, DEFAULT_GROUPERS, group_probs, group_counts, mc_summary,          # noqa: F401  (moved
                                     averaged_group_counts, herd_values as _herd_values)                                  #  to the package)
from cowmrp.viz.farm import write_farm_outputs
from cowmrp.domains.cow.rewards import compute_rewards
from cowmrp.model import Model, build_model
from runners.cow.farm.build import current_config, FARM_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)


def herd_values(states, probs, matrix, farm_df, rid, gamma_annual=0.95):
    """Kept for callers of the previous version of this script; see cowmrp.domains.cow.farm.herd_values (rid is derived there)."""
    return _herd_values(Model(states, probs, compute_rewards(states, probs), matrix), farm_df, gamma_annual)


def load_herd(farm_csv, num_states):
    meta_path = os.path.splitext(farm_csv)[0] + ".meta.json"
    live = {**current_config(), "num_states": num_states}
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found -- cannot verify the herd matches the current cow configuration. Regenerate with generate.py.")
    with open(meta_path) as f:
        fm = json.load(f)
    bad = {k: (fm.get(k), v) for k, v in live.items() if fm.get(k) != v}
    if bad:
        raise SystemExit(f"{farm_csv} was generated under a different configuration:\n" +
                         "\n".join(f"  {k}: herd has {a!r}, current is {b!r}" for k, (a, b) in bad.items()) +
                         "\nRegenerate it with generate.py under the same --time-unit (state ids are not comparable across configurations).")
    return pd.read_csv(farm_csv), fm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHAT IT DOES")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="expected", choices=["expected", "stochastic", "monte_carlo"])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--farm-csv", default=os.path.join(FARM_DIR, "farm_data.csv"))
    ap.add_argument("--groupers", default=",".join(DEFAULT_GROUPERS))
    ap.add_argument("--compare-methods", action="store_true")
    ap.add_argument("--no-values", action="store_true")
    ap.add_argument("--time-unit", default=None)
    ap.add_argument("--out-dir", default=os.path.join(FARM_DIR, "results"))
    a = ap.parse_args()
    groupers = a.groupers.split(",")
    os.makedirs(a.out_dir, exist_ok=True)

    model = build_model("cow")
    farm_df, fmeta = load_herd(a.farm_csv, len(model.states))
    print(f"{len(farm_df):,} cows | {len(model.states):,} states | {TIME_UNIT} ({UNIT_DAYS} d/step) | {a.mode}, {a.steps} steps = {a.steps * UNIT_DAYS} days")
    result = project_herd(farm_df, model, a.mode, a.steps, a.seed, a.sims, groupers, values=not a.no_values, compare=a.compare_methods)
    files = write_farm_outputs(result, a.out_dir)
    values_df = result["values"]
    if values_df is not None:
        print(f"Herd V {values_df['V'].sum():,.0f} | replacement value {values_df['replacement_value'].sum():,.0f} | "
              f"pregnancy value (pregnant cows) {np.nansum(values_df['pregnancy_value']):,.0f}")
    for g in groupers:
        print(f"\n--- {g} ---")
        print(pd.DataFrame({"current": result["current"][g], "predicted": result["predicted"][g]}).fillna(0).round(1).to_string())
    plotly_ok = any(f.endswith("dashboard_plotly.html") for f in files)
    print(f"\nOutputs in {a.out_dir}  (dashboard.html{' + dashboard_plotly.html' if plotly_ok else ''})")


if __name__ == "__main__":
    main()

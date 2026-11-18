"""
runners/cow/farm/generate.py
========================
Generates a herd CSV for the farm layer. The herd is described in the model's own
terms -- lactation, days in milk, days pregnant (REAL DAYS; converted to model steps by the mid-step rule in
cow/parameters.py) -- never health or DNB flags: those are not state variables. Reproductive status is read off
dp; "do not breed" is a REGION (open, past the last breeding day), expressed as a dim_days range.

    python runners/cow/farm/generate.py [--n 10000] [--seed 42] [--mode equilibrium|mix|exact]
                                    [--time-unit month|week|day] [--out runners/cow/outputs/farm/farm_data.csv]

MODES
  equilibrium  (default) cows are drawn from the model's OWN steady-state herd: the stationary distribution of the
               closed chain (replacement closes the herd), i.e. a herd that is neither growing nor ageing.
  mix          the HERD_MIX list below: category proportions with lactation / DIM / DP ranges in real days; a cow is
               drawn uniformly among the model states that satisfy the category (an empty category is a clear error).
  exact        the EXACT_STATES list below: (lactation, dim_days, dp_days, count).

(The state mapping and the herd draws live in cowmrp.domains.cow.farm; this script keeps the editable herd lists and writes the files.)

OUTPUT  CSV with cow_id, state_id, lactation, uim, up, dim_days, dp_days, status  (+ a .meta.json sidecar recording
the cow config the state_ids belong to; simulate.py refuses a herd built under a different configuration).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root
from runners.cow.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import json
import os
import argparse

import pandas as pd

from cowmrp.domains.cow.parameters import UNIT_DAYS, MAX_LACTATION, LAST_BREEDING_DIM_DAYS
from cowmrp.domains.cow.farm import (generate_herd, status_of, steady_state, sample_mix, exact_ids,        # noqa: F401  (moved to the package;
                                     DEFAULT_HERD_MIX, DEFAULT_EXACT_STATES)                               #  re-exported for older callers)
from cowmrp.model import Model
from runners.cow.farm.build import build_engine, current_config, FARM_DIR

# ============================================================
# USER CONFIGURATION
# ============================================================
# status: "open" | "pregnant" | "dry" | "dnb" (open AND past the last breeding day -- a region, not a state)
# ranges are inclusive, in REAL DAYS (dim_days, dp_days); lactation is [min, max].
HERD_MIX = [   # editable copy of cowmrp.domains.cow.farm.DEFAULT_HERD_MIX
    {"proportion": 0.30, "status": "open",     "lactation": [1, 1],             "dim_days": [0, 150]},
    {"proportion": 0.12, "status": "open",     "lactation": [2, MAX_LACTATION], "dim_days": [0, 150]},
    {"proportion": 0.06, "status": "dnb",      "lactation": [1, MAX_LACTATION], "dim_days": [LAST_BREEDING_DIM_DAYS + 1, 400]},
    {"proportion": 0.36, "status": "pregnant", "lactation": [1, MAX_LACTATION], "dp_days": [30, 215]},
    {"proportion": 0.16, "status": "dry",      "lactation": [2, MAX_LACTATION], "dp_days": [222, 277]},
]
EXACT_STATES = [      # (lactation, dim_days, dp_days, count); editable copy of DEFAULT_EXACT_STATES
    (1, 60, 0, 200), (2, 120, 0, 150), (2, 200, 100, 300), (3, 280, 230, 100),
]


def generate(states, matrix, mode, n_cows, seed, out_path):
    """Generate a herd and write it (CSV + .meta.json). The mapping and the draws are cowmrp.domains.cow.farm.generate_herd (moved from here);
    HERD_MIX and EXACT_STATES above are passed to it, so editing them here still works."""
    df = generate_herd(Model(states, None, None, matrix), mode, n_cows, seed, mix=HERD_MIX, exact=EXACT_STATES)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(os.path.splitext(out_path)[0] + ".meta.json", "w") as f:
        json.dump({**current_config(), "num_states": int(len(states)), "mode": mode, "seed": seed}, f, indent=2)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("OUTPUT")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="equilibrium", choices=["equilibrium", "mix", "exact"])
    ap.add_argument("--time-unit", default=None)
    ap.add_argument("--out", default=os.path.join(FARM_DIR, "farm_data.csv"))
    a = ap.parse_args()
    states, probs, matrix = build_engine()
    print(f"State space: {len(states):,} states ({current_config()['time_unit']}, {UNIT_DAYS} d/step)")
    df = generate(states, matrix, a.mode, a.n, a.seed, a.out)
    print(f"Generated {len(df)} cows ({a.mode}) -> {a.out}")
    print(df["status"].value_counts().to_string())
    print(df["lactation"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

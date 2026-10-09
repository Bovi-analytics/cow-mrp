"""
runners/farm/generate.py
========================
Generates a herd CSV for the farm layer. (Formerly farm_generator.py.) The herd is described in the model's own
terms -- lactation, days in milk, days pregnant (REAL DAYS; converted to model steps by the mid-step rule in
cow/parameters.py) -- never health or DNB flags: those are not state variables. Reproductive status is read off
dp; "do not breed" is a REGION (open, past the last breeding day), expressed as a dim_days range.

    python runners/farm/generate.py [--n 1000] [--seed 42] [--mode equilibrium|mix|exact]
                                    [--time-unit month|week|day] [--out outputs/farm/farm_data.csv]

MODES
  equilibrium  (default) cows are drawn from the model's OWN steady-state herd: the stationary distribution of the
               closed chain (replacement closes the herd), i.e. a herd that is neither growing nor ageing.
  mix          the HERD_MIX list below: category proportions with lactation / DIM / DP ranges in real days; a cow is
               drawn uniformly among the model states that satisfy the category (an empty category is a clear error).
  exact        the EXACT_STATES list below: (lactation, dim_days, dp_days, count).

OUTPUT  CSV with cow_id, state_id, lactation, uim, up, dim_days, dp_days, status  (+ a .meta.json sidecar recording
the cow config the state_ids belong to; simulate.py refuses a herd built under a different configuration).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))   # project root
from runners.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import json
import os
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cow.parameters import UNIT_DAYS, dim_step_of_days, dp_step_of_days, DRY_OFF_DP, LAST_BREEDING_DIM_DAYS, MAX_LACTATION
from cow.states import repro_status
from cow.transitions import get_replacement_state_id
from runners.farm.build import build_engine, current_config, FARM_DIR

# ============================================================
# USER CONFIGURATION
# ============================================================
# status: "open" | "pregnant" | "dry" | "dnb" (open AND past the last breeding day -- a region, not a state)
# ranges are inclusive, in REAL DAYS (dim_days, dp_days); lactation is [min, max].
HERD_MIX = [
    {"proportion": 0.30, "status": "open",     "lactation": [1, 1],             "dim_days": [0, 150]},
    {"proportion": 0.12, "status": "open",     "lactation": [2, MAX_LACTATION], "dim_days": [0, 150]},
    {"proportion": 0.06, "status": "dnb",      "lactation": [1, MAX_LACTATION], "dim_days": [LAST_BREEDING_DIM_DAYS + 1, 400]},
    {"proportion": 0.36, "status": "pregnant", "lactation": [1, MAX_LACTATION], "dp_days": [30, 215]},
    {"proportion": 0.16, "status": "dry",      "lactation": [2, MAX_LACTATION], "dp_days": [222, 277]},
]
EXACT_STATES = [      # (lactation, dim_days, dp_days, count)
    (1, 60, 0, 200), (2, 120, 0, 150), (2, 200, 100, 300), (3, 280, 230, 100),
]


def status_of(states):
    """Reproductive status label per state: open / pregnant / dry, plus 'dnb' for open cows past the last breeding day."""
    s = np.char.lower(repro_status(states["dp"].values).astype(str))
    age = (states["dim"].values + 1) * UNIT_DAYS                       # mean age during the step (mid-step convention)
    return np.where((s == "open") & (age > LAST_BREEDING_DIM_DAYS), "dnb", s)


def steady_state(states, matrix, rid):
    """Stationary distribution of the closed herd: expected visits per replacement-to-replacement cycle, normalised."""
    n = len(states)
    Q = matrix.tocsr().copy().tolil()
    Q[:, rid] = 0
    e = np.zeros(n)
    e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)
    return np.clip(x, 0, None) / x.sum()


def sample_mix(states, mix, n_cows, rng):
    status = status_of(states)
    total = sum(m["proportion"] for m in mix)
    counts = [int(round(m["proportion"] / total * n_cows)) for m in mix]
    counts[0] += n_cows - sum(counts)
    ids = []
    for m, k in zip(mix, counts):
        mask = status == m["status"]
        lo, hi = m.get("lactation", [1, MAX_LACTATION])
        mask &= (states["lactation"].values >= lo) & (states["lactation"].values <= hi)
        if "dim_days" in m:
            dlo, dhi = m["dim_days"]
            mid = (states["dim"].values + 1) * UNIT_DAYS
            mask &= (mid >= dlo) & (mid <= dhi)
        if "dp_days" in m:
            plo, phi = m["dp_days"]
            mid = states["dp"].values * UNIT_DAYS
            mask &= (mid >= plo) & (mid <= phi)
        pool = states["state_id"].values[mask]
        if len(pool) == 0:
            raise ValueError(f"No model state satisfies the herd category {m} at this time resolution "
                             f"(the state space has only reachable states -- loosen the ranges).")
        ids.extend(rng.choice(pool, size=k, replace=True))
    return np.array(ids)


def exact_ids(states, entries):
    ids = []
    for lac, dim_days, dp_days, count in entries:
        q = (states["lactation"] == lac) & (states["dim"] == dim_step_of_days(dim_days)) & (states["dp"] == dp_step_of_days(dp_days))
        if not q.any():
            raise ValueError(f"EXACT_STATES entry (lactation={lac}, dim_days={dim_days}, dp_days={dp_days}) is not a "
                             f"reachable model state (dim step {dim_step_of_days(dim_days)}, dp step {dp_step_of_days(dp_days)}).")
        ids.extend([int(states.loc[q, "state_id"].iloc[0])] * count)
    return np.array(ids)


def generate(states, matrix, mode, n_cows, seed, out_path):
    rng = np.random.default_rng(seed)
    if mode == "equilibrium":
        pi = steady_state(states, matrix, get_replacement_state_id(states))
        ids = rng.choice(len(states), size=n_cows, p=pi)
    elif mode == "mix":
        ids = sample_mix(states, HERD_MIX, n_cows, rng)
    elif mode == "exact":
        ids = exact_ids(states, EXACT_STATES)
    else:
        raise ValueError(f"Unknown mode {mode!r} -- use equilibrium | mix | exact")
    rows = states.set_index("state_id").loc[ids].reset_index()
    df = pd.DataFrame({
        "cow_id": np.arange(1, len(ids) + 1), "state_id": ids,
        "lactation": rows["lactation"].values,
        "uim": rows["dim"].values, "up": rows["dp"].values,                      # resolution-neutral step columns
        "dim_days": (rows["dim"].values + 1) * UNIT_DAYS, "dp_days": rows["dp"].values * UNIT_DAYS,
        "status": status_of(rows),
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(os.path.splitext(out_path)[0] + ".meta.json", "w") as f:
        json.dump({**current_config(), "num_states": int(len(states)), "mode": mode, "seed": seed}, f, indent=2)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("OUTPUT")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
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

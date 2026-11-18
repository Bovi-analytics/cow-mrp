"""
runners/cow/profile_build.py
========================
Where the time goes when the cow model is built and solved, and what is cheap to store for a small farm PC.
Each phase is timed (wall s, peak RAM) with the same Phase meter as scale_benchmark.py; then the build products are
written to disk and read back, so "rebuild" and "load from cache" can be compared.

    python runners/cow/profile_build.py --time-unit month|week|day [--years 5] [--out-dir DIR]

PHASES: states, probabilities, transitions, assemble, validate, rewards, backward_solve (one V for every state over
--years), then CACHE: save / load of matrix + reward + V (compressed npz) with file sizes.
OUTPUT: runners/cow/outputs/profile_build/profile_<unit>.csv (one row per phase). Nothing here changes the model.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))   # project root
from runners.cow.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import argparse
import contextlib
import io
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz

from cowmrp.mrp import core
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.cow.states import build_cow_states
from cowmrp.domains.cow.probabilities import compute_probabilities
from cowmrp.domains.cow.transitions import compute_transitions
from cowmrp.domains.cow.rewards import compute_rewards
from runners._shared.scale_benchmark import Phase

ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from runners._shared import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-unit", default=None, help="handled before import (month|week|day)")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--gamma-annual", type=float, default=0.95)
    ap.add_argument("--out-dir", default=common.module_outputs("cow", "profile_build"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows, hold = [], {}

    def phase(name, fn):
        p = Phase()
        with p:
            with contextlib.redirect_stdout(io.StringIO()):
                out = fn()
        rows.append(dict(phase=name, **p.row()))
        return out

    states = phase("states", build_cow_states)
    probs = phase("probabilities", lambda: compute_probabilities(states))
    rcv = phase("transitions", lambda: compute_transitions(states, probs))
    matrix = phase("assemble", lambda: core.assemble_matrix(states, *rcv))
    phase("validate", lambda: core.validate_row_stochasticity(matrix, states))
    reward = np.asarray(phase("rewards", lambda: compute_rewards(states, probs))["total"])
    V = phase("backward_solve", lambda: core.evaluate_finite_horizon(matrix, reward, a.gamma_annual, a.years, UNIT_DAYS))
    V = np.asarray(V["V"] if isinstance(V, dict) else V)
    hold["V"] = V

    path = os.path.join(tempfile.mkdtemp(prefix="cowmrp_cache_"), "model")
    os.makedirs(path)
    mp, rp, vp = (os.path.join(path, n) for n in ("matrix.npz", "reward.npy", "V.npy"))
    phase("cache_save", lambda: (save_npz(mp, matrix), np.save(rp, reward), np.save(vp, V)))
    size = {n: os.path.getsize(f) / 2**20 for n, f in (("matrix", mp), ("reward", rp), ("V", vp))}
    phase("cache_load", lambda: (load_npz(mp), np.load(rp), np.load(vp)))

    df = pd.DataFrame(rows)
    total_build = df[df["phase"].isin(["states", "probabilities", "transitions", "assemble", "validate", "rewards"])]["wall_s"].sum()
    out = os.path.join(a.out_dir, f"profile_{TIME_UNIT}.csv")
    df.to_csv(out, index=False)
    print(f"{TIME_UNIT} ({UNIT_DAYS} d/step): {len(states):,} states, {matrix.nnz:,} nnz, {a.years} y")
    print(df.round(3).to_string(index=False))
    print(f"build total {total_build:.2f} s | cache files: " + ", ".join(f"{k} {v:.2f} MB" for k, v in size.items()))
    print(f"written to {out}")


if __name__ == "__main__":
    main()

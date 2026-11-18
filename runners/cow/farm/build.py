"""
runners/cow/farm/build.py
=====================
Builds the cow MRP from domains.cow/'s LIVE configuration for the farm layer, and saves / loads it as a small on-disk
artifact that a herd can be projected against. Independent of runners/cow/config.py's
benchmark models: "farm" always means "the current cow model".

    python runners/cow/farm/build.py [--time-unit month|week|day]

ARTIFACT (runners/cow/outputs/farm/model/):
    states.csv             the (pruned, reachable-only) state table; state_id == row index
    transition_matrix.npz  scipy.sparse matrix
    metadata.json          {time_unit, unit_days, max_lactation, max_dim_days, max_dp_days, num_states, built_at}
                           -- used for staleness checks and to refuse a herd file built under another state space
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root


def set_time_unit_from_argv(argv=None):
    """--time-unit must be applied BEFORE cow.parameters is imported (it reads COW_TIME_UNIT once)."""
    argv = _sys.argv if argv is None else argv
    if "--time-unit" in argv:
        _os.environ["COW_TIME_UNIT"] = argv[argv.index("--time-unit") + 1]


set_time_unit_from_argv()

import json
import os
from datetime import datetime, timezone

from scipy.sparse import save_npz

from cowmrp.model import build_model
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS, MAX_LACTATION, MAX_DIM_DAYS, MAX_DP_DAYS

ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from runners._shared import common
FARM_DIR = common.module_outputs("cow", "farm")
MODEL_DIR = os.path.join(FARM_DIR, "model")
STATES_FILENAME, MATRIX_FILENAME, METADATA_FILENAME = "states.csv", "transition_matrix.npz", "metadata.json"


def current_config():
    """Plain dict of the live cow/parameters.py configuration (no cow objects), comparable with a saved metadata.json."""
    return {"time_unit": TIME_UNIT, "unit_days": UNIT_DAYS, "max_lactation": MAX_LACTATION,
            "max_dim_days": MAX_DIM_DAYS, "max_dp_days": MAX_DP_DAYS}


def build_engine():
    """(states, probs, matrix) for the live config. Row-stochasticity is validated (a hard error, never normalized)."""
    model = build_model("cow")          # cowmrp.model: states -> probabilities -> rewards -> matrix, strict row check
    return model.states, model.probs, model.P


def save_artifact(states, matrix, output_dir=None):
    output_dir = output_dir or MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)
    states.to_csv(os.path.join(output_dir, STATES_FILENAME), index=False)
    save_npz(os.path.join(output_dir, MATRIX_FILENAME), matrix)
    meta = {**current_config(), "num_states": int(len(states)), "built_at": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(output_dir, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_artifact(output_dir=None):
    states, probs, matrix = build_engine()
    return save_artifact(states, matrix, output_dir)


def artifact_exists(output_dir=None):
    output_dir = output_dir or MODEL_DIR
    return all(os.path.exists(os.path.join(output_dir, fn)) for fn in (STATES_FILENAME, MATRIX_FILENAME, METADATA_FILENAME))


def load_metadata(output_dir=None):
    path = os.path.join(output_dir or MODEL_DIR, METADATA_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_stale(output_dir=None):
    """True if the saved artifact was built under a different cow config; None if no artifact exists."""
    meta = load_metadata(output_dir)
    if meta is None:
        return None
    return any(meta.get(k) != v for k, v in current_config().items())


if __name__ == "__main__":
    print(f"Building the farm model from the live cow config: {current_config()}")
    meta = build_artifact()
    print(f"States: {meta['num_states']:,}  |  time unit: {meta['time_unit']} ({meta['unit_days']} d/step)  |  saved to {MODEL_DIR}")

"""
runners/_shared/engine_run.py
=============================
The one place that builds a domain's MRP with the generic engine (mrp/) and writes the standard model outputs. Every
module's run.py calls it, so a fix or an engine speed-up is made once and every domain gets it.

A domain `<name>` provides (see runners/_shared/new_domain.py):
    domains/<name>/states.py          build_<name>_states()                -> DataFrame with state_id
    domains/<name>/probabilities      compute_probabilities(states)        -> dict
    domains/<name>/transitions.py     compute_transitions(states, probs)   -> rows, cols, vals
    domains/<name>/rewards            compute_rewards(states, probs)       -> dict with "total"   (optional)

    build(name)                         -> Model(states, probs, rewards, P)   P checked row-stochastic
    export(name, model, out, ...)       writes states, long matrix (csv, parquet), npz, reward vector
    write_model_manifest(...)           manifest.json with counts + matrix checksum (+ extra fields)
    run_domain(name, time_unit=None)    build + export to runners/<name>/outputs/model/<unit>/ + manifest, returns (model, out)

Nothing here changes a model or mrp/. The time unit must be set before the first domain import (run.py does this).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners._shared import common
# the build and export logic live in the package (cowmrp.model, cowmrp.io); this file only adds the runner-side output folders and manifest
from cowmrp.domain import load_domain
from cowmrp.model import Model, build_model, matrix_checksum, row_sum_report   # noqa: F401  (re-exported for the runners)
from cowmrp.io import export_model


def build(name, strict=True):
    """strict=True (default): every row must sum to 1 or the build stops (cowmrp.mrp.core.validate_row_stochasticity). strict=False is a
    call-site opt-out for a domain with known rows that do not (see its runner); the count is then written to the manifest."""
    return build_model(load_domain(name), strict)


def export(name, model, out, state_cols=None):
    """Standard files: states.csv, transition_matrix_<name>_long.{csv,parquet}, transition_matrix_<name>.npz, reward_vector_<name>.csv."""
    export_model(model, load_domain(name), out, state_cols=state_cols)


def write_model_manifest(name, model, out, **extra):
    """manifest.json beside the model files: counts, row-sum report and the matrix checksum (what verify_baseline.py compares), plus `extra`."""
    extra = {**row_sum_report(model.P), **extra}
    return common.write_manifest(out, name, time_unit=os.environ.get("COW_TIME_UNIT", "month"), num_states=len(model.states),
                                 nnz_transitions=int(model.P.nnz), matrix_checksum=matrix_checksum(model.P), **extra)


def run_domain(name, time_unit=None, state_cols=None, strict=True, **manifest_extra):
    """build + export + manifest for a domain with nothing extra to compute. Returns (model, out_dir)."""
    if time_unit:
        os.environ["COW_TIME_UNIT"] = time_unit
    model = build(name, strict)
    out = common.module_outputs(name, "model", os.environ.get("COW_TIME_UNIT", "month"))
    export(name, model, out, state_cols)
    write_model_manifest(name, model, out, **manifest_extra)
    return model, out

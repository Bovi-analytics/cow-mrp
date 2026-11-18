"""Write a built model to disk: states, long-format transition matrix, sparse matrix, reward vector."""
from __future__ import annotations

import os

from cowmrp.domain import load_domain
from cowmrp.mrp import core


def export_model(model, dom, out_dir: str, state_cols=None, csv: bool = True, parquet: bool = True, npz: bool = True) -> list[str]:
    """Standard files in ``out_dir``: ``states.csv``, ``transition_matrix_<name>_long.{csv,parquet}``, ``transition_matrix_<name>.npz``,
    ``reward_vector_<name>.csv`` (when the domain has rewards). Column names of the state variables come from the domain
    (``EXPORT_COLUMN_NAMES`` hook); the engine itself labels nothing. ``csv``/``parquet``/``npz`` switch formats off (the long table is
    the slow and large one at day resolution; parquet needs the optional ``pyarrow``). Returns the paths written.
    """
    from scipy.sparse import save_npz
    dom = load_domain(dom)
    os.makedirs(out_dir, exist_ok=True)
    names, written = dom.export_column_names, []

    def path(fn):
        written.append(os.path.join(out_dir, fn))
        return written[-1]

    model.states.to_csv(path("states.csv"), index=False)
    if csv or parquet:
        long = core.export_transitions(model.P, model.states if state_cols is None else model.states[state_cols], column_names=names)
        if csv:
            long.to_csv(path(f"transition_matrix_{dom.name}_long.csv"), index=False)
        if parquet:
            long.to_parquet(path(f"transition_matrix_{dom.name}_long.parquet"), index=False)
    if npz:
        save_npz(path(f"transition_matrix_{dom.name}.npz"), model.P)
    if model.rewards is not None:
        core.export_rewards(model.states, model.rewards, column_names=names).to_csv(path(f"reward_vector_{dom.name}.csv"), index=False)
    return written

"""
domains/cow/notebook_hooks.py
=============================
What the shared notebook (runners/_shared/notebook/mrp_engine.ipynb) needs from the cow beyond the standard domain files.
Every name is optional; the notebook skips the step that needs a missing one. No model logic: these only point at what
domains/cow already defines.

    INITIAL_STATE                   dict of state columns: the state the distribution and the simulations start from
    STRICT                          True: a row that does not sum to 1 stops the build (default True)
    gamma_per_step(gamma_annual)    per-step discount factor (default: cowmrp.mrp.core.annual_to_per_step_gamma with the domain's UNIT_DAYS)
    replacement_state_id(states)    state id exits are sent to (enables the replacement-aware step)
    group_labels(states)            one label per state, for the grouped Monte Carlo summary and the distribution chart
    GROUP_NAME                      what those labels are
"""
from cowmrp.domains.cow.states import repro_status
from cowmrp.domains.cow.transitions import get_replacement_state_id as replacement_state_id   # noqa: F401  (name read by the notebook)

INITIAL_STATE = {"lactation": 1, "dim": 0, "dp": 0}
STRICT = True
GROUP_NAME = "Reproductive status (from dp)"


def group_labels(states):
    return repro_status(states["dp"].values)


# Exported column names for the state variables (used by cowmrp.io.export_model): resolution-neutral time names.
EXPORT_COLUMN_NAMES = {"dim": "uim", "dp": "up"}

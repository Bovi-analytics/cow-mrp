"""
runners/youngstock/scale_inputs.py
Adapter for runners/_shared/scale_benchmark.py: youngstock as an outcome subject (no rewards). Entity = one newborn calf
from the birth mix of a parity-2 dam; compared on the share of the cohort ending in each outcome at the horizon.
"""
RESOLUTIONS = ("month", "week", "day")
NOTES = ("Entity: one newborn calf (birth mix of a parity-2 dam, twins and stillbirths included). No rewards: accuracy is the "
         "share of the cohort in each outcome (graduated, sold, died, culled, stillborn, still alive) at the horizon, scale 1.")


def inputs(args):
    import numpy as np
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from runners._shared import engine_run
    model = engine_run.build("youngstock")
    ys = model.states
    labels = np.where(ys["alive"].values, "alive", ys["stage"].str.lower().values)
    births = np.asarray(model.probs["birth_mix"]["2"], float)
    return dict(kind="outcomes", states=ys, matrix=model.P, labels=labels, unit_days=UNIT_DAYS, start_distribution=births / births.sum())

"""REQUIRED: compute_probabilities(states) -> dict of arrays, one value per state (0 where the event cannot happen)."""
import numpy as np

from . import parameters as P


def compute_probabilities(states):
    phase, k = states["phase"].values, states["k"].values
    n = len(states)
    p_conceive, p_loss, p_calve, p_cull = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if phase[i] == "open":
            p_conceive[i] = P.CONCEPTION_BY_MONTH[k[i]]
            p_cull[i] = (1.0 - p_conceive[i]) if k[i] == P.LAST_OPEN_MONTH else 0.0     # still open after the last month: culled
        elif k[i] < P.GESTATION_MONTHS:
            p_loss[i] = P.PREGNANCY_LOSS_PER_MONTH
        else:
            p_calve[i] = 1.0                                                              # end of the last pregnant month
    return {"p_conceive": p_conceive, "p_loss": p_loss, "p_calve": p_calve, "p_cull": p_cull}

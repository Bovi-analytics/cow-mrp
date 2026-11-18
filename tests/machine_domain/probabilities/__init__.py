import numpy as np
from machine_domain import parameters as P
def compute_probabilities(states):
    n = len(states)
    wear = np.where(states["condition"] == "new", P.P_WEAR, 0.0)
    brk = np.where(states["condition"] == "worn", P.P_BREAK, 0.0)
    return {"wear": wear, "break": brk}

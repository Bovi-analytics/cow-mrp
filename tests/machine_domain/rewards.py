import numpy as np
from machine_domain import parameters as P
def compute_rewards(states, probs):
    r = states["condition"].map(P.REWARD).to_numpy(float)
    return {"income": r, "total": r}

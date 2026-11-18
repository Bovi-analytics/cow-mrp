"""OPTIONAL: compute_rewards(states, probs) -> dict of arrays that must include 'total'. Without this file the engine still builds the
transition matrix, but cannot compute values. Here each reward is the EXPECTED reward of the state for one step."""
import numpy as np

from . import parameters as P


def compute_rewards(states, probs):
    is_open = (states["phase"].values == "open").astype(float)
    breeding_cost = P.BREEDING_COST_PER_OPEN_MONTH * is_open
    calf_value = P.CALF_VALUE * probs["p_calve"]
    cull_cost = P.CULL_AND_REPLACE_COST * probs["p_cull"]
    loss_cost = P.PREGNANCY_LOSS_COST * probs["p_loss"]
    return {"breeding_cost": breeding_cost, "calf_value": calf_value, "cull_cost": cull_cost, "loss_cost": loss_cost,
            "total": calf_value - breeding_cost - cull_cost - loss_cost}

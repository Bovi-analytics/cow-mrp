"""OPTIONAL hooks (see cowmrp.domain). Here: where a new animal starts, and which state a replaced animal re-enters."""
import numpy as np

INITIAL_STATE = {"phase": "open", "k": 0}


def replacement_state_id(states):
    return int(np.flatnonzero((states["phase"] == "open") & (states["k"] == 0))[0])

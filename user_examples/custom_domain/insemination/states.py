"""REQUIRED: build_<name>_states() -> DataFrame with a 'state_id' column equal to the row number 0..n-1. <name> is the folder name."""
import pandas as pd

from . import parameters as P


def build_insemination_states():
    rows = [("open", k) for k in range(P.LAST_OPEN_MONTH + 1)] + [("pregnant", m) for m in range(1, P.GESTATION_MONTHS + 1)]
    states = pd.DataFrame(rows, columns=["phase", "k"])        # k = months open (0..) or months pregnant (1..)
    states["state_id"] = range(len(states))
    return states

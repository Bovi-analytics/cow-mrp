"""
cow/transitions.py
==================
Builds the transition matrix from states and probabilities.

State = (lactation, dim, dp). Two branches only, keyed on dp:

1. Open cows (dp == 0): replaced (death or culling -- including the milk-floor
   reproductive cull, which is p_cull = 1 for open cows at/after their cull DIM),
   conceive (next DIM, dp = 1), or stay open (next DIM, dp = 0).
   Past LAST_BREEDING_DIM p_preg is 0, so such a cow just ages until she is culled.
2. Pregnant / dry cows (dp > 0): replaced, abort (next DIM, dp = 0 -- or dp = 1 with probability p_resume: she
   conceives again in the remaining half step, cow/parameters.py RESUME_BREEDING_AFTER_LOSS), calve
   (next lactation, dim 0, dp 0), or stay pregnant (next DIM, dp + 1).

Replacement = new cow (lactation=1, dim=0, dp=0).
MAX LACTATION: cows at max lactation who calve are REPLACED (routed to the
replacement state instead of staying at max lactation forever).
"""

import numpy as np
import pandas as pd
from cowmrp.domains.cow.parameters import MAX_LACTATION
from cowmrp.domains.cow.tables import get_tables


def find_state_ids(states, lac, dim, dp):
    """
    Find state IDs given state attributes (arrays of the same length).
    Raises error if any target state doesn't exist.
    """
    query = pd.DataFrame({'lactation': lac, 'dim': dim, 'dp': dp})
    merged = query.merge(
        states[['state_id', 'lactation', 'dim', 'dp']],
        how='left', on=['lactation', 'dim', 'dp'],
    )
    if merged['state_id'].isna().any():
        bad = merged[merged['state_id'].isna()].drop_duplicates()
        raise ValueError(f"Some target states don't exist:\n{bad.head()}")
    return merged['state_id'].values


def get_replacement_state_id(states):
    """
    The single state_id that compute_transitions() routes death/culling/
    max-lactation-calving replacement mass into: (lactation=1, dim=0, dp=0).
    Exposed so callers outside this module can identify "the replacement
    state" without keeping an independent definition of it.
    """
    ids = find_state_ids(states, np.array([1]), np.array([0]), np.array([0]))
    return int(ids[0])


def compute_transitions(states, probs):
    """Compute all transitions. Returns (rows, cols, vals) for a sparse matrix."""
    lac = states['lactation'].values
    dim = states['dim'].values
    dp = states['dp'].values
    sid = states['state_id'].values
    n = len(states)

    p_dead, p_cull = probs['p_dead'], probs['p_cull']
    p_preg, p_abort, p_calve = probs['p_preg'], probs['p_abort'], probs['p_calve']
    p_resume = probs['p_resume']

    dim_max, dp_max = get_tables()['dim_max'], get_tables()['dp_max']
    replacement_id = get_replacement_state_id(states)
    next_dim = np.minimum(dim + 1, dim_max)

    rows, cols, vals = [], [], []

    def add(src_mask, target_lac, target_dim, target_dp, prob):
        """Add one branch for the source states in src_mask with probability `prob` (array over states)."""
        m = src_mask & (prob > 0)
        if not m.any():
            return
        rows.append(sid[m])
        cols.append(find_state_ids(states, target_lac[m], target_dim[m], target_dp[m]))
        vals.append(prob[m])

    # Replacement (death or culling): every state
    rows.append(sid)
    cols.append(np.full(n, replacement_id))
    vals.append(p_dead + (1 - p_dead) * p_cull)

    survive = (1 - p_dead) * (1 - p_cull)
    zeros = np.zeros(n, dtype=int)

    # ---- Open cows (dp == 0) ----
    is_open = dp == 0
    add(is_open, lac, next_dim, zeros + 1, survive * p_preg)            # conceive
    add(is_open, lac, next_dim, zeros, survive * (1 - p_preg))          # stay open

    # ---- Pregnant / dry cows (dp > 0) ----
    is_preg = ~is_open
    p_lost = survive * p_abort
    add(is_preg, lac, next_dim, zeros, p_lost * (1 - p_resume))         # abort -> open
    add(is_preg, lac, next_dim, zeros + 1, p_lost * p_resume)           # abort -> conceives again in the remaining half step
    survive_2 = survive * (1 - p_abort)
    p_calved = survive_2 * p_calve
    at_max = lac >= MAX_LACTATION
    add(is_preg & ~at_max, lac + 1, zeros, zeros, p_calved)             # calve -> next lactation
    m = is_preg & at_max & (p_calved > 0)                               # max lactation -> replacement
    if m.any():
        rows.append(sid[m])
        cols.append(np.full(m.sum(), replacement_id))
        vals.append(p_calved[m])
    add(is_preg, lac, next_dim, np.minimum(dp + 1, dp_max), survive_2 * (1 - p_calve))  # remain pregnant

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

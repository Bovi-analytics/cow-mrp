"""
cow/flows.py
============
BIOLOGICAL FLOW vectors derived from the cow chain -- animal counts per step, NOT economic rewards. Never added to
rewards['total'] (see the module docstring of cowmrp.domains.cow.rewards).

calving_events[s] = expected calving events per cow in state s per model step: the EXACT branch probability that
cow/transitions.py::compute_transitions gives to "calve" for pregnant/dry rows (dp > 0),
    (1 - p_dead) * (1 - p_cull) * (1 - p_abort) * p_calve,
and 0 for open rows (dp == 0). Max-lactation cows DO calve (a calf is born) although they are then routed to the
replacement state. A cow that dies or is culled in the step in which she would calve loses the calf -- the transition
model's own ordering convention. Every cow-chain calving is a parity >= 2 calving; calving_events_by_dam_parity()
splits it into parity '2' (lactation 1) and '3+' (lactation >= 2). First calvings (parity '1') come from the
youngstock chain's graduates, never from a cow state.
"""

import numpy as np


def calving_events(states, probs):
    p = ((1 - probs['p_dead']) * (1 - probs['p_cull']) * (1 - probs['p_abort']) * probs['p_calve'])
    return np.where(states['dp'].values > 0, p, 0.0)


def calving_events_by_dam_parity(states, events):
    lac = states['lactation'].values
    c = np.asarray(events)
    return {'2': np.where(lac == 1, c, 0.0), '3+': np.where(lac >= 2, c, 0.0)}

"""REQUIRED: compute_transitions(states, probs) -> (rows, cols, vals): the non-zero entries of the transition matrix P[from, to].
Every row must add up to 1 (the engine checks this and stops otherwise). A replaced or calved cow re-enters at open month 0."""
from . import parameters as P


def compute_transitions(states, probs):
    index = {(ph, k): i for ph, k, i in zip(states["phase"], states["k"], states["state_id"])}
    start = index[("open", 0)]
    rows, cols, vals = [], [], []

    def add(src, dst, p):
        if p > 0:
            rows.append(src); cols.append(dst); vals.append(float(p))

    for i, (ph, k) in enumerate(zip(states["phase"], states["k"])):
        if ph == "open":
            pc = probs["p_conceive"][i]
            add(i, index[("pregnant", 1)], pc)
            if k < P.LAST_OPEN_MONTH:
                add(i, index[("open", k + 1)], 1.0 - pc)
            else:
                add(i, start, probs["p_cull"][i])                   # culled: a replacement starts at open month 0
        elif k < P.GESTATION_MONTHS:
            add(i, start, probs["p_loss"][i])                       # pregnancy lost: open again
            add(i, index[("pregnant", k + 1)], 1.0 - probs["p_loss"][i])
        else:
            add(i, start, probs["p_calve"][i])                      # calved: a new lactation, open month 0
    return rows, cols, vals

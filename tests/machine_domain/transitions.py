import numpy as np
def compute_transitions(states, probs):
    rows, cols, vals = [], [], []
    for i in (0, 1, 2):
        if i == 0:
            moves = [(0, 1 - probs["wear"][0]), (1, probs["wear"][0])]
        elif i == 1:
            moves = [(1, 1 - probs["break"][1]), (2, probs["break"][1])]
        else:
            moves = [(0, 1.0)]
        for j, v in moves:
            rows.append(i); cols.append(j); vals.append(v)
    return np.array(rows), np.array(cols), np.array(vals)

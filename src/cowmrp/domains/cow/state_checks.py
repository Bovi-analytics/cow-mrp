"""
cow/state_checks.py
===================
Structural checks that are NOT state rules: they test a finished state table + transition matrix.

  * landing closure : every positive-probability transition lands on a state that exists
  * reachability    : every state can be reached from the replacement state through positive-probability transitions
  * support closure : the probabilities say "conception is possible" exactly where the rules say the landing state exists
  * legacy filter   : the state table as it was built before cow/state_constraints.py existed (conception steps with
                      positive probability); kept as an independent regression reference

Keeping these apart from domains.cow/state_constraints.py (feasibility rules) and cow/states.py (table construction) means a new
scientifically justified rule is one entry in CONSTRAINTS; the checks below then say whether the probabilities and the
transitions agree with it.
"""
import numpy as np
import scipy.sparse as sp


def landing_closure_errors(states, rows, cols, vals):
    """Transitions with positive probability whose target id is missing or out of range ([] = closed)."""
    n = len(states)
    bad = (np.asarray(vals) > 0) & ((np.asarray(cols) < 0) | (np.asarray(cols) >= n))
    return np.flatnonzero(bad).tolist()


def unreachable_states(states, matrix, start_id):
    """state_ids that cannot be reached from `start_id` (sparse CSR matrix, rows = from)."""
    n = len(states)
    seen = np.zeros(n, dtype=bool)
    seen[start_id] = True
    frontier = [start_id]
    while frontier:
        nxt = np.unique(matrix[frontier].indices)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = list(nxt)
    return np.flatnonzero(~seen).tolist()


def to_csr(states, rows, cols, vals):
    n = len(states)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def support_mismatches(states, p_preg, feasible_landing, dim_max=None):
    """Open states whose p_preg > 0 disagrees with `feasible_landing(dim, lactation)` (bool array: is the landing state
    (dim+1, dp=1) possible?). States whose landing step would fall beyond `dim_max` are skipped. Returns a list of
    (lactation, dim, p_preg, landing_feasible)."""
    dim, dp, lac = states['dim'].values, states['dp'].values, states['lactation'].values
    m = dp == 0
    if dim_max is not None:
        m = m & (dim + 1 <= dim_max)
    landing = feasible_landing(dim[m], lac[m])
    got = p_preg[m] > 1e-12
    bad = np.flatnonzero(got != landing)
    idx = np.flatnonzero(m)[bad]
    return [(int(lac[i]), int(dim[i]), float(p_preg[i]), bool(landing[b])) for i, b in zip(idx, bad)]


def legacy_state_table(domain_product, entry_steps):
    """The previous state filter, verbatim: keep open states and pregnant states whose conception step
    (dim - dp) is a step with positive conception probability. `domain_product` = the engine's full product table."""
    keep = (domain_product['dp'].values == 0) | np.isin(domain_product['dim'].values - domain_product['dp'].values, entry_steps)
    out = domain_product[keep].reset_index(drop=True)
    out['state_id'] = out.index
    return out

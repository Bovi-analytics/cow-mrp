"""cowmrp.mrp.utils -- small domain-free helpers for any transition matrix (no domain, no runner, no file I/O)."""
import hashlib

import numpy as np


def matrix_checksum(P):
    """Hash of a sparse matrix (positions and values rounded to 12 decimals): equal for identical matrices, so two builds can be compared."""
    M = P.tocsr()
    M.sort_indices()
    h = hashlib.sha256()
    for a in (M.indptr, M.indices):
        h.update(a.astype("int64").tobytes())
    h.update(M.data.round(12).astype("float64").tobytes())
    return h.hexdigest()[:16]


def row_sum_report(P, tol=1e-8):
    """How far the rows of P are from summing to 1: number of rows off, smallest and largest row sum."""
    rs = np.asarray(P.sum(axis=1)).ravel()
    return {"rows_not_summing_to_1": int((abs(rs - 1) > tol).sum()), "row_sum_min": float(rs.min()), "row_sum_max": float(rs.max())}


def reachable_states(P, start):
    """Indices of the states reachable from `start` (an index, or a list of indices) by following transitions with probability > 0."""
    P = P.tocsr()
    n = P.shape[0]
    seen = np.zeros(n, dtype=bool)
    frontier = np.atleast_1d(np.asarray(start, dtype=int))
    seen[frontier] = True
    while frontier.size:
        nxt = np.unique(P[frontier].indices)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    return np.flatnonzero(seen)

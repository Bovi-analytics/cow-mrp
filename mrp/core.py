"""
mrp_core.py
===========
Generic Markov Reward Process math/data-structure primitives (domain-
independent, spec-independent).

This is the Core layer: matrix assembly, Bellman/exact-solve, discounting,
distribution evolution, and simulation. It has no notion of a "spec"
(name/depends_on/method/methods) at all -- that orchestration lives one
layer up, in mrp_engine.py.

Dependency direction: mrp_engine.py -> mrp_core.py -> (nothing). This file
must never import from mrp_engine.py or from any domain package (e.g. cow/).
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import spsolve


# ============================================================
# GAMMA CONVERSION
# ============================================================

def annual_to_per_step_gamma(gamma_annual, unit_days):
    """Convert annual discount to per-step discount."""
    return gamma_annual ** (unit_days / 365.0)


# ============================================================
# GENERIC LOOKUP (wildcard table matcher)
# ============================================================

def lookup(table, keys):
    """
    Generic wildcard table lookup.

    table: DataFrame with key columns + 'p' column
    keys: dict of {column: array_of_values}

    'ANY' in a table row matches regardless of that key.
    Most specific row (fewest 'ANY') wins.
    """
    key_columns = list(keys.keys())
    n = len(next(iter(keys.values())))
    result = np.full(n, np.nan)

    specificity = table[key_columns].apply(lambda row: sum(v != 'ANY' for v in row), axis=1)
    most_specific_first = table.assign(_spec=specificity).sort_values('_spec', ascending=False)

    for _, record in most_specific_first.iterrows():
        still_unmatched = np.isnan(result)
        if not still_unmatched.any():
            break
        matches = still_unmatched.copy()
        for col in key_columns:
            if record[col] != 'ANY':
                matches &= (keys[col] == record[col])
        result[matches] = record['p']

    return result


# ============================================================
# MATRIX OPERATIONS
# ============================================================

def assemble_matrix(states, rows, cols, vals):
    """Build a sparse CSR transition matrix from (row, col, val) lists."""
    n = len(states)
    matrix = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    return matrix

def validate_row_stochasticity(matrix, states, tolerance=1e-8):
    """Check that every row sums to 1.0."""
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    bad_ids = np.where(np.abs(row_sums - 1.0) > tolerance)[0]
    if len(bad_ids) > 0:
        by_id = states.set_index('state_id', drop=False)
        offenders = by_id.loc[bad_ids[:10], [c for c in states.columns if c != 'state_id']]
        offenders['row_sum'] = row_sums[bad_ids[:10]]
        raise ValueError(f"{len(bad_ids)} row(s) don't sum to 1:\n{offenders}")
    print(f"All {matrix.shape[0]:,} rows sum to 1.0 (within {tolerance})")
    return True

def get_transitions(matrix, states, **query):
    """Inspect transitions from a specific state."""
    variable_names = [c for c in states.columns if c != 'state_id']
    matches = states
    for column, value in query.items():
        matches = matches[matches[column] == value]
    if matches.empty:
        raise ValueError(f"No state matches {query}.")
    state_id = matches['state_id'].iloc[0]
    row = matrix.getrow(state_id)
    by_id = states.set_index('state_id', drop=False)
    targets = by_id.loc[row.indices, variable_names].copy()
    targets['probability'] = row.data
    return targets


# ============================================================
# VALUE ANALYSIS
# ============================================================

def steps_per_year(unit_days):
    """Number of time steps in one year."""
    return round(365 / unit_days)

def evaluate_finite_horizon(matrix, reward, gamma, num_years, unit_days):
    """
    Finite-horizon Bellman iteration: V_{t+1} = R + gamma * P * V_t
    Prints a summary once per year.
    """
    n = matrix.shape[0]
    steps_in_a_year = steps_per_year(unit_days)
    total_steps = steps_in_a_year * num_years
    V = np.zeros(n)
    print(f"\n{steps_in_a_year} steps/year x {num_years} years = {total_steps} steps")
    for step in range(1, total_steps + 1):
        V = reward + gamma * matrix.dot(V)
        if step % steps_in_a_year == 0:
            year = step // steps_in_a_year
            print(f"Year {year:3d}: mean V = {V.mean():12.2f}  min = {V.min():12.2f}  max = {V.max():12.2f}")
    return V

def evaluate_infinite_horizon(matrix, reward, gamma):
    """
    Infinite-horizon exact solve: V = (I - gamma*P)^-1 * R
    Uses sparse linear solve for efficiency.
    """
    n = matrix.shape[0]
    A = (sparse_eye(n, format='csc') - gamma * matrix).tocsc()
    return spsolve(A, reward)


# ============================================================
# STATE DISTRIBUTION EVOLUTION (3 methods)
# ============================================================

def find_state_id(states, **query):
    """
    Find the single state_id matching the given attribute query.
    Raises a ValueError with nearby-match suggestions if the query
    doesn't resolve to exactly one state -- e.g. wrong flag for a
    given dp, or a typo'd health string.
    """
    matches = states
    for column, value in query.items():
        matches = matches[matches[column] == value]
    if matches.empty:
        # Help the caller find a valid combination: relax one field at a
        # time and show what IS available for the rest of the query.
        hints = []
        for column in query:
            relaxed = states
            for c, v in query.items():
                if c != column:
                    relaxed = relaxed[relaxed[c] == v]
            if not relaxed.empty:
                options = sorted(relaxed[column].unique().tolist(), key=str)
                hints.append(f"  valid '{column}' values for the rest of your query: {options}")
        hint_text = "\n".join(hints) if hints else "  (no field in isolation has valid options -- check the whole combination)"
        raise ValueError(f"No state matches {query}.\n{hint_text}")
    if len(matches) > 1:
        raise ValueError(f"Multiple states match {query}. Be more specific.\n{matches.head()}")
    return matches['state_id'].iloc[0]


def state_values(states, V, query_A, query_B):
    """
    Look up V for two states by attribute query, WITHOUT subtracting --
    the paired counterpart to state_value_diff (below), which returns
    only their difference. Added because domain-layer comparisons (e.g.
    cow/value_comparisons.py's pregnancy_value/replacement_value) need
    the two raw values for reporting/display (a "V(this state) vs V(that
    state)" table), not just the difference, and state_value_diff alone
    can't provide that without a second, duplicate pair of lookups at
    each call site.

    Domain-independent, same guarantee as state_value_diff: no notion of
    what the query attributes mean. Returns (V[id_a], V[id_b]). Raises
    ValueError (via find_state_id) if either query does not resolve to
    exactly one existing state.
    """
    id_a = find_state_id(states, **query_A)
    id_b = find_state_id(states, **query_B)
    return V[id_a], V[id_b]


def state_value_diff(states, V, query_A, query_B):
    """
    Generic state-value comparison: V(state matching query_A) minus
    V(state matching query_B), where each query is an attribute dict
    resolved via find_state_id and V is any per-state value vector (e.g.
    from evaluate_finite_horizon / evaluate_infinite_horizon).

    Domain-independent: this function has no notion of what the query
    attributes MEAN (pregnancy, health, or any other domain's state
    variables) -- it is the natural generalization of find_state_id
    applied against a value vector instead of the state table alone.
    Every domain-specific "value of X" comparison (e.g.
    cow/value_comparisons.py::pregnancy_value) is built by supplying two
    attribute dicts to this one function; it does not duplicate this
    lookup-and-subtract logic itself.

    Raises ValueError (via find_state_id) if either query does not
    resolve to exactly one existing state -- never silently returns a
    value for a state that doesn't exist. This function makes no claim
    about whether the two queried states form a scientifically valid
    counterfactual pair -- that judgment belongs to the caller (the
    domain layer), not to this generic primitive.
    """
    V_a, V_b = state_values(states, V, query_A, query_B)
    return V_a - V_b


def evaluate_state_trajectory(matrix, metric_vector, pi0, num_steps):
    """
    Per-step (NOT cumulative) expected value of metric_vector, starting
    from distribution pi0, for num_steps steps ahead.

    trajectory[t] = (pi0 @ P^(t+1)) @ metric_vector   for t = 0..num_steps-1

    i.e. trajectory[0] is the expected metric value one step after pi0,
    trajectory[1] is the expected value two steps after pi0, etc. Each
    entry stands alone -- it is the expected value AT that step, not
    the running total.
    """
    pi = pi0.copy()
    trajectory = np.empty(num_steps)
    for t in range(num_steps):
        pi = pi @ matrix
        trajectory[t] = pi @ metric_vector
    return trajectory


def evaluate_state_trajectory_to_convergence(matrix, metric_vector, pi0, tol=1e-6, max_steps=2000):
    """
    Same as evaluate_state_trajectory, but run forward until the
    DISTRIBUTION itself converges to steady-state (L1 change between
    successive steps < tol), capped at max_steps as a safety net. This
    is the "infinite horizon" trajectory: the flat tail is the
    steady-state per-step expected value.

    Returns (trajectory, steps_run, converged) where trajectory has
    length steps_run and converged is False if the cap was hit first.
    """
    pi = pi0.copy()
    trajectory = []
    converged = False
    for t in range(max_steps):
        pi_next = pi @ matrix
        trajectory.append(pi_next @ metric_vector)
        if np.abs(pi_next - pi).sum() < tol:
            pi = pi_next
            converged = True
            break
        pi = pi_next
    return np.array(trajectory), len(trajectory), converged


def make_initial_distribution(states, **query):
    """Create a one-hot initial distribution from state attributes."""
    state_id = find_state_id(states, **query)
    pi0 = np.zeros(len(states))
    pi0[state_id] = 1.0
    return pi0

def distribution_recursive(matrix, pi0, num_steps):
    """Method 1: pi_{t+1} = pi_t * P (step by step)."""
    pi = pi0.copy()
    for _ in range(num_steps):
        pi = pi @ matrix
    return pi

def distribution_matrix_power(matrix, pi0, num_steps):
    """Method 2: pi_t = pi_0 * P^t (direct matrix power)."""
    P_power = np.linalg.matrix_power(matrix.toarray(), num_steps)
    return pi0 @ P_power

def distribution_eigen(matrix, pi0, num_steps):
    """Method 3: pi_t = pi_0 * V * Lambda^t * V^-1 (eigendecomposition)."""
    P_dense = matrix.toarray()
    eigvals, eigvecs = np.linalg.eig(P_dense)
    eigvecs_pinv = np.linalg.pinv(eigvecs)
    P_t = eigvecs @ np.diag(eigvals ** num_steps) @ eigvecs_pinv
    return (pi0 @ P_t).real


def distribution_recursive_origin_tracked(matrix, pi_untagged, pi_tagged, num_steps, origin_state_id):
    """
    Generalizes distribution_recursive to also partition the evolving
    distribution by whether probability mass has ever RE-ENTERED a single
    marked state (origin_state_id) since the caller's own t=0 -- e.g. a
    domain's "replacement"/reset state that only certain transitions
    route into. Domain-agnostic: knows nothing about cows, replacement,
    or any other domain concept, only "which single state id marks
    re-entry into the tracked category."

    At each of num_steps steps: both pi_untagged and pi_tagged are
    advanced through the SAME matrix; any mass that lands on
    origin_state_id via pi_untagged's own evolution is reclassified into
    pi_tagged going forward (mass already in pi_tagged that lands on
    origin_state_id again simply stays tagged -- no special-casing
    needed, since it's never moved out of pi_tagged).

    pi_untagged + pi_tagged == distribution_recursive(matrix, pi_untagged
    + pi_tagged, num_steps) at every step, exactly (both partitions
    evolve through the identical matrix; only the bookkeeping differs).

    Composable exactly like distribution_recursive: call once with the
    full step count, or call repeatedly with num_steps=1 feeding each
    call's output back in as the next call's input to build a step-by-
    step trajectory (both give identical results, since this is just
    repeated matrix multiplication) -- this is NOT a one-shot endpoint-
    only function.

    Convention: pass pi_tagged=np.zeros_like(pi_untagged) for a fresh
    start -- nothing is tagged at t=0 even if pi_untagged already has
    mass sitting on origin_state_id (that reflects how the initial
    population was seeded, not a re-entry event).
    """
    for _ in range(num_steps):
        raw_untagged = pi_untagged @ matrix
        raw_tagged = pi_tagged @ matrix
        newly_tagged = raw_untagged[origin_state_id]
        raw_untagged[origin_state_id] = 0.0
        raw_tagged[origin_state_id] += newly_tagged
        pi_untagged, pi_tagged = raw_untagged, raw_tagged
    return pi_untagged, pi_tagged


def trajectory_tagged_mask(history, origin_state_id):
    """
    Domain-agnostic counterpart to distribution_recursive_origin_tracked
    for per-entity state-ID histories from simulate_stochastic_trajectory/
    simulate_monte_carlo (shape (n_steps+1, n_entities)).

    Returns a same-shape boolean array: True at [step, entity] iff that
    entity's own trajectory visited origin_state_id at some step >= 1 and
    <= the given step (i.e. "has this entity re-entered the tracked
    state by this point"). Step 0 (the initial condition) is NEVER
    tagged, even if an entity's initial state IS origin_state_id --
    starting there reflects how the initial population was seeded, not a
    re-entry event (same t=0 convention as
    distribution_recursive_origin_tracked's pi_tagged(0)=0).
    """
    visited = (history == origin_state_id)
    visited[0, :] = False
    return np.maximum.accumulate(visited, axis=0)


# ============================================================
# EXPORT HELPERS
# ============================================================

def export_transitions(matrix, states, unit_prefix='m'):
    """Export transition matrix to long-format DataFrame, sorted by source state."""
    variable_names = [c for c in states.columns if c != 'state_id']
    export_names = {'dim': f'{unit_prefix}im', 'dp': f'{unit_prefix}p'}

    coo = matrix.tocoo()
    by_id = states.set_index('state_id', drop=False)
    src = by_id.loc[coo.row, variable_names].reset_index(drop=True)
    tgt = by_id.loc[coo.col, variable_names].reset_index(drop=True)

    out = pd.DataFrame({f'src_{export_names.get(v,v)}': src[v] for v in variable_names})
    for v in variable_names:
        out[f'tgt_{export_names.get(v,v)}'] = tgt[v]
    out['probability'] = coo.data

    src_cols = [f'src_{export_names.get(v,v)}' for v in variable_names]
    tgt_cols = [f'tgt_{export_names.get(v,v)}' for v in variable_names]
    out = out.sort_values(by=src_cols + tgt_cols)
    out = out.reset_index(drop=True)
    return out

def export_rewards(states, rewards, unit_prefix='m'):
    """Export reward vector to DataFrame."""
    variable_names = [c for c in states.columns if c != 'state_id']
    export_names = {'dim': f'{unit_prefix}im', 'dp': f'{unit_prefix}p'}

    out = pd.DataFrame({export_names.get(v,v): states[v].values for v in variable_names})
    for k, v in rewards.items():
        out[k] = v
    return out


# ============================================================
# STOCHASTIC SIMULATION EXTENSIONS (domain-agnostic)
# ============================================================
# These functions work with ANY transition matrix and ANY set of
# initial state IDs. They are the stochastic counterparts to the
# deterministic distribution methods above.
# ============================================================

def precompute_transition_lists(matrix):
    """
    Precompute transition target IDs and probabilities for fast sampling.

    Converts a sparse CSR matrix into a dict of (targets, probs) tuples,
    avoiding repeated sparse row lookups during stochastic simulation.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Row-stochastic transition matrix

    Returns
    -------
    dict
        {state_id: (target_ids_array, probs_array)}
    """
    trans = {}
    n = matrix.shape[0]
    for i in range(n):
        row = matrix.getrow(i)
        if row.nnz == 0:
            trans[i] = (np.array([i]), np.array([1.0]))
        else:
            probs = row.data.copy()
            probs = probs / probs.sum()
            trans[i] = (row.indices.copy(), probs)
    return trans


def simulate_stochastic_trajectory(matrix, initial_state_ids, n_steps, seed=None):
    """
    Single stochastic trajectory simulation.

    For each entity (cow, machine, etc.), at each step, sample next state
    from the transition row. Domain-agnostic -- works with any transition
    matrix from any domain.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Row-stochastic transition matrix from assemble_matrix()
    initial_state_ids : array-like of int
        State ID for each entity at time 0
    n_steps : int
        Number of time steps to simulate
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_steps+1, n_entities)
        State ID for each entity at each time step
    """
    rng = np.random.default_rng(seed)
    trans = precompute_transition_lists(matrix)

    entity_states = np.asarray(initial_state_ids).copy()
    n_entities = len(entity_states)

    history = np.empty((n_steps + 1, n_entities), dtype=int)
    history[0] = entity_states.copy()

    for step in range(1, n_steps + 1):
        next_states = np.empty(n_entities, dtype=int)
        for idx in range(n_entities):
            targets, probs = trans[entity_states[idx]]
            next_states[idx] = rng.choice(targets, p=probs)
        entity_states = next_states
        history[step] = entity_states.copy()

    return history


def simulate_monte_carlo(matrix, initial_state_ids, n_steps, n_sims=100, seed=None):
    """
    Monte Carlo simulation: run N independent stochastic trajectories.

    Domain-agnostic. Returns raw state ID histories. The caller is
    responsible for grouping/analysis based on their domain.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
    initial_state_ids : array-like of int
    n_steps : int
    n_sims : int
        Number of independent simulations (default: 100)
    seed : int or None

    Returns
    -------
    list of np.ndarray
        Each element has shape (n_steps+1, n_entities)
    """
    rng = np.random.default_rng(seed)
    trans = precompute_transition_lists(matrix)
    n_entities = len(initial_state_ids)
    trajectories = []

    for _ in range(n_sims):
        entity_states = np.asarray(initial_state_ids).copy()
        history = np.empty((n_steps + 1, n_entities), dtype=int)
        history[0] = entity_states.copy()

        for step in range(1, n_steps + 1):
            next_states = np.empty(n_entities, dtype=int)
            for idx in range(n_entities):
                targets, probs = trans[entity_states[idx]]
                next_states[idx] = rng.choice(targets, p=probs)
            entity_states = next_states
            history[step] = entity_states.copy()

        trajectories.append(history)

    return trajectories

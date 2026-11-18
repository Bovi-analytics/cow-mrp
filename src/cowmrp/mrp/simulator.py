"""
simulator.py
============
Domain-agnostic MRP simulation orchestrator.

Wraps mrp_core's deterministic distribution methods and stochastic
sampling methods into a clean API. Works with ANY domain's transition
matrix and state space.

USAGE
-----
    from cowmrp.mrp.simulator import run_expected, run_stochastic, run_monte_carlo, compare_methods

    # Expected mode (deterministic)
    results = run_expected(matrix, pi0, n_steps=12, methods=['recursive','matrix_power','eigen'])

    # Stochastic mode (single trajectory)
    history = run_stochastic(matrix, initial_state_ids, n_steps=12, seed=42)

    # Monte Carlo mode (batch trajectories)
    trajectories = run_monte_carlo(matrix, initial_state_ids, n_steps=12, n_sims=100, seed=42)

    # Compare all 3 expected methods
    comparison_df = compare_methods(matrix, pi0, n_steps=12)

DEPENDENCIES
------------
- numpy, pandas
- mrp_core.py (generic engine)
"""

import numpy as np
import pandas as pd

from cowmrp.mrp.core import (
    distribution_recursive,
    distribution_matrix_power,
    distribution_eigen,
    distribution_recursive_origin_tracked,
    simulate_stochastic_trajectory,
    simulate_monte_carlo,
)


# ============================================================
# EXPECTED MODE (deterministic distribution evolution)
# ============================================================

def run_expected(matrix, pi0, n_steps, methods=None):
    """
    Run one or more deterministic distribution evolution methods.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Row-stochastic transition matrix
    pi0 : np.ndarray
        Initial probability distribution (sums to 1.0)
    n_steps : int
        Number of time steps to evolve
    methods : list of str or None
        Which methods to run. Options: "recursive", "matrix_power", "eigen".
        Default: all three.

    Returns
    -------
    dict
        {method_name: pi_vector_at_step_n}
    """
    if methods is None:
        methods = ["recursive", "matrix_power", "eigen"]

    results = {}
    for method in methods:
        if method == "recursive":
            results[method] = distribution_recursive(matrix, pi0, n_steps)
        elif method == "matrix_power":
            results[method] = distribution_matrix_power(matrix, pi0, n_steps)
        elif method == "eigen":
            results[method] = distribution_eigen(matrix, pi0, n_steps)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'recursive', 'matrix_power', or 'eigen'.")
    return results


def run_expected_origin_tracked(matrix, pi_untagged, pi_tagged, n_steps, origin_state_id):
    """
    Expected-mode counterpart to run_expected() that also splits the
    distribution into "untagged"/"tagged" sub-populations by whether mass
    has re-entered origin_state_id (e.g. a domain's replacement state).
    Thin wrapper over mrp_core.distribution_recursive_origin_tracked --
    see that function's docstring for the exact semantics and for why
    it's safe to call once for the full n_steps or repeatedly with
    n_steps=1 to build a step-by-step trajectory.
    """
    return distribution_recursive_origin_tracked(matrix, pi_untagged, pi_tagged, n_steps, origin_state_id)


def compare_methods(matrix, pi0, n_steps, reward_vector=None):
    """
    Run all 3 expected methods and return a comparison DataFrame.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
    pi0 : np.ndarray
    n_steps : int
    reward_vector : np.ndarray or None
        If provided, computes expected reward for each method.

    Returns
    -------
    pd.DataFrame
        Columns: method, sum_check, nonzero_count, expected_reward (if provided)
    """
    results = run_expected(matrix, pi0, n_steps)
    rows = []
    for method, pi in results.items():
        row = {
            "method": method,
            "sum_check": float(pi.sum()),
            "nonzero_count": int(np.count_nonzero(pi)),
        }
        if reward_vector is not None:
            row["expected_reward"] = float(pi @ reward_vector)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# STOCHASTIC MODE (single trajectory)
# ============================================================

def run_stochastic(matrix, initial_state_ids, n_steps, seed=None):
    """
    Run a single stochastic trajectory simulation.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
    initial_state_ids : array-like of int
    n_steps : int
    seed : int or None

    Returns
    -------
    np.ndarray of shape (n_steps+1, n_entities)
    """
    return simulate_stochastic_trajectory(matrix, initial_state_ids, n_steps, seed=seed)


# ============================================================
# MONTE CARLO MODE (batch trajectories)
# ============================================================

def run_monte_carlo(matrix, initial_state_ids, n_steps, n_sims=100, seed=None):
    """
    Run Monte Carlo simulation with N independent trajectories.

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
    return simulate_monte_carlo(matrix, initial_state_ids, n_steps, n_sims=n_sims, seed=seed)


def summarize_monte_carlo(trajectories, states, label_fn):
    """
    Generic Monte Carlo summarizer.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Output from run_monte_carlo()
    states : pd.DataFrame
        State table with columns including state_id
    label_fn : callable
        Function: label = label_fn(row) where row is a pd.Series from states.
        Example for cows: lambda r: r['flag']

    Returns
    -------
    dict
        {step: pd.DataFrame with columns [mean, std, p05, p95] indexed by label}
    """
    n_steps = trajectories[0].shape[0] - 1
    all_labels = set()

    # Discover all possible labels
    for step in range(n_steps + 1):
        for traj in trajectories:
            labels = [label_fn(states.iloc[sid]) for sid in traj[step]]
            all_labels.update(labels)
    all_labels = sorted(all_labels)

    summary = {}
    for step in range(n_steps + 1):
        df = pd.DataFrame(index=all_labels)
        for label in all_labels:
            vals = []
            for traj in trajectories:
                labels = [label_fn(states.iloc[sid]) for sid in traj[step]]
                count = labels.count(label)
                vals.append(count)
            df.loc[label, 'mean'] = np.mean(vals)
            df.loc[label, 'std'] = np.std(vals)
            df.loc[label, 'p05'] = np.percentile(vals, 5)
            df.loc[label, 'p95'] = np.percentile(vals, 95)
        summary[step] = df

    return summary

"""cowmrp.model -- build and solve the MRP of any domain with the generic engine (``cowmrp.mrp``).

Layering (one-way): ``cowmrp.mrp`` (pure math; imports nothing else from cowmrp)  <-  ``cowmrp.model`` (this file: calls the engine
with what a ``cowmrp.domain.Domain`` provides; no equations, no domain names)  <-  ``cowmrp.domains.*`` / ``runners``.

    from cowmrp.domain import load_domain
    from cowmrp.model import build_model, evaluate
    dom = load_domain("cow")                 # or "youngstock", or "my_package.my_domain"
    model = build_model(dom)                 # states, probabilities, rewards, sparse transition matrix P
    full = evaluate(model, dom)              # every state of the full table      (scope "full")
    reach = evaluate(model, dom, start="replacement")   # only states reachable from the replacement state (scope "reachable")

Full-table and reachable-state results are different quantities: means over the full table include states no animal can reach under the
default settings (cow, month: 2,970 states of which 1,210 reachable). Every result says which it is.
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np

from cowmrp.domain import Domain, load_domain
from cowmrp.mrp import core
from cowmrp.mrp.utils import matrix_checksum, reachable_states, row_sum_report   # noqa: F401  (re-exported)

if TYPE_CHECKING:                       # only for the annotation of group_summary; pandas is imported inside the function
    import pandas as pd

Model = namedtuple("Model", "states probs rewards P")


def build_model(dom, strict: bool | None = None) -> Model:
    """States -> probabilities -> rewards (if the domain has them) -> transition matrix.

    ``strict=True``: every row must sum to 1 or the build stops; ``False``: rows are kept as the domain defines them (a domain with known
    rows that do not sum to 1). ``None`` (default): the domain's ``STRICT`` hook, else True.
    """
    dom = load_domain(dom)
    strict = dom.strict if strict is None else strict
    states = dom.build_states()
    probs = dom.probabilities.compute_probabilities(states)
    rewards = dom.rewards.compute_rewards(states, probs) if dom.has_rewards else None
    rows, cols, vals = dom.transitions.compute_transitions(states, probs)
    P = core.assemble_matrix(states, rows, cols, vals)
    if strict:
        core.validate_row_stochasticity(P, states)
    return Model(states, probs, rewards, P)


def per_step_gamma(dom, gamma_annual: float = 0.95) -> float:
    """Per-step discount factor: the domain's own convention (``gamma_per_step`` hook) or the generic annual -> per-step conversion."""
    dom = load_domain(dom)
    fn = dom.hook("gamma_per_step")
    return fn(gamma_annual) if fn else core.annual_to_per_step_gamma(gamma_annual, dom.unit_days)


def start_state_id(model: Model, dom, start) -> int:
    """Resolve `start`: a state id (int), ``"replacement"`` (the domain's replacement_state_id hook) or ``"initial"`` (INITIAL_STATE hook, else id 0)."""
    dom = load_domain(dom)
    if isinstance(start, (int, np.integer)):
        return int(start)
    if start == "replacement":
        fn = dom.hook("replacement_state_id")
        if fn is None:
            raise ValueError(f"domain {dom.name!r} has no replacement_state_id hook")
        return int(fn(model.states))
    if start == "initial":
        return int(np.argmax(initial_distribution(model, dom)))
    raise ValueError(f"start must be a state id, 'replacement' or 'initial', not {start!r}")


def evaluate(model: Model, dom, gamma_annual: float = 0.95, years: int = 5, start=None, which=("finite", "infinite")) -> dict:
    """Expected discounted value per state (needs the domain's rewards).

    ``start=None``: all states of the table (``scope`` "full").
    ``start=<state id | 'replacement' | 'initial'>``: only the states reachable from it (``scope`` "reachable"). That set is closed under the
    transitions, so its values equal the full-table values on those states; the solve runs on the smaller closed subsystem, which is
    exact and cheaper (cow, month: 1,210 instead of 2,970 states; day: far fewer than the full table). ``state_ids`` are indices into
    ``model.states``. ``which`` skips what is not needed (the infinite-horizon solve is the expensive one).
    """
    dom = load_domain(dom)
    if model.rewards is None:
        raise ValueError(f"domain {dom.name!r} defines no rewards: nothing to evaluate")
    gamma = per_step_gamma(dom, gamma_annual)
    total = np.asarray(model.rewards["total"])
    P, ids, scope = model.P, np.arange(model.P.shape[0]), "full"
    if start is not None:
        ids = reachable_states(model.P, start_state_id(model, dom, start))
        P, total, scope = model.P.tocsr()[ids][:, ids], total[ids], "reachable"
    out = {"scope": scope, "state_ids": ids, "gamma": gamma, "n_states": len(ids)}
    if "finite" in which:
        out["V_finite"] = core.evaluate_finite_horizon(P, total, gamma, years, dom.unit_days)
    if "infinite" in which:
        out["V_infinite"] = core.evaluate_infinite_horizon(P, total, gamma)
    return out


def initial_distribution(model: Model, dom, **state) -> np.ndarray:
    """Probability mass 1.0 on one state: ``state`` (e.g. ``lactation=2, dim=5, dp=0``), else the domain's ``INITIAL_STATE`` hook, else state id 0."""
    dom = load_domain(dom)
    state = state or dom.hook("INITIAL_STATE", {})
    if state:
        return core.make_initial_distribution(model.states, **state)
    pi0 = np.zeros(model.P.shape[0])
    pi0[0] = 1.0
    return pi0


def evolve_distribution(model: Model, pi0, dom, steps: int | None = None, method: str = "recursive") -> np.ndarray:
    """Where the mass is after ``steps`` steps (default one year). ``recursive`` is sparse and always fine; ``matrix_power`` and ``eigen`` are dense (small domains)."""
    dom = load_domain(dom)
    steps = core.steps_per_year(dom.unit_days) if steps is None else steps
    fn = {"recursive": core.distribution_recursive, "matrix_power": core.distribution_matrix_power, "eigen": core.distribution_eigen}.get(method)
    if fn is None:
        raise ValueError(f"method must be recursive, matrix_power or eigen, not {method!r}")
    return fn(model.P, pi0, steps)


def replacement_split(model: Model, pi0, dom, steps: int | None = None, keep_trajectory: bool = True):
    """Distribution split into 'never replaced since t=0' and 'replaced at least once' (the domain's ``replacement_state_id`` hook).

    Returns ``(never_replaced, replaced)``: lists of ``steps + 1`` vectors, or only the last vector of each when ``keep_trajectory=False``
    (memory O(n) instead of O(steps x n)).
    """
    dom = load_domain(dom)
    fn = dom.hook("replacement_state_id")
    if fn is None:
        raise ValueError(f"domain {dom.name!r} has no replacement_state_id hook")
    steps = core.steps_per_year(dom.unit_days) if steps is None else steps
    rid = fn(model.states)
    orig, tag = np.array(pi0, dtype=float), np.zeros(len(pi0))
    if not keep_trajectory:
        for _ in range(steps):
            orig, tag = core.distribution_recursive_origin_tracked(model.P, orig, tag, 1, rid)
        return orig, tag
    orig_traj, tag_traj = [orig.copy()], [tag.copy()]
    for _ in range(steps):
        orig, tag = core.distribution_recursive_origin_tracked(model.P, orig, tag, 1, rid)
        orig_traj.append(orig.copy())
        tag_traj.append(tag.copy())
    return orig_traj, tag_traj


def simulate(model: Model, initial_state_ids, n_steps: int, seed: int | None = None):
    """One stochastic trajectory per entity: array (n_steps + 1, n_entities) of state ids."""
    return core.simulate_stochastic_trajectory(model.P, np.asarray(initial_state_ids), n_steps, seed=seed)


def monte_carlo(model: Model, initial_state_ids, n_steps: int, n_sims: int = 50, seed: int | None = None):
    """``n_sims`` independent stochastic simulations (list of trajectories)."""
    return core.simulate_monte_carlo(model.P, np.asarray(initial_state_ids), n_steps, n_sims=n_sims, seed=seed)


def group_summary(model: Model, dom, trajectories) -> "pd.DataFrame":
    """Monte Carlo summary by the domain's ``group_labels`` hook: mean / std / 5th / 95th percentile of the number of entities per label at the final step.

    Vectorised: one pass over the stacked final states (no per-simulation Python loop over labels).
    """
    import pandas as pd
    dom = load_domain(dom)
    fn = dom.hook("group_labels")
    if fn is None:
        raise ValueError(f"domain {dom.name!r} has no group_labels hook")
    labels = np.asarray(fn(model.states)).astype(str)
    final = np.stack([np.asarray(t)[-1] for t in trajectories])                  # (n_sims, n_entities)
    names, code_of_state = np.unique(labels, return_inverse=True)
    codes = code_of_state[final]                                                  # label code of every entity in every simulation
    n_sims, k = codes.shape[0], len(names)
    counts = np.bincount((codes + k * np.arange(n_sims)[:, None]).ravel(), minlength=n_sims * k).reshape(n_sims, k)
    present = counts.sum(axis=0) > 0
    return pd.DataFrame({"label": names[present], "mean": counts[:, present].mean(axis=0), "std": counts[:, present].std(axis=0),
                         "p05": np.percentile(counts[:, present], 5, axis=0), "p95": np.percentile(counts[:, present], 95, axis=0)})

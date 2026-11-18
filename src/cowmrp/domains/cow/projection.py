"""cowmrp.domains.cow.projection -- "where will this cow be, and what will she earn, over the next N years?"

Cow A versus cow B (or any number of scenarios): each scenario is a starting state (lactation, days in milk, days pregnant); the
model's own transition matrix is applied step by step and the expected value of EVERY reward component (and the physical quantities:
milk, dry-matter intake, methane, CO2, nitrogen) is accumulated, per day and cumulative with discounting.
This module contains no model equation: it calls the cow domain (``cowmrp.model.build_model``) and the generic engine
(``cowmrp.mrp.core``). ``runners/cow/projection.py`` calls it.

    from cowmrp.domains.cow.projection import compare_cows
    results = compare_cows([dict(label="A", time_unit="month", lactation=2, dim_days=150, dp_days=0),
                            dict(label="B", time_unit="month", lactation=2, dim_days=150, dp_days=60)], years=2)
    from cowmrp.viz.projection import projection_tables, projection_figures      # tables (pandas) and Plotly figures

A scenario is a dict: ``label``; ``time_unit`` ("month" | "week" | "day"); ``lactation`` (1-10); the state either in calendar days
(``dim_days``, ``dp_days``) or in model steps (``dim``, ``dp``). Reward at step t is collected in the state occupied at step t; the cumulative
curve is sum_t gamma^t E[reward_t] (money discounted, physical quantities not) and its final 'total' equals the engine's finite-horizon
value (checked on every run: ``check_vs_finite_horizon``). A cow that dies or is culled is replaced by a heifer (the model's replacement
state), so the projection follows the cow SLOT; replacement cost is NOT in the reward vector or in 'total' and is returned as the extra
series ``replacement_cost_memo``. ``original_still_here`` tracks when the first cow has left.

Time resolution: the cow domain reads ``COW_TIME_UNIT`` once at import, so one process holds one resolution. ``compare_cows`` runs the
scenarios of the current resolution in this process and starts a worker process (``python -m cowmrp.domains.cow.projection``) for each other
resolution; ``project_scenarios`` is the single-resolution core.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

PHYSICAL_QUANTITIES = {"milk_produced_kg": "milk produced, kg", "dmi_kg": "dry matter intake, kg", "enteric_ch4_g": "enteric methane, g",
                       "respiratory_co2_kg": "respiratory CO2, kg", "nitrogen_intake_g": "nitrogen intake, g", "manure_n_g": "manure nitrogen, g"}
UNITS = ("month", "week", "day")


def validate_scenarios(scenarios):
    """Raise ValueError for a missing label, a lactation outside 1-10 or an unknown time_unit (before any model is built)."""
    if not scenarios:
        raise ValueError("scenarios is empty: give at least one scenario")
    for i, sc in enumerate(scenarios):
        if sc.get("time_unit") not in UNITS:
            raise ValueError(f"Scenario {i} ('{sc.get('label')}'): time_unit must be 'month', 'week' or 'day'.")
        if "label" not in sc or not (1 <= int(sc.get("lactation", 0)) <= 10):
            raise ValueError(f"Scenario {i}: needs a label and lactation 1-10.")


def project_scenarios(scenarios, years=2, annual_discount=0.95, n_trajectories=0, seed=1, model=None, verbose=True):
    """Project each scenario of the CURRENT time resolution. Returns {index: result dict} (JSON-serialisable).

    ``model``: an already built ``cowmrp.model.Model`` of the cow (built here when None). Every scenario's ``time_unit`` must equal
    the resolution of this process (``cowmrp.domains.cow.parameters.TIME_UNIT``); use ``compare_cows`` to mix resolutions.
    """
    import numpy as np
    import cowmrp.domains.cow.parameters as P
    from cowmrp.mrp import core as mc, simulator as sim
    from cowmrp.model import build_model
    from cowmrp.domains.cow.states import repro_status
    from cowmrp.domains.cow.rewards import BASE_REWARD_SPECS, replacement_cost_vector
    from cowmrp.domains.cow.transitions import get_replacement_state_id

    validate_scenarios(scenarios)
    unit, U = P.TIME_UNIT, P.UNIT_DAYS
    wrong = [sc["label"] for sc in scenarios if sc["time_unit"] != unit]
    if wrong:
        raise ValueError(f"this process runs at {unit!r} resolution but scenarios {wrong} ask for another: use compare_cows() or set COW_TIME_UNIT before importing cowmrp")
    model = model or build_model("cow")
    states, probs, rewards, matrix = model.states, model.probs, model.rewards, model.P
    rid = get_replacement_state_id(states)
    n_steps = mc.steps_per_year(U) * years
    gamma = mc.annual_to_per_step_gamma(annual_discount, U)

    # reward matrix: signed monetary components, then total, environmental rewards if switched on, then physical quantities
    money = [(s["name"], s["sign"] * np.asarray(rewards[s["name"]], float)) for s in BASE_REWARD_SPECS]
    money.append(("total", np.asarray(rewards["total"], float)))
    for k in ("reward_nitrogen", "reward_enteric_methane", "reward_manure_methane", "reward_n2o"):
        if k in rewards and np.abs(np.asarray(rewards[k], float)).sum() > 0:
            money.append((k, np.asarray(rewards[k], float)))
    money.append(("replacement_cost_memo", -replacement_cost_vector(states, probs)))     # memo: NOT part of "total"
    phys = [(k, np.asarray(rewards[k], float)) for k in PHYSICAL_QUANTITIES if k in rewards]
    names = [n for n, _ in money] + [n for n, _ in phys]
    R = np.column_stack([v for _, v in money] + [v for _, v in phys])
    n_money = len(money)

    status = np.array(repro_status(states["dp"].values))
    lac = states["lactation"].values
    dim_bin = np.minimum(((states["dim"].values + 1) * U) // 30, 35).astype(int)     # 30-day bins of days in milk (mean age during the step)
    results = {}
    for idx, sc in enumerate(scenarios):
        dim = sc["dim"] if "dim" in sc else P.dim_step_of_days(sc.get("dim_days", 0))
        dp = sc["dp"] if "dp" in sc else P.dp_step_of_days(sc.get("dp_days", 0))
        try:
            s0 = mc.find_state_id(states, lactation=sc["lactation"], dim=dim, dp=dp)
        except ValueError as e:
            near = states[(states["lactation"] == sc["lactation"]) & ((states["dp"] > 0) == (dp > 0))].copy()
            near["gap"] = (near["dim"] - dim).abs() + (near["dp"] - dp).abs()
            ex = near.sort_values("gap").head(5)
            sugg = "\n".join(f"   dict(lactation={int(r.lactation)}, dim_days={int((r.dim + 1) * U)}, dp_days={int(r.dp * U)})   # model steps dim={int(r.dim)}, dp={int(r.dp)}" for r in ex.itertuples())
            raise ValueError(f"\nScenario '{sc['label']}' ({unit}): this state does not exist in the model.\n"
                             f"A pregnant cow must have been bred between day 50 and day 300 in milk (dim_days - dp_days must lie in that window); one {unit} step = {U} day(s).\n"
                             f"Closest valid states, in calendar days:\n{sugg}\n(technical detail: {str(e).splitlines()[0]})") from None
        pi_u = mc.make_initial_distribution(states, lactation=sc["lactation"], dim=dim, dp=dp)
        pi_t = np.zeros_like(pi_u)
        T = n_steps
        exp = np.zeros((T, R.shape[1]))
        shares_status = np.zeros((T, 3)); shares_lac = np.zeros((T, P.MAX_LACTATION)); dist_dim = np.zeros((T, 36))
        orig = np.zeros(T)
        for t in range(T):
            pi = pi_u + pi_t                                              # distribution over states at step t
            exp[t] = pi @ R
            for j, s in enumerate(("Open", "Pregnant", "Dry")):
                shares_status[t, j] = pi[status == s].sum()
            shares_lac[t] = np.bincount(lac - 1, weights=pi, minlength=P.MAX_LACTATION)
            dist_dim[t] = np.bincount(dim_bin, weights=pi, minlength=36)
            orig[t] = pi_u.sum()
            pi_u, pi_t = mc.distribution_recursive_origin_tracked(matrix, pi_u, pi_t, 1, rid)   # advance one step
        disc = gamma ** np.arange(T)
        wmat = np.ones((T, R.shape[1]))
        wmat[:, :n_money] = disc[:, None]                                   # money is discounted, physical quantities are not
        cum = np.cumsum(exp * wmat, axis=0)
        # consistency with the engine's own finite-horizon value iteration
        with contextlib.redirect_stdout(io.StringIO()):
            V = mc.evaluate_finite_horizon(matrix, rewards["total"], gamma, years, U)
        total_col = names.index("total")
        check = abs(float(V[s0]) - float(cum[-1, total_col])) / max(1.0, abs(float(V[s0])))
        # same distribution as the engine's plain recursion
        end_ref = mc.distribution_recursive(matrix, mc.make_initial_distribution(states, lactation=sc["lactation"], dim=dim, dp=dp), T)
        check_dist = float(np.abs(end_ref - (pi_u + pi_t)).sum())
        r = {"label": sc["label"], "unit": unit, "U": U, "start": {"lactation": sc["lactation"], "dim_step": int(dim), "dp_step": int(dp)},
             "t_days": (np.arange(T) * U).tolist(), "names": names, "n_money": n_money, "exp_per_step": exp.tolist(), "cum": cum.tolist(),
             "status": shares_status.tolist(), "lactation_shares": shares_lac.tolist(), "dim_bins": dist_dim.tolist(), "original_still_here": orig.tolist(),
             "gamma": gamma, "finite_horizon_value": float(V[s0]), "check_vs_finite_horizon": check, "check_distribution": check_dist,
             "n_states": int(len(states))}
        if n_trajectories > 0:
            hist = sim.run_monte_carlo(matrix, [s0], T, n_sims=n_trajectories, seed=seed)
            ids = np.array([h[:T, 0] for h in hist])                         # (n_sims, T)
            per = R[ids]                                                      # (n_sims, T, n_cols)
            cum_sims = np.cumsum(per * wmat[None, :, :], axis=1)
            r["mc"] = {"p10": np.percentile(cum_sims, 10, axis=0).tolist(), "p50": np.percentile(cum_sims, 50, axis=0).tolist(), "p90": np.percentile(cum_sims, 90, axis=0).tolist(),
                       "n": n_trajectories}
        results[idx] = r
        if verbose:
            print(f"[{unit}] '{sc['label']}': {T} steps, cumulative discounted total {cum[-1, total_col]:,.0f} US$ "
                  f"(engine finite-horizon value {float(V[s0]):,.0f}; relative difference {check:.1e}; distribution check {check_dist:.1e})")
    return results


def compare_cows(scenarios, years=2, annual_discount=0.95, n_trajectories=0, seed=1, verbose=True):
    """Project every scenario, whatever its time resolution. Returns {scenario index: result dict}, in the order of ``scenarios``.

    Scenarios of the resolution this process already runs at are projected here; each other resolution is projected in a worker
    process (the cow domain fixes its resolution at import). Raises ValueError for an invalid scenario or a state the model does not have.
    """
    import cowmrp.domains.cow.parameters as P
    validate_scenarios(scenarios)
    results = {}
    for unit in UNITS:
        idx = [i for i, sc in enumerate(scenarios) if sc["time_unit"] == unit]
        if not idx:
            continue
        if verbose:
            print(f"--- {unit.upper()}: {len(idx)} scenario(s)")
        sub = [scenarios[i] for i in idx]
        if unit == P.TIME_UNIT:
            part = project_scenarios(sub, years, annual_discount, n_trajectories, seed, verbose=verbose)
        else:
            part = _run_worker(unit, sub, years, annual_discount, n_trajectories, seed, verbose)
        for k, i in enumerate(idx):
            results[i] = part[k]
    return results


def _run_worker(unit, scenarios, years, annual_discount, n_trajectories, seed, verbose):
    with tempfile.TemporaryDirectory() as tmp:
        job, out = os.path.join(tmp, "job.json"), os.path.join(tmp, "out.json")
        json.dump({"scenarios": scenarios, "years": years, "annual_discount": annual_discount, "n_trajectories": n_trajectories,
                   "seed": seed, "verbose": verbose}, open(job, "w"))
        p = subprocess.run([sys.executable, "-m", "cowmrp.domains.cow.projection", "--worker", unit, job, out],
                           env=dict(os.environ, COW_TIME_UNIT=unit), capture_output=not verbose, text=True)
        if p.returncode:
            raise ValueError((p.stderr or p.stdout or f"worker for {unit} failed").strip()[-3000:])
        return {int(k): v for k, v in json.load(open(out)).items()}


if __name__ == "__main__":        # worker entry point: python -m cowmrp.domains.cow.projection --worker <unit> <job.json> <out.json>
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _job = json.load(open(sys.argv[3]))
        try:
            _res = project_scenarios(_job["scenarios"], _job["years"], _job["annual_discount"], _job["n_trajectories"], _job["seed"], verbose=_job["verbose"])
        except ValueError as e:
            sys.exit(str(e))
        json.dump(_res, open(sys.argv[4], "w"))
    else:
        sys.exit(__doc__)

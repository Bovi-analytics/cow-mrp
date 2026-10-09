"""
runners/benchmark.py
====================
Computational benchmark of the Cow MRP at several state-space sizes / time resolutions.

ORCHESTRATOR (default):  python runners/benchmark.py
    Reads runners/config.py, runs each model in EXPERIMENTS as its own clean subprocess (so different
    state-space sizes never contaminate each other's timing/memory numbers), collects each worker's
    profiling summary and produces:
        outputs/benchmark/benchmark_summary.csv     -- the comparison table
        outputs/benchmark/comparison_<metric>.png   -- one bar chart per metric
        outputs/benchmark/<Model Name>/...          -- each model's raw outputs (csv/npz/parquet,
                                                       milk trajectories + plots, worker log, result json)

WORKER (internal):  python runners/benchmark.py --worker <worker_config.json>
    Runs ONE model end-to-end (states -> probabilities -> transitions -> rewards -> finite/infinite
    horizon -> milk trajectories -> saves), profiling wall time per phase, CPU %, peak RAM, GPU memory
    (if available) and per-format save time / file size, and writes a JSON summary. The model's overrides
    are baked into the CHILD process environment (COW_TIME_UNIT etc.) before it starts importing, which
    is what keeps each model's state space isolated. (Formerly benchmark.py + benchmark_worker.py.)

This file only USES the MRP engine (mrp/) and the cow domain (cow/); it contains no model logic.
"""
import os
import sys
import json
import time
import threading
import platform
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runners import config

# resource module is Unix/Linux only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


import numpy as np
import psutil
from scipy.sparse import save_npz


# ============================================================
# PROFILING HELPERS
# ============================================================

class CPUSampler:
    """Background-thread sampler for CPU% (of this process) over the run."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self.proc = psutil.Process(os.getpid())

    def _loop(self):
        self.proc.cpu_percent(interval=None)  # prime (first call is always 0)
        while not self._stop.is_set():
            self.samples.append(self.proc.cpu_percent(interval=None))
            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return float(np.mean(self.samples)) if self.samples else 0.0


# Track peak memory manually on Windows (psutil doesn't give us peak RSS)
_peak_memory_mb = 0.0

def _update_peak_memory():
    """Update the global peak memory tracker."""
    global _peak_memory_mb
    current = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    _peak_memory_mb = max(_peak_memory_mb, current)


def peak_ram_mb():
    """Peak RSS for this process so far, in MB.
    On Unix/Linux: uses resource.getrusage (accurate kernel-tracked value).
    On Windows: manually tracked via periodic psutil checks -- the
    background sampler will have called _update_peak_memory during the run.
    """
    if HAS_RESOURCE:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    else:
        # Windows: return the tracked peak
        return _peak_memory_mb


def gpu_memory_mb():
    """Best-effort GPU memory read. Returns None if no GPU backend is
    available/used -- this pipeline is pure-CPU (scipy sparse) today."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


class Timings:
    """Simple named-phase stopwatch."""
    def __init__(self):
        self.data = {}

    def track(self, name):
        return _Timer(self, name)


class _Timer:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.owner.data[self.name] = time.perf_counter() - self.t0


class MemorySampler(threading.Thread):
    """Background sampler for Windows: periodically checks memory and
    updates the global peak tracker."""
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            _update_peak_memory()
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2) if os.path.exists(path) else 0.0


# ============================================================
# BATCHED TRAJECTORY HELPERS (diagnostic/reporting layer only)
# ============================================================
# Reporting on milk + up to 5 environmental quantities per tracked state
# means the SAME distribution-evolution sequence (pi_{t+1} = pi_t @ P) is
# needed 6 times over -- the distribution itself doesn't depend on which
# metric vector we later dot it against. mrp_core.evaluate_state_trajectory
# (and its convergence variant) each redo that evolution independently
# per call, which is fine for one quantity but wasteful for several.
# These two helpers replicate mrp_core's exact math/semantics -- same
# per-step "not cumulative" definition, same convergence criterion and
# (trajectory, steps_run, converged) contract -- just evolving pi ONCE
# and dotting it against every requested metric vector each step. This
# is a benchmark-script-local optimization, not a Core/Engine change:
# nothing here computes a new kind of value, it only avoids recomputing
# the same distribution sequence redundantly.

def multi_metric_trajectory(matrix, metric_vectors, pi0, num_steps):
    """Batched mrp_core.evaluate_state_trajectory: metric_vectors is a
    dict {name: vector}; returns dict {name: np.array(num_steps)}."""
    pi = pi0.copy()
    out = {name: np.empty(num_steps) for name in metric_vectors}
    for t in range(num_steps):
        pi = pi @ matrix
        for name, vec in metric_vectors.items():
            out[name][t] = pi @ vec
    return out


def multi_metric_trajectory_to_convergence(matrix, metric_vectors, pi0, tol=1e-6, max_steps=2000):
    """Batched mrp_core.evaluate_state_trajectory_to_convergence.
    Returns (dict {name: np.array(steps_run)}, steps_run, converged)."""
    pi = pi0.copy()
    trajectories = {name: [] for name in metric_vectors}
    converged = False
    for t in range(max_steps):
        pi_next = pi @ matrix
        for name, vec in metric_vectors.items():
            trajectories[name].append(pi_next @ vec)
        if np.abs(pi_next - pi).sum() < tol:
            pi = pi_next
            converged = True
            break
        pi = pi_next
    steps_run = len(next(iter(trajectories.values())))
    return {name: np.array(vals) for name, vals in trajectories.items()}, steps_run, converged


# ============================================================
# MAIN WORKER
# ============================================================

def worker_main(config_path):
    global _peak_memory_mb
    _peak_memory_mb = 0.0

    with open(config_path) as f:
        cfg = json.load(f)

    name = cfg["name"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    T = Timings()
    sampler = CPUSampler().start()

    # On Windows, also start the memory sampler
    mem_sampler = None
    if not HAS_RESOURCE:
        mem_sampler = MemorySampler(interval=0.1)
        mem_sampler.start()

    # ------------------------------------------------------------
    # Import cow.* / mrp_core AFTER env vars are set
    # ------------------------------------------------------------
    from mrp.core import (
        annual_to_per_step_gamma, assemble_matrix,
        validate_row_stochasticity, evaluate_finite_horizon,
        evaluate_infinite_horizon, steps_per_year,
        export_transitions, export_rewards,
        make_initial_distribution,
    )
    # Note: mrp_core.evaluate_state_trajectory[_to_convergence] are the
    # single-metric versions this file's own multi_metric_trajectory*
    # helpers (above) replicate exactly, batched across quantities --
    # not imported here since nothing below calls the single-metric form.
    from mrp.engine import build_states
    from cow.parameters import UNIT_DAYS, dim_step_of_days, dp_step_of_days

    def _days_to_step(key, days):
        # real days -> model step by the mid-step (entry-offset) convention; any other *_days key is plain rounding
        if key == 'dim_days':
            return dim_step_of_days(days)
        if key == 'dp_days':
            return dp_step_of_days(days)
        return round(days / UNIT_DAYS)
    from cow.states import variable_domains, derived_variables
    from cow.probabilities import compute_probabilities
    from cow.rewards import compute_rewards
    from cow.transitions import compute_transitions
    # (cow.state_query was removed from modular structure; use mrp_core directly)

    gamma = annual_to_per_step_gamma(cfg["gamma_annual"], UNIT_DAYS)

    with T.track("build_states"):
        from cow.states import build_cow_states
        states = build_cow_states()
    num_states = len(states)

    with T.track("probabilities"):
        probs = compute_probabilities(states)

    with T.track("transitions"):
        rows, cols, vals = compute_transitions(states, probs)
        matrix = assemble_matrix(states, rows, cols, vals)
        validate_row_stochasticity(matrix, states)

    with T.track("rewards"):
        rewards = compute_rewards(states, probs)

    V_finite = None
    if cfg["run_finite"]:
        with T.track("finite_horizon"):
            V_finite = evaluate_finite_horizon(
                matrix, rewards["total"], gamma, cfg["num_years"], UNIT_DAYS
            )

    V_infinite = None
    if cfg["run_infinite"]:
        with T.track("infinite_horizon"):
            V_infinite = evaluate_infinite_horizon(matrix, rewards["total"], gamma)

    # ------------------------------------------------------------
    # Value comparisons (diagnostic state-value differences -- pregnancy
    # value, replacement value, ...). Read-only: uses the V(s) already
    # computed above, never touches rewards['total']. Modular by design --
    # see cow/value_comparisons.py's VALUE_COMPARISON_FUNCTIONS registry
    # and config.py's VALUE_COMPARISONS list; adding/removing a comparison
    # never requires a change here.
    # ------------------------------------------------------------
    from cow.value_comparisons import VALUE_COMPARISON_FUNCTIONS

    V_for_comparisons = V_infinite if V_infinite is not None else V_finite
    value_comparisons_info = []
    if V_for_comparisons is not None:
        for entry in cfg.get("value_comparisons", []):
            fn = VALUE_COMPARISON_FUNCTIONS.get(entry["comparison"])
            if fn is None:
                value_comparisons_info.append({
                    "label": entry["label"], "comparison": entry["comparison"],
                    "error": f"unknown comparison '{entry['comparison']}' -- not in "
                             f"cow.value_comparisons.VALUE_COMPARISON_FUNCTIONS",
                })
                continue
            # Same real-days -> model-steps convention as TRACK_STATES above.
            params = {
                (k[:-5] if k.endswith("_days") else k):
                    (_days_to_step(k, v) if k.endswith("_days") else v)
                for k, v in entry["params"].items()
            }
            try:
                result = fn(states, V_for_comparisons, **params)
            except ValueError as e:
                value_comparisons_info.append({
                    "label": entry["label"], "comparison": entry["comparison"], "error": str(e),
                })
                continue
            value_comparisons_info.append({
                "label": entry["label"], "comparison": entry["comparison"],
                "params": entry["params"],
                "value": float(result["value"]),
                "V_state": float(result["V_state"]),
                "V_comparison": float(result["V_comparison"]),
                "horizon_used": "infinite" if V_infinite is not None else "finite",
            })

    # ------------------------------------------------------------
    # Milk production + environmental-emissions trajectories for tracked
    # states.
    #
    # Environmental quantities reuse the exact same generic, domain-
    # agnostic mrp_core.evaluate_state_trajectory()/
    # evaluate_state_trajectory_to_convergence() machinery milk already
    # uses -- no Core/Engine change, no re-derivation of any biology.
    # cow/rewards/environment/*.py remains the sole source of truth for
    # these values; this layer only reads rewards[...] (already computed
    # above) and reports/plots it, in each quantity's own native unit
    # (no unit conversion, so plotted values trace 1:1 to model output).
    # ------------------------------------------------------------
    ENV_QUANTITIES = [
        ("enteric_ch4_g", "Enteric CH4", "g CH4 (this step)"),
        ("manure_ch4_kg", "Manure CH4", "kg CH4 (this step)"),
        ("respiratory_co2_kg", "Respiratory CO2", "kg CO2 (this step)"),
        ("n2o_total_kg", "N2O (total)", "kg N2O (this step)"),
        ("reward_environment_total", "Environmental reward (total)", "$ (this step)"),
    ]
    env_quantities_available = [(k, lbl, u) for k, lbl, u in ENV_QUANTITIES if k in rewards]
    env_quantities_skipped = [k for k, _, _ in ENV_QUANTITIES if k not in rewards]

    trajectory_rows = []
    env_trajectory_rows = []
    trajectory_info = []
    with T.track("trajectories"):
        milk_vector = rewards["milk_produced_kg"]
        finite_steps = steps_per_year(UNIT_DAYS) * cfg["num_years"]

        for s in cfg["track_states"]:
            try:
                # Convert days to steps for state lookup
                dim_step = dim_step_of_days(s.get("dim_days", 0))
                dp_step = dp_step_of_days(s.get("dp_days", 0))

                pi0 = make_initial_distribution(
                    states, lactation=s["lactation"], dim=dim_step, dp=dp_step,
                )
            except ValueError as e:
                trajectory_info.append({"label": s["label"], "error": str(e)})
                continue

            # Milk + every available environmental quantity share the
            # exact same pi0/matrix, so their distribution evolution is
            # identical -- compute it ONCE (multi_metric_trajectory*)
            # rather than once per quantity. Numerically identical to
            # calling mrp_core's single-metric functions N times; see
            # the helpers' docstrings above.
            all_vectors = {"milk_produced_kg": milk_vector}
            all_vectors.update({key: rewards[key] for key, _l, _u in env_quantities_available})

            finite_results = multi_metric_trajectory(matrix, all_vectors, pi0, finite_steps)
            inf_results, steps_run, converged = multi_metric_trajectory_to_convergence(
                matrix, all_vectors, pi0,
                tol=cfg["trajectory_tol"], max_steps=cfg["trajectory_max_steps"],
            )

            traj_finite = finite_results["milk_produced_kg"]
            traj_inf = inf_results["milk_produced_kg"]
            for t, val in enumerate(traj_finite, start=1):
                trajectory_rows.append({
                    "state_label": s["label"], "horizon": "finite",
                    "step": t, "days": t * UNIT_DAYS, "milk_kg_this_step": val,
                })
            for t, val in enumerate(traj_inf, start=1):
                trajectory_rows.append({
                    "state_label": s["label"], "horizon": "infinite",
                    "step": t, "days": t * UNIT_DAYS, "milk_kg_this_step": val,
                })

            info = {
                "label": s["label"], "steady_state_milk_kg_per_step": float(traj_inf[-1]),
                "infinite_steps_run": steps_run, "infinite_converged": converged,
            }

            # Same pi0, same finite horizon/step convention as milk above
            # -- directly comparable on the same day axis.
            for key, _label, _unit in env_quantities_available:
                env_finite = finite_results[key]
                env_inf = inf_results[key]
                for t, val in enumerate(env_finite, start=1):
                    env_trajectory_rows.append({
                        "state_label": s["label"], "horizon": "finite", "quantity": key,
                        "step": t, "days": t * UNIT_DAYS, "value": val,
                    })
                for t, val in enumerate(env_inf, start=1):
                    env_trajectory_rows.append({
                        "state_label": s["label"], "horizon": "infinite", "quantity": key,
                        "step": t, "days": t * UNIT_DAYS, "value": val,
                    })

                # Summary statistics over the finite horizon (per tracked
                # state), plus the converged infinite-horizon steady-state
                # per-step value.
                info[f"{key}_cumulative_finite"] = float(env_finite.sum())
                info[f"{key}_mean_finite"] = float(env_finite.mean())
                info[f"{key}_final_finite"] = float(env_finite[-1])
                info[f"{key}_steady_state_per_step"] = float(env_inf[-1])

            trajectory_info.append(info)

    import pandas as pd
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_csv_path = os.path.join(out_dir, "milk_trajectories.csv")
    trajectory_df.to_csv(trajectory_csv_path, index=False)

    env_trajectory_df = pd.DataFrame(env_trajectory_rows)
    env_trajectory_csv_path = os.path.join(out_dir, "environmental_trajectories.csv")
    env_trajectory_df.to_csv(env_trajectory_csv_path, index=False)

    if value_comparisons_info:
        pd.DataFrame(value_comparisons_info).to_csv(
            os.path.join(out_dir, "value_comparisons.csv"), index=False
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Milk plots (finite + infinite, one figure each, all tracked states
    # overlaid) -- UNCHANGED from before this extension.
    plot_paths = {}
    if not trajectory_df.empty:
        for horizon in ["finite", "infinite"]:
            sub = trajectory_df[trajectory_df["horizon"] == horizon]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, grp in sub.groupby("state_label"):
                ax.plot(grp["days"], grp["milk_kg_this_step"], label=label)
            ax.set_xlabel("Days since state entry")
            ax.set_ylabel("Milk produced this step (kg)")
            ax.set_title(f"{name} -- {horizon} horizon milk trajectory (per step, not cumulative)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(out_dir, f"milk_trajectory_{horizon}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            plot_paths[horizon] = path

    # Environmental plots -- same figure size/style/legend/x-axis
    # convention as the milk plots above, one figure per quantity per
    # horizon (kept separate rather than merged, since different
    # quantities use different native units -- e.g. enteric CH4 in g vs.
    # manure CH4 in kg -- and merging them would require a unit
    # conversion this pass deliberately avoids, per "plotted values come
    # directly from the model's actual environmental outputs").
    env_plot_paths = {}
    if not env_trajectory_df.empty:
        for key, label, unit in env_quantities_available:
            for horizon in ["finite", "infinite"]:
                sub = env_trajectory_df[
                    (env_trajectory_df["quantity"] == key) & (env_trajectory_df["horizon"] == horizon)
                ]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 5))
                for state_label, grp in sub.groupby("state_label"):
                    ax.plot(grp["days"], grp["value"], label=state_label)
                ax.set_xlabel("Days since state entry")
                ax.set_ylabel(f"{label} ({unit})")
                ax.set_title(f"{name} -- {horizon} horizon {label} trajectory (per step, not cumulative)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                path = os.path.join(out_dir, f"environmental_{key}_trajectory_{horizon}.png")
                fig.savefig(path, dpi=120)
                plt.close(fig)
                env_plot_paths[f"{key}_{horizon}"] = path

    # ------------------------------------------------------------
    # Saves: CSV / NPZ / Parquet, each timed and sized individually
    # ------------------------------------------------------------
    save_timings = {}
    file_sizes = {}

    transitions_long = export_transitions(matrix, states, unit_prefix="u")
    reward_df = export_rewards(states, rewards, unit_prefix="u")

    # Retain the full per-state value vector(s) instead of discarding them
    # after collapsing to V_finite_mean/V_infinite_mean below -- state_id
    # order matches states' row order (mrp_engine.build_states sets
    # states['state_id'] = states.index), same order V_finite/V_infinite
    # are indexed in, so this is a direct positional column assignment.
    # See docs/superpowers/specs/2026-08-17-state-value-difference-framework-design.md.
    if V_finite is not None:
        reward_df["V_finite"] = V_finite
    if V_infinite is not None:
        reward_df["V_infinite"] = V_infinite

    if cfg["save_npz"]:
        npz_path = os.path.join(out_dir, "transition_matrix.npz")
        t0 = time.perf_counter()
        save_npz(npz_path, matrix)
        save_timings["save_npz"] = time.perf_counter() - t0
        file_sizes["npz_mb"] = file_size_mb(npz_path)

    if cfg["save_parquet"]:
        tr_parquet = os.path.join(out_dir, "transitions_long.parquet")
        t0 = time.perf_counter()
        transitions_long.to_parquet(tr_parquet, index=False)
        save_timings["save_parquet_transitions"] = time.perf_counter() - t0
        file_sizes["parquet_transitions_mb"] = file_size_mb(tr_parquet)

        rw_parquet = os.path.join(out_dir, "reward_vector.parquet")
        t0 = time.perf_counter()
        reward_df.to_parquet(rw_parquet, index=False)
        save_timings["save_parquet_rewards"] = time.perf_counter() - t0
        file_sizes["parquet_rewards_mb"] = file_size_mb(rw_parquet)

    if cfg["save_csv"]:
        tr_csv = os.path.join(out_dir, "transitions_long.csv")
        t0 = time.perf_counter()
        transitions_long.to_csv(tr_csv, index=False)
        save_timings["save_csv_transitions"] = time.perf_counter() - t0
        file_sizes["csv_transitions_mb"] = file_size_mb(tr_csv)

        rw_csv = os.path.join(out_dir, "reward_vector.csv")
        t0 = time.perf_counter()
        reward_df.to_csv(rw_csv, index=False)
        save_timings["save_csv_rewards"] = time.perf_counter() - t0
        file_sizes["csv_rewards_mb"] = file_size_mb(rw_csv)

    # ------------------------------------------------------------
    # Wrap up profiling and write summary
    # ------------------------------------------------------------
    cpu_percent_avg = sampler.stop()
    if mem_sampler:
        mem_sampler.stop()
    peak_ram = peak_ram_mb()
    gpu_mb = gpu_memory_mb()

    core_pipeline_time = sum(T.data.get(k, 0.0) for k in
                              ["build_states", "probabilities", "transitions",
                               "rewards", "finite_horizon", "infinite_horizon"])
    save_time = sum(save_timings.values())
    total_time = core_pipeline_time + T.data.get("trajectories", 0.0) + save_time

    summary = {
        "name": name,
        "num_states": num_states,
        "nnz_transitions": int(matrix.nnz),
        "timings_sec": {**T.data, **save_timings},
        "core_pipeline_time_sec": core_pipeline_time,
        "save_time_sec": save_time,
        "total_time_sec": total_time,
        "cpu_percent_avg": cpu_percent_avg,
        "peak_ram_mb": peak_ram,
        "gpu_mem_mb": gpu_mb,
        "file_sizes_mb": file_sizes,
        "V_finite_mean": float(V_finite.mean()) if V_finite is not None else None,
        "V_infinite_mean": float(V_infinite.mean()) if V_infinite is not None else None,
        "trajectory_info": trajectory_info,
        "value_comparisons": value_comparisons_info,
        "plot_paths": plot_paths,
        "env_plot_paths": env_plot_paths,
        "env_quantities_plotted": [k for k, _, _ in env_quantities_available],
        "env_quantities_skipped": env_quantities_skipped,
        "out_dir": out_dir,
        "platform": platform.system(),
    }

    with open(cfg["result_path"], "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{name}] done: {num_states:,} states, {total_time:.2f}s total, "
          f"peak RAM {peak_ram:.1f} MB, CPU% avg {cpu_percent_avg:.1f}")




# ============================================================
# ORCHESTRATOR
# ============================================================

def build_worker_config(experiment, out_dir, result_path):
    """Merge one EXPERIMENTS entry with the global TRACK_STATES/trajectory
    settings into the flat dict benchmark_worker.py expects."""
    return {
        "name": experiment["name"],
        "gamma_annual": experiment["gamma_annual"],
        "num_years": experiment["num_years"],
        "run_finite": experiment["run_finite"],
        "run_infinite": experiment["run_infinite"],
        "save_csv": experiment["save_csv"],
        "save_npz": experiment["save_npz"],
        "save_parquet": experiment["save_parquet"],
        "track_states": config.TRACK_STATES,
        "value_comparisons": config.VALUE_COMPARISONS,
        "trajectory_tol": config.TRAJECTORY_CONVERGENCE_TOL,
        "trajectory_max_steps": config.TRAJECTORY_MAX_STEPS,
        "out_dir": out_dir,
        "result_path": result_path,
    }


def run_one_model(experiment, repo_root, output_dir):
    name = experiment["name"]
    model_dir = os.path.join(output_dir, name.replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    result_path = os.path.join(model_dir, "worker_result.json")
    worker_cfg = build_worker_config(experiment, model_dir, result_path)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(worker_cfg, f)
        cfg_path = f.name

    # Environment overrides take effect BEFORE the child even starts
    # importing -- that's what keeps each model's state space isolated.
    env = {**os.environ, **{k: str(v) for k, v in experiment["overrides"].items()}}

    log_path = os.path.join(model_dir, "worker_log.txt")
    print(f"\n=== Running {name} ===")
    print(f"  overrides: {experiment['overrides']}")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", cfg_path],
            cwd=repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )

    os.unlink(cfg_path)

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) -- see {log_path}")
        with open(log_path) as f:
            print(f.read()[-2000:])
        return None

    with open(result_path) as f:
        result = json.load(f)
    print(f"  OK: {result['num_states']:,} states, "
          f"{result['total_time_sec']:.2f}s, "
          f"peak RAM {result['peak_ram_mb']:.1f} MB")
    return result


def build_summary_table(results):
    rows = []
    for r in results:
        fs = r["file_sizes_mb"]
        rows.append({
            "Model": r["name"],
            "States": r["num_states"],
            "Time (s)": round(r["core_pipeline_time_sec"], 2),
            "Save Time (s)": round(r["save_time_sec"], 2),
            "Total Time (s)": round(r["total_time_sec"], 2),
            "CPU %": round(r["cpu_percent_avg"], 1),
            "Peak RAM (MB)": round(r["peak_ram_mb"], 1),
            "GPU Mem (MB)": round(r["gpu_mem_mb"], 1) if r["gpu_mem_mb"] is not None else "N/A",
            "CSV Size (MB)": round(fs.get("csv_transitions_mb", 0) + fs.get("csv_rewards_mb", 0), 2),
            "NPZ Size (MB)": round(fs.get("npz_mb", 0), 2),
            "Parquet Size (MB)": round(fs.get("parquet_transitions_mb", 0) + fs.get("parquet_rewards_mb", 0), 2),
        })
    return pd.DataFrame(rows)


def plot_comparison(df, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    ax.bar(df["Model"], numeric, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for experiment in config.EXPERIMENTS:
        result = run_one_model(experiment, repo_root, output_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo models completed successfully -- nothing to summarize.")
        return

    summary_df = build_summary_table(results)
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    for metric, ylabel in [
        ("Time (s)", "seconds"), ("Save Time (s)", "seconds"),
        ("CPU %", "% of one core"), ("Peak RAM (MB)", "MB"),
        ("CSV Size (MB)", "MB"), ("NPZ Size (MB)", "MB"), ("Parquet Size (MB)", "MB"),
    ]:
        plot_comparison(summary_df, metric, ylabel,
                         os.path.join(output_dir, f"comparison_{metric.split(' ')[0].lower()}.png"))

    print(f"Saved comparison plots to: {output_dir}")
    print(f"Per-model outputs (csv/npz/parquet, milk trajectory plots, logs) are under:")
    for r in results:
        print(f"  {r['out_dir']}")




if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2])
    else:
        main()

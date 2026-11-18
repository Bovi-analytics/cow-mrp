"""
runners/_shared/scale_benchmark.py
==================================
How the cost and the accuracy of handling a whole population grow with its size, for every subject that has an adapter
(cow, victor, youngstock, herd = cow + youngstock coupled). Every number comes from calling the engine / the coupling;
nothing is re-implemented here, so a change in mrp/ or in a domain is picked up on the next run.

    python runners/_shared/scale_benchmark.py [--subject cow,victor,youngstock,herd | all] [--sizes 10,100,1000,10000]
         [--years 3] [--time-units month,week] [--error-limit 0.1] [--cap-minutes 30] [--sims 1] [--herd-sims 10]
         [--methods sim,mrp_distinct,mrp_all] [--seed 42] [--quick] [--report-only]
    python runners/<module>/run.py --scale            (one subject, defaults)

DAY RESOLUTION IS OPT-IN: the default is --time-units month,week. Add it explicitly:  --time-units month,week,day
(cow and youngstock only; Victor has one native resolution, month; the herd runs month and week: its sampled mode keeps an array of
cow-state count x steps, 10.6 GiB at day, which does not fit in memory here). A resolution a subject does not support is recorded as
skipped with the reason, never silently dropped.

HORIZON: HORIZON_YEARS below (also --years) is the one place that sets the benchmark horizon. The herd PROJECTION tool
(runners/herd/farm/population.py), the saved baselines and the checks keep their own 5-year horizons; this benchmark does not touch them.

WHAT IS COMPARED (entity = one animal; the exact method is always run and is the reference)
  value subjects (cow, victor)   sim: one sampled path per entity, discounted reward summed along it | mrp_distinct: engine forward solve
                                 from each DISTINCT start state | mrp_all: one backward solve for every state (exact; entities look up)
  youngstock (no rewards)        sim: N sampled newborn calves followed for the horizon | exact: one forward evolution of the matrix;
                                 compared on the share of the cohort ending in each outcome (graduated, sold, died, culled, stillborn, alive)
  herd (cow + youngstock)        sampled: whole animals, --herd-sims runs per farm size | expected: exact expected counts (rng=None);
                                 compared on cows, live youngstock, calves sold and discounted net income at the horizon

ACCURACY METRIC (written to accuracy.csv, one row per metric)
  error_pct = 100 * |estimate - exact| / max(|exact|, floor),   floor = 0.01 * scale
  scale = the natural size of the quantity: portfolio of cows: sum of |exact values|; a share of a cohort: 1 (the whole cohort);
  an animal count in the herd: the number of cows at the start; discounted net income: its own exact value.
  The floor is what keeps near-zero outcomes (a share of 1e-6, a count near 0) from producing huge relative errors: below 1 % of the scale
  the error is measured against that 1 %, i.e. it becomes an absolute error relative to the scale.
  Metrics that DECIDE (compared with --error-limit): value: the portfolio total (sum over entities); youngstock: every outcome share;
  herd: every quantity, both its BIAS (|mean of the sampled runs - expected|) and its SPREAD (half of the 5 %-95 % band of single sampled
  runs). Reported but not deciding: per-entity RMSE (value subjects). A method "meets the limit" when the largest deciding error_pct <= limit;
  the summary picks the cheapest (smallest wall time) method that does.

CAP: a method whose predicted time (from the previous size's measured time, scaled by the work) exceeds --cap-minutes is skipped and
recorded in skipped.csv with the reason; a run that raises MemoryError is recorded the same way. Nothing is dropped silently.

OUTPUT  runners/<subject>/outputs/benchmark_scale/  timing.csv  accuracy.csv  skipped.csv  manifest.json  (+ scale_benchmark.csv for value
        subjects)  index.html  plotly.min.js ;  runners/_shared/outputs/benchmark_scale/index.html = summary across subjects.
ADAPTERS  runners/<module>/scale_inputs.py: RESOLUTIONS, NOTES, inputs(args) -> dict (see the adapters). 'package.module:function' also works.
"""
import os
import sys
import time
import json
import argparse
import threading
import subprocess
import contextlib
import io
import platform
import importlib
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root
from runners._shared import common
common.set_time_unit_from_argv()

import numpy as np
import pandas as pd
import psutil

from cowmrp.mrp.core import (annual_to_per_step_gamma, steps_per_year, evaluate_finite_horizon,
                      evaluate_state_trajectory, validate_row_stochasticity, distribution_recursive)
from cowmrp.mrp.simulator import run_stochastic
def gpu_memory_mb():
    """Best-effort GPU memory read (MB); None if no GPU backend is available (the pipeline is pure CPU today)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / (1024 ** 2)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None

ROOT = common.project_root()

# ---- the settings one is expected to change --------------------------------------------------------------------
HORIZON_YEARS = 3                      # benchmark horizon in years (also --years). Other tools keep their own horizons.
ERROR_LIMIT_PCT = 0.1                  # "accurate enough": largest deciding error_pct (also --error-limit; 1 is the looser comparison)
CAP_MINUTES = 30                       # skip a method predicted to take longer than this per cell (also --cap-minutes)
DEFAULT_SIZES = "10,100,1000,10000"
DEFAULT_TIME_UNITS = "month,week"      # day is opt-in: --time-units month,week,day
HERD_SIMS = 10                         # sampled herd runs per farm size (bias and spread need more than one)
# -----------------------------------------------------------------------------------------------------------------
SUBJECTS = ("cow", "victor", "youngstock", "herd")
ALL_METHODS = ("sim", "mrp_distinct", "mrp_all")
FLOOR_FRACTION = 0.01


# ============================================================
# PROFILING
# ============================================================
class Phase:
    """Context manager: wall time, mean CPU %, peak RAM (MB, sampled) and RAM growth, GPU memory (MB or NaN)."""
    def __init__(self, interval=0.02):
        self.interval = interval

    def __enter__(self):
        self.proc = psutil.Process(os.getpid())
        self.rss0 = self.proc.memory_info().rss / 2**20
        self.peak, self.cpu, self._stop = self.rss0, [], threading.Event()
        self.proc.cpu_percent(None)

        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss / 2**20)
                self.cpu.append(self.proc.cpu_percent(None))
                time.sleep(self.interval)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self._stop.set(); self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss / 2**20)
        self.cpu_pct = float(np.mean(self.cpu)) if self.cpu else 0.0
        gpu = gpu_memory_mb()
        self.gpu_mb = np.nan if gpu is None else gpu

    def row(self):
        return dict(wall_s=self.wall, cpu_pct=self.cpu_pct, peak_ram_mb=self.peak,
                    ram_growth_mb=self.peak - self.rss0, gpu_mb=self.gpu_mb)


def quiet(fn, *a, **k):
    """The engine prints a per-year summary in evaluate_finite_horizon; keep the benchmark output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class Budget:
    """Predicts a method's time at a new size from its last measured size (time per unit of work x work) and says when to skip."""
    def __init__(self, cap_minutes):
        self.cap_s = 60.0 * cap_minutes
        self.cap_minutes = cap_minutes
        self.last = {}                                     # method -> (wall_s, units)

    def check(self, method, units, runs=1):
        if method not in self.last:
            return None
        wall, u0 = self.last[method]
        predicted = wall / max(u0, 1) * units * runs
        if predicted > self.cap_s:
            return (f"predicted {predicted / 60:.1f} min (measured {wall:.1f} s for {u0:,} units, now {units:,}) exceeds the "
                    f"{self.cap_minutes:g} min cap")
        return None

    def record(self, method, units, wall):
        self.last[method] = (wall, units)


def err_pct(err_abs, exact, scale):
    """100 * |error| / max(|exact|, floor), floor = FLOOR_FRACTION * scale (see the module docstring)."""
    floor = FLOOR_FRACTION * abs(scale)
    return 100.0 * err_abs / max(abs(exact), floor, 1e-300), floor


def acc_row(base, method, metric, exact, estimate, scale, kind="error", decides=True, err_abs=None):
    err_abs = abs(estimate - exact) if err_abs is None else err_abs
    pct, floor = err_pct(err_abs, exact, scale)
    return dict(base, method=method, metric=metric, kind_of_error=kind, exact=float(exact), estimate=float(estimate),
                error_abs=float(err_abs), scale=float(scale), floor=float(floor), error_pct=float(pct), decides=bool(decides))


# ============================================================
# THE METHODS (engine calls only)
# ============================================================
def values_sim(matrix, reward, gamma, T, start_ids, n_sims, seed):
    """Discounted reward summed along sampled paths of states at t = 0..T-1 (same definition as the finite-horizon V)."""
    disc = gamma ** np.arange(T)
    total = np.zeros(len(start_ids))
    for k in range(n_sims):
        hist = run_stochastic(matrix, start_ids, T - 1, seed=seed + k)        # shape (T, n)
        total += disc @ reward[hist]
    return total / n_sims


def values_distinct(matrix, reward, gamma, T, start_ids):
    """V_T(s) = R[s] + sum_{t=0}^{T-2} gamma^(t+1) * E[R(s_{t+1})], from the engine's forward expectations."""
    n = matrix.shape[0]
    disc = gamma ** np.arange(1, T)
    out = {}
    for s in np.unique(start_ids):
        pi0 = np.zeros(n); pi0[s] = 1.0
        traj = evaluate_state_trajectory(matrix, reward, pi0, T - 1)          # E[R] at steps 1..T-1
        out[s] = reward[s] + disc @ traj
    return np.array([out[s] for s in start_ids]), len(out)


def values_all(matrix, reward, gamma, years, unit_days):
    return quiet(evaluate_finite_horizon, matrix, reward, gamma, years, unit_days)


# ============================================================
# ONE CELL = one subject at one time resolution
# ============================================================
def _timing_row(base, phase_name, n, method, ph, runs=1, **extra):
    return dict(base, phase=phase_name, n_entities=n, method=method, runs=runs, **ph.row(),
                per_entity_ms=(1000 * ph.wall / max(n, 1)) if n else np.nan, **extra)


def _skip(base, n, method, reason):
    print(f"  N={n:>6,} {method:<13} SKIPPED: {reason}")
    return dict(base, n_entities=n, method=method, reason=reason)


def run_value(d, args, base, build_ph):
    states, matrix, reward = d["states"], d["matrix"].tocsr(), np.asarray(d["reward"], float)
    unit_days, n_states = d["unit_days"], len(d["states"])
    if d.get("strict", True):
        validate_row_stochasticity(matrix, states)
    gamma = d["gamma"] if d.get("gamma") is not None else annual_to_per_step_gamma(args.gamma_annual, unit_days)
    T = steps_per_year(unit_days) * args.years
    pi = d.get("start_distribution")
    base = dict(base, unit_days=unit_days, n_states=n_states, nnz=matrix.nnz, years=args.years, steps=T)
    timing = [_timing_row(base, "build_model", 0, "", build_ph)]
    accuracy, skipped = [], []
    budget, rng = Budget(args.cap_minutes), np.random.default_rng(args.seed)
    print(f"[{base['subject']} | {base['time_unit']}] {n_states:,} states, {matrix.nnz:,} nnz, {T} steps ({args.years} y), build {build_ph.wall:.1f}s")
    for N in args.sizes:
        start = rng.choice(n_states, size=N, p=pi / pi.sum()) if pi is not None else rng.integers(0, n_states, size=N)
        n_distinct = int(len(np.unique(start)))
        results = {}
        with Phase() as ph:                                        # the exact reference is always computed
            V = values_all(matrix, reward, gamma, args.years, unit_days)
        results["mrp_all"] = (V[start], ph, {})
        budget.record("mrp_all", 1, ph.wall)
        for method in ("sim", "mrp_distinct"):
            if method not in args.methods:
                continue
            units = N if method == "sim" else n_distinct
            reason = budget.check(method, units)
            if reason:
                skipped.append(_skip(base, N, method, reason)); continue
            try:
                with Phase() as ph:
                    if method == "sim":
                        v, extra = values_sim(matrix, reward, gamma, T, start, args.sims, args.seed), {}
                    else:
                        v, nd = values_distinct(matrix, reward, gamma, T, start); extra = {"distinct_states": nd}
                results[method] = (v, ph, extra)
                budget.record(method, units, ph.wall)
            except MemoryError as e:
                skipped.append(_skip(base, N, method, f"failed: MemoryError ({str(e)[:120]})"))
        exact = results["mrp_all"][0]
        scale_total = float(np.abs(exact).sum())
        for method in ALL_METHODS:
            if method not in results or (method == "mrp_all" and "mrp_all" not in args.methods):
                continue
            v, ph, ex = results[method]
            err = v - exact
            over = ph.wall > 60 * args.cap_minutes
            timing.append(_timing_row(base, "value_entities", N, method, ph, over_cap=over,
                                      distinct_states=ex.get("distinct_states", n_distinct), mean_value=float(v.mean()),
                                      portfolio_error_pct=100 * float(err.sum() / exact.sum()),
                                      rmse_per_entity=float(np.sqrt(np.mean(err ** 2))), max_abs_error=float(np.abs(err).max())))
            ab = dict(base, n_entities=N)
            accuracy.append(acc_row(ab, method, "portfolio_value", exact.sum(), v.sum(), scale_total))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            accuracy.append(acc_row(ab, method, "per_entity_rmse", 0.0, 0.0, float(np.abs(exact).mean()), decides=False,
                                    err_abs=rmse) | {"exact": float(np.abs(exact).mean()), "error_pct": 100 * rmse / max(float(np.abs(exact).mean()), 1e-300)})
            print(f"  N={N:>6,} {method:<13} {ph.wall:9.3f}s {1000 * ph.wall / N:10.3f} ms/entity  peak {ph.peak:7.0f} MB  "
                  f"portfolio err {accuracy[-2]['error_pct']:.4f}%")
    return timing, accuracy, skipped


def run_outcomes(d, args, base, build_ph):
    """No rewards: follow N sampled newborns for the horizon and compare the share ending in each outcome with the exact forward evolution."""
    states, Q, labels = d["states"], d["matrix"].tocsr(), np.asarray(d["labels"])
    unit_days, n_states = d["unit_days"], len(states)
    validate_row_stochasticity(Q, states)
    T = steps_per_year(unit_days) * args.years
    pi0 = np.asarray(d["start_distribution"], float)
    pi0 = pi0 / pi0.sum()
    names = sorted(set(labels))
    onehot = np.stack([(labels == o).astype(float) for o in names], axis=1)         # state -> outcome
    base = dict(base, unit_days=unit_days, n_states=n_states, nnz=Q.nnz, years=args.years, steps=T)
    timing = [_timing_row(base, "build_model", 0, "", build_ph)]
    accuracy, skipped = [], []
    budget, rng = Budget(args.cap_minutes), np.random.default_rng(args.seed)
    print(f"[{base['subject']} | {base['time_unit']}] {n_states:,} states, {Q.nnz:,} nnz, {T} steps ({args.years} y), build {build_ph.wall:.1f}s")
    for N in args.sizes:
        with Phase() as ph:
            exact = quiet(distribution_recursive, Q, pi0, T) @ onehot               # exact outcome shares at the horizon
        timing.append(_timing_row(base, "outcome_shares", N, "exact", ph)); budget.record("exact", 1, ph.wall)
        ab = dict(base, n_entities=N)
        for k, o in enumerate(names):
            accuracy.append(acc_row(ab, "exact", f"share_{o}", exact[k], exact[k], 1.0))
        reason = budget.check("sim", N)
        if reason:
            skipped.append(_skip(base, N, "sim", reason)); continue
        try:
            start = rng.choice(n_states, size=N, p=pi0)
            with Phase() as ph:
                hist = run_stochastic(Q, start, T, seed=args.seed)
                est = onehot[hist[-1]].mean(axis=0)
            budget.record("sim", N, ph.wall)
            timing.append(_timing_row(base, "outcome_shares", N, "sim", ph, over_cap=ph.wall > 60 * args.cap_minutes,
                                      max_share_error_pct=100 * float(np.abs(est - exact).max())))
            for k, o in enumerate(names):
                accuracy.append(acc_row(ab, "sim", f"share_{o}", exact[k], est[k], 1.0))
            print(f"  N={N:>6,} sim           {ph.wall:9.3f}s {1000 * ph.wall / N:10.3f} ms/entity  peak {ph.peak:7.0f} MB  "
                  f"largest share error {100 * np.abs(est - exact).max():.3f} pp")
        except MemoryError as e:
            skipped.append(_skip(base, N, "sim", f"failed: MemoryError ({str(e)[:120]})"))
    return timing, accuracy, skipped


def run_herd(d, args, base, build_ph):
    """Expected projection (exact) against whole-animal sampled projections of an N-cow farm, with bias and spread reported apart."""
    n_states, unit_days = len(d["states"]), d["unit_days"]
    base = dict(base, unit_days=unit_days, n_states=n_states, nnz=d["nnz"], years=args.years, steps=int(round(args.years * 365.25 / unit_days)))
    timing = [_timing_row(base, "build_model", 0, "", build_ph)]
    accuracy, skipped = [], []
    budget, rng = Budget(args.cap_minutes), np.random.default_rng(args.seed)
    S = args.herd_sims
    print(f"[{base['subject']} | {base['time_unit']}] {n_states:,} cow states, {base['steps']} steps ({args.years} y), build {build_ph.wall:.1f}s")
    for N in args.sizes:
        pre = d["prepare"](N, rng)
        with Phase() as ph:
            exact = d["project"](pre, None)
        timing.append(_timing_row(base, "herd_projection", N, "expected", ph)); budget.record("expected", 1, ph.wall)
        ab = dict(base, n_entities=N)
        scales = d["scales"](N, exact)
        for q, v in exact.items():
            accuracy.append(acc_row(ab, "expected", q, v, v, scales[q], kind="error"))
        reason = budget.check("sampled", N, runs=S)
        if reason:
            skipped.append(_skip(base, N, "sampled", reason)); continue
        try:
            runs = []
            with Phase() as ph:
                for k in range(S):
                    runs.append(d["project"](pre, np.random.default_rng(args.seed + k)))
            budget.record("sampled", N, ph.wall / S)
            ph.wall_total = ph.wall
            for key in ("wall_s",):
                pass
            row = _timing_row(base, "herd_projection", N, "sampled", ph, runs=S)
            row["wall_s"] = ph.wall / S; row["per_entity_ms"] = 1000 * ph.wall / S / N          # per ONE sampled run
            row["over_cap"] = ph.wall > 60 * args.cap_minutes
            timing.append(row)
            for q, v in exact.items():
                arr = np.array([r[q] for r in runs])
                p05, p95 = np.percentile(arr, [5, 95])
                accuracy.append(acc_row(ab, "sampled", q, v, arr.mean(), scales[q], kind="bias"))
                accuracy.append(acc_row(ab, "sampled", q, v, arr.mean(), scales[q], kind="spread", err_abs=(p95 - p05) / 2))
            print(f"  N={N:>6,} sampled      {ph.wall / S:9.3f}s per run (x{S})  peak {ph.peak:7.0f} MB  expected {exact['cows_end']:.0f} cows -> "
                  f"sampled mean {np.mean([r['cows_end'] for r in runs]):.0f}")
        except MemoryError as e:
            skipped.append(_skip(base, N, "sampled", f"failed: MemoryError ({str(e)[:120]})"))
    return timing, accuracy, skipped


KINDS = {"value": run_value, "outcomes": run_outcomes, "herd": run_herd}


def load_adapter(subject):
    if ":" in subject:
        mod, _, func = subject.partition(":")
        m = importlib.import_module(mod)
        return getattr(m, func), tuple(getattr(m, "RESOLUTIONS", ("month", "week", "day"))), "_shared"
    m = importlib.import_module(f"runners.{subject}.scale_inputs")
    return m.inputs, tuple(m.RESOLUTIONS), subject


def run_cell(subject, unit, args):
    """One subject at one resolution, in this process (the time unit is read once at import by the domain parameters)."""
    if unit:
        os.environ["COW_TIME_UNIT"] = unit
    inputs, _, _ = load_adapter(subject)
    base = dict(subject=subject, time_unit=unit)
    with Phase() as build_ph:
        d = inputs(args)
    return KINDS[d.get("kind", "value")](d, args, base, build_ph)


# ============================================================
# ORCHESTRATION
# ============================================================
def machine_info():
    vm = psutil.virtual_memory()
    import scipy
    return dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                cpu_cores_physical=psutil.cpu_count(logical=False), cpu_threads=psutil.cpu_count(logical=True),
                ram_total_gb=round(vm.total / 2**30, 1), gpu_memory_mb=gpu_memory_mb())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="all", help="comma list of cow,victor,youngstock,herd, 'all', or package.module:function")
    ap.add_argument("--domain", default=None, help="(old name of --subject)")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--years", type=int, default=HORIZON_YEARS, help="benchmark horizon (default HORIZON_YEARS)")
    ap.add_argument("--sims", type=int, default=1, help="sampled paths per entity for the sim method (value subjects)")
    ap.add_argument("--herd-sims", type=int, default=HERD_SIMS, help="sampled herd projections per farm size")
    ap.add_argument("--time-units", default=DEFAULT_TIME_UNITS, help="comma list; day is opt-in: month,week,day")
    ap.add_argument("--time-unit", default=None, help="(internal) one resolution for this process")
    ap.add_argument("--part-dir", default=None, help="(internal) where a per-cell subprocess writes its rows")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--gamma-annual", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--error-limit", type=float, default=ERROR_LIMIT_PCT, help="percent; the cheapest method with largest deciding error <= this is chosen")
    ap.add_argument("--cap-minutes", type=float, default=CAP_MINUTES)
    ap.add_argument("--quick", action="store_true", help="sizes 10,100 and month only (minutes)")
    ap.add_argument("--report-only", action="store_true", help="redraw the pages from the existing csv files (e.g. with another --error-limit)")
    a = ap.parse_args(argv)
    if a.domain:
        a.subject = a.domain
    if a.quick:
        a.sizes, a.time_units = "10,100", "month"
    a.sizes = [int(x) for x in a.sizes.split(",")]
    a.methods = [m for m in a.methods.split(",") if m]
    bad = set(a.methods) - set(ALL_METHODS)
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; choose from {ALL_METHODS}")
    subjects = list(SUBJECTS) if a.subject == "all" else [s for s in a.subject.split(",") if s]
    units = [u for u in a.time_units.split(",") if u]

    if a.time_unit is not None:                                       # subprocess: one cell
        subj = subjects[0]
        timing, accuracy, skipped = run_cell(subj, a.time_unit, a)
        os.makedirs(a.part_dir, exist_ok=True)
        for name, rows in (("timing", timing), ("accuracy", accuracy), ("skipped", skipped)):
            pd.DataFrame(rows).to_csv(os.path.join(a.part_dir, f"{name}.csv"), index=False)
        return

    from runners._shared import scale_report
    written = []
    for subj in subjects:
        _, resolutions, module = load_adapter(subj)
        out = common.module_outputs(module, "benchmark_scale")
        if not a.report_only:
            T, A, S = [], [], []
            for u in units:
                if u not in resolutions:
                    reason = (f"{subj} supports {', '.join(resolutions)} only" +
                              (" (its own native resolution)" if subj == "victor" else
                               " (the sampled herd keeps an array of cow states x steps: 10.6 GiB at day, out of memory on this machine)" if subj == "herd" and u == "day" else ""))
                    S.append(dict(subject=subj, time_unit=u, n_entities="", method="", reason=reason))
                    print(f"[{subj} | {u}] SKIPPED: {reason}")
                    continue
                part = os.path.join(out, f"_part_{u}")
                cmd = [sys.executable, os.path.abspath(__file__), "--subject", subj, "--time-unit", u, "--part-dir", part,
                       "--sizes", ",".join(map(str, a.sizes)), "--years", str(a.years), "--sims", str(a.sims), "--herd-sims", str(a.herd_sims),
                       "--methods", ",".join(a.methods), "--gamma-annual", str(a.gamma_annual), "--seed", str(a.seed),
                       "--cap-minutes", str(a.cap_minutes)]
                r = subprocess.run(cmd)
                if r.returncode != 0:
                    S.append(dict(subject=subj, time_unit=u, n_entities="", method="", reason=f"the run failed (exit code {r.returncode}); see the log"))
                    continue
                for lst, name in ((T, "timing"), (A, "accuracy"), (S, "skipped")):
                    p = os.path.join(part, f"{name}.csv")
                    try:
                        lst.append(pd.read_csv(p))
                    except (FileNotFoundError, pd.errors.EmptyDataError):
                        pass                                         # an empty list of rows (e.g. nothing skipped)
                import shutil; shutil.rmtree(part, ignore_errors=True)
            cat = lambda L: pd.concat([x if isinstance(x, pd.DataFrame) else pd.DataFrame([x]) for x in L], ignore_index=True) if L else pd.DataFrame()
            timing, accuracy, skipped = cat(T), cat(A), cat(S)
            timing.to_csv(os.path.join(out, "timing.csv"), index=False)
            accuracy.to_csv(os.path.join(out, "accuracy.csv"), index=False)
            skipped.to_csv(os.path.join(out, "skipped.csv"), index=False)
            if "mean_value" in timing:                                 # the old cow file, same name and columns
                legacy = timing[timing["phase"].isin(["build_model", "value_entities"])].rename(columns={"subject": "domain"})
                legacy.to_csv(os.path.join(out, "scale_benchmark.csv"), index=False)
            with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(dict(subject=subj, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), machine=machine_info(),
                               horizon_years=a.years, sizes=a.sizes, time_units_requested=units, methods=a.methods, sims=a.sims,
                               herd_sims=a.herd_sims, seed=a.seed, cap_minutes=a.cap_minutes, error_limit_pct=a.error_limit,
                               floor_fraction=FLOOR_FRACTION, skipped=skipped.to_dict("records") if len(skipped) else []), f, indent=1, default=str)
        written.append(scale_report.write_subject_page(subj, module, a.error_limit))
    summary = scale_report.write_summary(subjects, a.error_limit)
    print("subject pages:", *written, sep="\n  ")
    print("summary:", summary)


if __name__ == "__main__":
    main()

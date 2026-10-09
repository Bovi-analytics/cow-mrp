"""
runners/farm/simulate.py
========================
Projects a herd forward with the MRP and reports where it goes, what it is worth, and how much of it has been
replaced. (Formerly farm_simulator.py + the generic stochastic helpers it called.)

    python runners/farm/simulate.py [--mode expected|stochastic|monte_carlo] [--steps 24] [--seed 42] [--sims 100]
                                    [--farm-csv outputs/farm/farm_data.csv] [--groupers status,lactation,dim_stage]
                                    [--compare-methods] [--no-values] [--time-unit month|week|day] [--out-dir DIR]

WHAT IT DOES
  * reads the herd CSV (generate.py) and refuses one built under a different cow configuration;
  * projects it with the MRP's own transition matrix: `expected` (exact distribution evolution, with the herd split
    into cows never replaced vs replaced at least once -- the split reconstructs the whole distribution exactly),
    `stochastic` (one sampled path per cow) or `monte_carlo` (many sampled paths: mean / std / 5-95 %);
  * summarises by cow groupings (reproductive status incl. the derived "do not breed" region, lactation, DIM stage);
  * links the herd to the model's VALUES (same infinite-horizon solve as the MRP): per cow the value V, the
    REPLACEMENT value V(cow) - V(replacement state), and for pregnant cows the PREGNANCY value
    V(cow) - V(same lactation and DIM, open). Read-only diagnostics -- never added to a reward;
  * writes CSVs and an HTML dashboard (matplotlib images embedded; + an interactive Plotly page if plotly is installed).

No model logic lives here: states, probabilities, transitions, rewards and values come from cow/ and mrp/.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))   # project root
from runners.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import argparse
import base64
import io
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mrp import core as mrp_core
from mrp.simulator import (
    run_stochastic, run_monte_carlo, compare_methods, run_expected_origin_tracked,
)
from cow.parameters import TIME_UNIT, UNIT_DAYS, LAST_BREEDING_DIM_DAYS
from cow.rewards import compute_rewards
from cow.transitions import get_replacement_state_id
from runners.farm.build import build_engine, current_config, FARM_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# COW GROUPINGS -- vectorised: each returns one label per STATE
# ============================================================
def _age_dim(states):                         # mean days in milk during the step (mid-step convention)
    return (states["dim"].values + 1) * UNIT_DAYS


def g_status(states):
    dp = states["dp"].values
    from cow.parameters import DRY_OFF_DP
    return np.where(dp >= DRY_OFF_DP, "Dry", np.where(dp > 0, "Pregnant",
                    np.where(_age_dim(states) > LAST_BREEDING_DIM_DAYS, "Open (do not breed)", "Open (breeding)")))


def g_lactation(states):
    lac = states["lactation"].values
    return np.where(lac == 1, "Lactation 1", np.where(lac == 2, "Lactation 2", "Lactation 3+"))


def g_lactation_exact(states):
    return np.array([f"Lac {int(l):02d}" for l in states["lactation"].values])


def g_dim_stage(states):
    a = _age_dim(states)
    return np.select([a <= 60, a <= 150, a <= 300], ["DIM 0-60 d", "DIM 61-150 d", "DIM 151-300 d"], default="DIM 300+ d")


GROUPERS = {"status": g_status, "lactation": g_lactation, "lactation_exact": g_lactation_exact, "dim_stage": g_dim_stage}
DEFAULT_GROUPERS = ["status", "lactation", "dim_stage"]


def group_probs(states, pi, grouper):
    return pd.Series(pi).groupby(GROUPERS[grouper](states)).sum().sort_index()


def group_counts(states, ids, grouper):
    return pd.Series(GROUPERS[grouper](states)[np.asarray(ids)]).value_counts().sort_index()


def mc_summary(trajs, labels, steps):
    """{step: DataFrame[mean, std, p05, p95] indexed by label} of the per-simulation group counts."""
    cats = sorted(set(labels))
    out = {}
    for s in range(steps + 1):
        counts = np.array([[np.sum(labels[t[s]] == c) for c in cats] for t in trajs], dtype=float)      # (sims, labels)
        out[s] = pd.DataFrame({"mean": counts.mean(0), "std": counts.std(0), "p05": np.percentile(counts, 5, axis=0),
                               "p95": np.percentile(counts, 95, axis=0)}, index=cats)
    return out


def averaged_group_counts(states, id_arrays, grouper):
    return pd.concat([group_counts(states, ids, grouper) for ids in id_arrays], axis=1).fillna(0.0).mean(axis=1)


# ============================================================
# VALUES -- linked to the MRP's own solve (read-only diagnostics)
# ============================================================
def herd_values(states, probs, matrix, farm_df, rid, gamma_annual=0.95):
    """Per-cow V, replacement value and (pregnant cows) pregnancy value from the model's infinite-horizon solve."""
    total = compute_rewards(states, probs)["total"]
    gamma = mrp_core.annual_to_per_step_gamma(gamma_annual, UNIT_DAYS)
    V = mrp_core.evaluate_infinite_horizon(matrix, total, gamma)
    sid = farm_df["state_id"].values
    key = pd.MultiIndex.from_arrays([states["lactation"], states["dim"], states["dp"]])
    pos = pd.Series(states["state_id"].values, index=key)
    open_ids = pos.reindex(pd.MultiIndex.from_arrays([states["lactation"].values[sid], states["dim"].values[sid],
                                                      np.zeros(len(sid), dtype=int)])).values
    out = farm_df[["cow_id", "state_id", "lactation", "dim_days", "dp_days"]].copy() if "dim_days" in farm_df else farm_df[["cow_id", "state_id", "lactation"]].copy()
    out["V"] = V[sid]
    out["replacement_value"] = V[sid] - V[rid]
    pregnant = states["dp"].values[sid] > 0
    with np.errstate(invalid="ignore"):
        pv = np.where(pregnant & ~np.isnan(open_ids), V[sid] - V[np.nan_to_num(open_ids).astype(int)], np.nan)
    out["pregnancy_value"] = pv
    out["status"] = g_status(states)[sid]
    return out


# ============================================================
# PLOTS / DASHBOARD
# ============================================================
def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_compare(cur, pred, title):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    idx = sorted(set(cur.index) | set(pred.index))
    x = np.arange(len(idx))
    ax.bar(x - 0.2, [cur.get(i, 0) for i in idx], 0.4, label="current")
    ax.bar(x + 0.2, [pred.get(i, 0) for i in idx], 0.4, label="predicted")
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=25, ha="right"); ax.set_ylabel("cows"); ax.set_title(title); ax.legend()
    return _b64(fig)


def plot_series(evolution, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    df = pd.DataFrame(evolution).fillna(0.0)
    for c in df.columns:
        ax.plot(df.index, df[c], label=c)
    ax.set_xlabel(xlabel); ax.set_ylabel("cows"); ax.set_title(title); ax.legend(fontsize=7)
    return _b64(fig)


def plot_replacement(regular, replaced, xlabel):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    reg, rep = pd.DataFrame(regular).fillna(0.0), pd.DataFrame(replaced).fillna(0.0)
    ax.stackplot(reg.index, reg.sum(axis=1), rep.sum(axis=1), labels=["never replaced", "replaced at least once"])
    ax.set_xlabel(xlabel); ax.set_ylabel("cows"); ax.set_title("Herd split: original cows vs replacements"); ax.legend(loc="center right", fontsize=8)
    return _b64(fig)


def write_dashboard(path, meta, current, predicted, evolution, regular, replaced, values_df, mode, steps):
    xlabel = f"{TIME_UNIT}s elapsed"
    imgs = []
    for g in current:
        imgs.append((g, plot_compare(current[g], predicted[g], f"{g}: current vs predicted after {steps} {TIME_UNIT}(s)"),
                     plot_series(evolution[g], f"{g}: herd composition over time", xlabel)))
    repl = plot_replacement(regular, replaced, xlabel) if regular is not None else None
    html = [f"<html><head><meta charset='utf-8'><title>Farm projection</title><style>body{{font-family:Segoe UI,sans-serif;"
            f"max-width:1300px;margin:auto;padding:1rem}}img{{max-width:48%}}td,th{{padding:2px 8px;border-bottom:1px solid #ddd}}</style></head><body>",
            f"<h1>Farm projection ({mode}, {steps} {TIME_UNIT}s = {steps * UNIT_DAYS} days, {meta['n_cows']:,} cows)</h1>",
            f"<p>State space: {meta['num_states']:,} states, 1 step = {UNIT_DAYS} day(s). Herd built under {meta['config']}.</p>"]
    if values_df is not None:
        by = values_df.groupby("status").agg(cows=("V", "size"), mean_V=("V", "mean"), mean_replacement_value=("replacement_value", "mean"),
                                             mean_pregnancy_value=("pregnancy_value", "mean")).round(1)
        html.append("<h2>Values (model infinite-horizon V; read-only diagnostics)</h2>"
                    f"<p>Herd V (sum): <b>{values_df['V'].sum():,.0f}</b> &nbsp; replacement value (sum): <b>{values_df['replacement_value'].sum():,.0f}</b> &nbsp; "
                    f"pregnancy value of pregnant cows (sum): <b>{np.nansum(values_df['pregnancy_value']):,.0f}</b></p>" + by.to_html())
    for g, a, b in imgs:
        html.append(f"<h2>{g}</h2><img src='data:image/png;base64,{a}'><img src='data:image/png;base64,{b}'>")
    if repl:
        html.append(f"<h2>Replacement tracking</h2><img src='data:image/png;base64,{repl}'>")
    html.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def write_plotly(path, evolution, steps):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    gs = list(evolution)
    fig = make_subplots(rows=len(gs), cols=1, subplot_titles=gs)
    for r, g in enumerate(gs, 1):
        df = pd.DataFrame(evolution[g]).fillna(0.0)
        for c in df.columns:
            fig.add_trace(go.Scatter(x=list(df.index), y=df[c], name=f"{g}: {c}", mode="lines"), row=r, col=1)
    fig.update_layout(height=320 * len(gs), title=f"Herd composition over {steps} {TIME_UNIT}(s)", xaxis_title=f"{TIME_UNIT}s elapsed")
    fig.write_html(path)
    return True


# ============================================================
# MAIN
# ============================================================
def load_herd(farm_csv, num_states):
    meta_path = os.path.splitext(farm_csv)[0] + ".meta.json"
    live = {**current_config(), "num_states": num_states}
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found -- cannot verify the herd matches the current cow configuration. Regenerate with generate.py.")
    with open(meta_path) as f:
        fm = json.load(f)
    bad = {k: (fm.get(k), v) for k, v in live.items() if fm.get(k) != v}
    if bad:
        raise SystemExit(f"{farm_csv} was generated under a different configuration:\n" +
                         "\n".join(f"  {k}: herd has {a!r}, current is {b!r}" for k, (a, b) in bad.items()) +
                         "\nRegenerate it with generate.py under the same --time-unit (state ids are not comparable across configurations).")
    return pd.read_csv(farm_csv), fm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHAT IT DOES")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="expected", choices=["expected", "stochastic", "monte_carlo"])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--farm-csv", default=os.path.join(FARM_DIR, "farm_data.csv"))
    ap.add_argument("--groupers", default=",".join(DEFAULT_GROUPERS))
    ap.add_argument("--compare-methods", action="store_true")
    ap.add_argument("--no-values", action="store_true")
    ap.add_argument("--time-unit", default=None)
    ap.add_argument("--out-dir", default=os.path.join(FARM_DIR, "results"))
    a = ap.parse_args()
    groupers = a.groupers.split(",")
    os.makedirs(a.out_dir, exist_ok=True)

    states, probs, matrix = build_engine()
    rid = get_replacement_state_id(states)
    farm_df, fmeta = load_herd(a.farm_csv, len(states))
    n_cows, ids0 = len(farm_df), farm_df["state_id"].values
    print(f"{n_cows:,} cows | {len(states):,} states | {TIME_UNIT} ({UNIT_DAYS} d/step) | {a.mode}, {a.steps} steps = {a.steps * UNIT_DAYS} days")

    pi0 = np.bincount(ids0, minlength=len(states)).astype(float)
    pi0 /= pi0.sum()
    current = {g: group_counts(states, ids0, g) for g in groupers}
    evolution = {g: [] for g in groupers}
    regular = replaced = None
    mc_rows = []

    if a.mode == "expected":
        if a.compare_methods:
            cmp_df = compare_methods(matrix, pi0, a.steps)
            cmp_df.to_csv(os.path.join(a.out_dir, "method_comparison.csv"), index=False)
        po, pt = pi0.copy(), np.zeros_like(pi0)
        orig, tag, dists = [po.copy()], [pt.copy()], [pi0.copy()]
        for _ in range(a.steps):
            po, pt = run_expected_origin_tracked(matrix, po, pt, 1, rid)
            orig.append(po.copy()); tag.append(pt.copy()); dists.append(po + pt)
        for step, (d, o, t) in enumerate(zip(dists, orig, tag)):
            assert np.allclose(o + t, d, atol=1e-9), f"regular + replaced does not reconstruct the herd at step {step}"
        for g in groupers:
            evolution[g] = [group_probs(states, d, g) * n_cows for d in dists]
        regular = [group_probs(states, p, "status") * n_cows for p in orig]
        replaced = [group_probs(states, p, "status") * n_cows for p in tag]
        predicted = {g: evolution[g][-1] for g in groupers}
        for step in range(a.steps + 1):
            tot = regular[step].add(replaced[step], fill_value=0).sum()
            assert abs(tot - n_cows) < 1e-6, f"cow count not conserved at step {step}: {tot}"
    elif a.mode == "stochastic":
        hist = run_stochastic(matrix, ids0, a.steps, seed=a.seed)
        for g in groupers:
            evolution[g] = [group_counts(states, hist[s], g) for s in range(a.steps + 1)]
        mask = mrp_core.trajectory_tagged_mask(hist, rid)
        regular = [group_counts(states, hist[s][~mask[s]], "status") for s in range(a.steps + 1)]
        replaced = [group_counts(states, hist[s][mask[s]], "status") for s in range(a.steps + 1)]
        predicted = {g: evolution[g][-1] for g in groupers}
        pd.DataFrame(hist.T, columns=[f"step_{s}" for s in range(a.steps + 1)]).assign(cow_id=farm_df["cow_id"].values).to_csv(
            os.path.join(a.out_dir, "trajectory.csv"), index=False)
    else:
        trajs = run_monte_carlo(matrix, ids0, a.steps, n_sims=a.sims, seed=a.seed)
        predicted = {}
        for g in groupers:
            summ = mc_summary(trajs, GROUPERS[g](states), a.steps)
            evolution[g] = [summ[s]["mean"] for s in range(a.steps + 1)]
            predicted[g] = summ[a.steps]["mean"]
            for s, df in summ.items():
                for label, r in df.iterrows():
                    mc_rows.append({"step": s, "grouper": g, "label": label, "mean": r["mean"], "std": r["std"], "p05": r["p05"], "p95": r["p95"]})
        masks = [mrp_core.trajectory_tagged_mask(t, rid) for t in trajs]
        regular = [averaged_group_counts(states, [t[s][~m[s]] for t, m in zip(trajs, masks)], "status") for s in range(a.steps + 1)]
        replaced = [averaged_group_counts(states, [t[s][m[s]] for t, m in zip(trajs, masks)], "status") for s in range(a.steps + 1)]
        pd.DataFrame(mc_rows).to_csv(os.path.join(a.out_dir, "monte_carlo_summary.csv"), index=False)

    rows = []
    for g in groupers:
        for label in sorted(set(current[g].index) | set(predicted[g].index)):
            c, p = float(current[g].get(label, 0)), float(predicted[g].get(label, 0))
            rows.append({"grouper": g, "label": label, "current": c, "predicted": p, "change": p - c})
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "distribution_summary.csv"), index=False)
    pd.DataFrame({f"step_{s}": pd.Series(regular[s]) for s in range(len(regular))}).to_csv(os.path.join(a.out_dir, "replacement_tracking_regular.csv"))
    pd.DataFrame({f"step_{s}": pd.Series(replaced[s]) for s in range(len(replaced))}).to_csv(os.path.join(a.out_dir, "replacement_tracking_replaced.csv"))

    values_df = None
    if not a.no_values:
        values_df = herd_values(states, probs, matrix, farm_df, rid)
        values_df.to_csv(os.path.join(a.out_dir, "cow_values.csv"), index=False)
        print(f"Herd V {values_df['V'].sum():,.0f} | replacement value {values_df['replacement_value'].sum():,.0f} | "
              f"pregnancy value (pregnant cows) {np.nansum(values_df['pregnancy_value']):,.0f}")

    for g in groupers:
        print(f"\n--- {g} ---")
        print(pd.DataFrame({"current": current[g], "predicted": predicted[g]}).fillna(0).round(1).to_string())
    meta = {"n_cows": n_cows, "num_states": len(states), "config": current_config()}
    write_dashboard(os.path.join(a.out_dir, "dashboard.html"), meta, current, predicted, evolution, regular, replaced, values_df, a.mode, a.steps)
    plotly_ok = write_plotly(os.path.join(a.out_dir, "dashboard_plotly.html"), evolution, a.steps)
    print(f"\nOutputs in {a.out_dir}  (dashboard.html{' + dashboard_plotly.html' if plotly_ok else ''})")


if __name__ == "__main__":
    main()

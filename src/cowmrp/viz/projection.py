"""cowmrp.viz.projection -- tables and Plotly figures for a cow comparison (``cowmrp.domains.cow.projection.compare_cows``).

No model code: it only draws and tabulates what ``compare_cows`` returned. Needs plotly (``pip install cowmrp[plot]``) for the figures;
the tables need only pandas. ``runners/cow/projection.py`` calls these functions.
"""
from __future__ import annotations

import re

from cowmrp.domains.cow.projection import PHYSICAL_QUANTITIES


def day_ticks(total_days):
    """Tick spacing (days) for the time axis: about 8-12 ticks, a multiple of 30 days."""
    return 30 * max(1, -(-total_days // 300))


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60]


def projection_figures(results, years=2, annual_discount=0.95, n_trajectories=0):
    """List of ``(file_stem, title, caption, plotly Figure)``: one figure per reward component / physical quantity (per-day and cumulative,
    all scenarios overlaid) and one distribution figure per scenario."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import qualitative
    figs = []        # (file, title, caption, fig)
    colors = qualitative.D3
    ordered = [results[i] for i in sorted(results)]
    names = ordered[0]["names"]; n_money = ordered[0]["n_money"]
    nice = {"milk_income": "Milk income", "pregnancy_reward": "Pregnancy reward (calf value)", "cull_reward": "Cull reward (salvage - culling cost)", "mortality_cost": "Mortality cost",
            "replacement_cost_memo": "Replacement cost (memo, not in TOTAL)", "feed_cost": "Feed cost", "reproduction_cost": "Reproduction cost (AI, diagnosis, synchronization)", "total": "TOTAL reward",
            "reward_nitrogen": "Nitrogen reward", "reward_enteric_methane": "Enteric methane reward", "reward_manure_methane": "Manure methane reward", "reward_n2o": "N2O reward"}
    nice.update({k: v for k, v in PHYSICAL_QUANTITIES.items()})
    for j, key in enumerate(names):
        money = j < n_money
        unit_txt = "US$" if money else PHYSICAL_QUANTITIES[key].split(", ")[-1]
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                            subplot_titles=(f"Expected {('reward' if money else 'amount')} per day ({unit_txt}/day)",
                                            f"Cumulative expected {('discounted ' if money else '')}{('reward, US$' if money else 'amount, ' + unit_txt)}"))
        for si, r in enumerate(ordered):
            x = np.array(r["t_days"])
            per_day = np.array(r["exp_per_step"])[:, j] / r["U"]
            cum = np.array(r["cum"])[:, j]
            c = colors[si % len(colors)]
            fig.add_trace(go.Scatter(x=x, y=per_day, mode="lines", name=r["label"], legendgroup=str(si), line=dict(color=c, width=2),
                                     hovertemplate="day %{x:.0f}<br>%{y:.3f}<extra>" + r["label"] + "</extra>"), row=1, col=1)
            if "mc" in r:
                fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[np.array(r["mc"]["p90"])[:, j], np.array(r["mc"]["p10"])[::-1, j]], fill="toself", fillcolor=c, opacity=0.15,
                                         line=dict(width=0), name=f"{r['label']}: 10-90 % of {r['mc']['n']} simulated cows", legendgroup=str(si), hoverinfo="skip"), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", name=r["label"], legendgroup=str(si), showlegend=False, line=dict(color=c, width=2),
                                     hovertemplate="day %{x:.0f}<br>%{y:,.0f}<extra>" + r["label"] + "</extra>"), row=1, col=2)
        fig.update_xaxes(title_text="days from now", title_standoff=8, dtick=day_ticks(years * 365), tick0=0, tickangle=0)
        fig.update_layout(template="plotly_white", title=dict(text=f"{nice.get(key, key)} over time", x=0.01), height=520,
                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), margin=dict(l=70, r=30, t=90, b=120))
        cap = (f"{nice.get(key, key)}. Left: expected amount per day at each future time, from the model's own transition and reward vectors "
               f"(a cow that is culled or dies is replaced by a heifer, whose cost and future income are included). "
               + ("Right: running total, discounted at " + f"{annual_discount} per year" + (" (shaded: 10-90 % of simulated individual cows)." if n_trajectories else ".") if money else "Right: running total, not discounted."))
        figs.append((f"reward_{key}", nice.get(key, key), cap, fig))
    # distribution figures
    for r in ordered:
        x = np.array(r["t_days"])
        fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.065,
                            subplot_titles=("Reproductive status", "Lactation number", "Days in milk (30-day bins)", "Original cow still in herd?"))
        st = np.array(r["status"])
        for k, (nm, col) in enumerate((("Open", "#1f77b4"), ("Pregnant", "#2ca02c"), ("Dry", "#ff7f0e"))):
            fig.add_trace(go.Scatter(x=x, y=st[:, k], mode="lines", stackgroup="s", name=nm, line=dict(color=col, width=0.5)), row=1, col=1)
        ls = np.array(r["lactation_shares"])
        from plotly.colors import sample_colorscale
        for L in range(ls.shape[1]):
            if ls[:, L].max() > 1e-4:
                fig.add_trace(go.Scatter(x=x, y=ls[:, L], mode="lines", stackgroup="l", name=f"lactation {L + 1}", line=dict(width=0.5, color=sample_colorscale("Viridis", L / max(ls.shape[1] - 1, 1))[0])), row=1, col=2)
        dd = np.array(r["dim_bins"])
        nb = int(np.argmax(dd.sum(axis=0) < 1e-6)) or dd.shape[1]
        fig.add_trace(go.Heatmap(z=dd[:, :nb].T, x=x, y=np.arange(nb) * 30 + 15, colorscale="Viridis", colorbar=dict(title=dict(text="prob.", side="top"), len=0.5, thickness=10, x=0.742, y=0.5), zmin=0), row=1, col=3)
        orig = np.array(r["original_still_here"])
        fig.add_trace(go.Scatter(x=x, y=orig, mode="lines", name="original cow still in herd", line=dict(color="#d62728", width=2)), row=1, col=4)
        fig.add_trace(go.Scatter(x=x, y=1 - orig, mode="lines", name="replaced by a heifer at least once", line=dict(color="#9467bd", width=2)), row=1, col=4)
        fig.update_xaxes(title_text="days from now", title_standoff=8, dtick=day_ticks(years * 365), tick0=0, tickangle=0)
        fig.update_yaxes(title_text="probability", row=1, col=1); fig.update_yaxes(title_text="probability", row=1, col=2)
        fig.update_yaxes(title_text="days in milk", row=1, col=3)
        s0 = r["start"]
        fig.update_layout(template="plotly_white", height=560, title=dict(text=f"Where the cow will be: {r['label']}", x=0.01),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0), margin=dict(l=70, r=30, t=100, b=130))
        cap = (f"Starting state: lactation {s0['lactation']}, model step dim = {s0['dim_step']}, dp = {s0['dp_step']} ({r['unit']} resolution, step = {r['U']} d, {r['n_states']:,} states). "
               "All panels are probability distributions of ONE cow slot over the future, computed by repeatedly applying the model's transition matrix. Replaced animals are renewed by a heifer, "
               "so the status and lactation panels describe the slot; the last panel isolates the original cow.")
        figs.append((f"distribution_{slug(r['label'])}", f"Distribution over time: {r['label']}", cap, fig))
    return figs


def projection_tables(results, years=2):
    """``(summary DataFrame, {slug(label): time-series DataFrame})``: cumulative value at the end of each year per reward, and the per-day /
    cumulative / status-share series of every scenario (the numbers behind the figures)."""
    import numpy as np, pandas as pd
    ordered = [results[i] for i in sorted(results)]
    names = ordered[0]["names"]; n_money = ordered[0]["n_money"]
    series = {}
    rows = []
    for r in ordered:
        cum = np.array(r["cum"]); t = np.array(r["t_days"]); U = r["U"]
        spy = round(365 / U)
        for j, key in enumerate(names):
            row = {"scenario": r["label"], "time_unit": r["unit"], "reward": key, "money": j < n_money}
            for y in range(1, years + 1):
                row[f"cumulative_year_{y}"] = float(cum[min(y * spy, len(cum)) - 1, j])
            rows.append(row)
        ts = pd.DataFrame({"year": t / 365.0, "days": t})
        for j, key in enumerate(names):
            ts[f"{key}_per_day"] = np.array(r["exp_per_step"])[:, j] / U
            ts[f"{key}_cumulative"] = cum[:, j]
        for k, nm in enumerate(("Open", "Pregnant", "Dry")):
            ts[f"share_{nm}"] = np.array(r["status"])[:, k]
        ts["original_cow_still_in_herd"] = r["original_still_here"]
        series[slug(r['label'])] = ts
    return pd.DataFrame(rows), series

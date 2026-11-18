"""cowmrp.viz.farm -- the farm dashboards and the result files of a herd projection (cowmrp.domains.cow.farm.project_herd).

write_farm_outputs(result, out_dir) writes the result files of a herd projection (runners/cow/farm/simulate.py calls it): distribution_summary.csv,
replacement_tracking_regular.csv / _replaced.csv, cow_values.csv, (stochastic) trajectory.csv, (monte_carlo) monte_carlo_summary.csv,
(compare) method_comparison.csv, and the two dashboards: dashboard.html (static, matplotlib images embedded) and dashboard_plotly.html
(interactive, plotly embedded once so it works offline). Needs matplotlib and plotly
(pip install cowmrp[plot]); a missing plotly only skips the interactive page.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:      # pragma: no cover
    plt = None


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


STATUS_ORDER = ["Open", "Pregnant", "Dry"]
STATUS_COLORS = {"Open": "#3498db", "Pregnant": "#27ae60", "Dry": "#e67e22", "Replaced": "#8e44ad"}
FALLBACK_COLORS = ["#1abc9c", "#e74c3c", "#f1c40f", "#34495e", "#16a085", "#d35400", "#7f8c8d", "#2980b9"]

PAGE_CSS = ("body{font-family:'Segoe UI',sans-serif;margin:0;background:#f8f9fa;color:#2c3e50}"
            ".header{background:linear-gradient(135deg,#2c3e50,#34495e);color:white;padding:2rem;text-align:center}"
            ".container{max-width:1400px;margin:0 auto;padding:2rem}"
            ".section{background:white;border-radius:12px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 2px 8px rgba(0,0,0,0.06)}"
            ".section h2{border-bottom:2px solid #3498db;padding-bottom:0.5rem;margin-bottom:1rem}"
            ".cards{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}"
            ".card{background:white;border-radius:12px;padding:1rem 1.5rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);min-width:150px;text-align:center}"
            ".card-value{font-size:1.8rem;font-weight:700}.card-label{color:#777}.card-sub{font-size:.85rem;color:#888}"
            ".row{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1rem}.row>div{min-width:0;overflow:hidden}.plotly-graph-div{width:100%!important}"
            "table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}"
            "img{max-width:100%}.note{color:#666;font-size:.85rem}")

_REPL_NOTE = ("Original herd = cows never replaced since the start. Replacement herd = cows that entered through the replacement state "
              "(death, culling or max-lactation calving -> a fresh heifer, lactation 1, DIM 0, which is Open), then followed the normal "
              "Open -> Pregnant -> Dry cycle. Together they are the whole herd.")


def _color(label, i=0):
    return STATUS_COLORS.get(label, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


def _frame(series_list, columns=None):
    """steps x labels DataFrame (rows = model steps) from a per-step list of label->cows Series."""
    df = pd.DataFrame(series_list).fillna(0.0)
    return df.reindex(columns=columns, fill_value=0.0) if columns is not None else df


def _dash_frames(evolution, regular, replaced):
    """Everything the two dashboards draw, derived once from the projection the MRP already produced."""
    reg, rep = _frame(regular, STATUS_ORDER), _frame(replaced, STATUS_ORDER)
    flag4 = reg.copy()                                   # original herd by status + everyone who entered as a replacement
    flag4["Replaced"] = rep.sum(axis=1)                  # (the four bands sum to the herd: original + replaced = whole distribution)
    return {"evolution": {g: _frame(evolution[g]) for g in evolution}, "regular": reg, "replaced": rep, "flag4": flag4}


def _summary_rows(current, predicted):
    cur, pred = current.get("status", pd.Series(dtype=float)), predicted.get("status", pd.Series(dtype=float))
    return [(s, float(cur.get(s, 0)), float(pred.get(s, 0))) for s in STATUS_ORDER]


def _section_title(g):
    return {"status": "Status (Open / Pregnant / Dry, + Replaced)", "lactation": "Lactation", "lactation_exact": "Lactation (exact)",
            "dim_stage": "Days in milk stage", "month_in_milk": "Month in milk (state)"}.get(g, g)


def _evo_title(g):
    return "Herd by status over time (original herd + Replaced)" if g == "status" else f"{_section_title(g)}: herd composition over time"


def _header_html(title, meta, mode, steps, extra=""):
    return (f"<div class='header'><h1>{title}</h1><p>Mode: <strong>{mode.upper()}</strong> | {TIME_UNIT.capitalize()}s: <strong>{steps}</strong> "
            f"(= {steps * UNIT_DAYS} days) | Herd: <strong>{meta['n_cows']:,}</strong> cows</p>"
            f"<p>State space: {meta['num_states']:,} states, 1 step = {UNIT_DAYS} day(s). Herd built under {meta['config']}.</p>{extra}</div>")


def _cards_html(rows, steps):
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='card-value' style='color:{_color(s)}'>{c:,.0f}</div><div class='card-label'>{s}</div>"
        f"<div class='card-sub'>{TIME_UNIT} {steps}: {p:,.0f} ({p - c:+,.0f})</div></div>" for s, c, p in rows) + "</div>"


def _summary_table_html(rows, steps):
    body = "".join(f"<tr><td>{s}</td><td>{c:,.1f}</td><td>{p:,.1f}</td><td>{p - c:+,.1f}</td></tr>" for s, c, p in rows)
    return (f"<table><thead><tr><th>Status</th><th>Current</th><th>Predicted ({TIME_UNIT} {steps})</th><th>Change</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _values_html(values_df):
    if values_df is None:
        return ""
    by = values_df.groupby("status").agg(cows=("V", "size"), mean_V=("V", "mean"), mean_replacement_value=("replacement_value", "mean"),
                                         mean_pregnancy_value=("pregnancy_value", "mean")).round(1)
    return ("<div class='section'><h2>Values (model infinite-horizon V; read-only diagnostics)</h2>"
            f"<p>Herd V (sum, reward vector without replacement cost): <b>{values_df['V'].sum():,.0f}</b> &nbsp; replacement value (sum; cow vs a heifer bought now, replacement costs included): <b>{values_df['replacement_value'].sum():,.0f}</b> &nbsp; "
            f"pregnancy value of pregnant cows (sum): <b>{np.nansum(values_df['pregnancy_value']):,.0f}</b></p>" + by.to_html() + "</div>")


# ---------------- static (matplotlib images) ----------------
def plot_compare(cur, pred, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    idx = sorted(set(cur.index) | set(pred.index))
    x = np.arange(len(idx))
    ax.bar(x - 0.2, [cur.get(i, 0) for i in idx], 0.4, label="Current", color="#95a5a6")
    ax.bar(x + 0.2, [pred.get(i, 0) for i in idx], 0.4, label="Predicted", color="#3498db")
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=25, ha="right"); ax.set_ylabel("cows"); ax.set_title(title); ax.legend()
    return _b64(fig)


def plot_series(df, title, xlabel, stacked=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = list(df.columns)
    if stacked:
        ax.stackplot(df.index, *[df[c] for c in cols], labels=cols, colors=[_color(c, i) for i, c in enumerate(cols)])
    else:
        for i, c in enumerate(cols):
            ax.plot(df.index, df[c], label=c, color=_color(c, i))
    ax.set_xlabel(xlabel); ax.set_ylabel("cows"); ax.set_title(title); ax.legend(fontsize=7)
    return _b64(fig)


def plot_overlay(reg, rep, xlabel):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, c in enumerate(reg.columns):
        ax.plot(reg.index, reg[c], color=_color(c, i), label=f"{c} (original)")
        ax.plot(rep.index, rep[c], color=_color(c, i), ls="--", label=f"{c} (replacement)")
    ax.set_xlabel(xlabel); ax.set_ylabel("cows"); ax.set_title("Original vs replacement herd (overlaid)"); ax.legend(fontsize=7, ncol=2)
    return _b64(fig)


def write_dashboard(path, meta, current, predicted, evolution, regular, replaced, values_df, mode, steps):
    xlabel = f"{TIME_UNIT}s elapsed"
    fr = _dash_frames(evolution, regular, replaced)
    rows = _summary_rows(current, predicted)
    img = lambda b: f"<div><img src='data:image/png;base64,{b}'></div>"
    html = [f"<html><head><meta charset='utf-8'><title>Farm projection</title><style>{PAGE_CSS}</style></head><body>",
            _header_html("Farm Herd Dashboard", meta, mode, steps), "<div class='container'>",
            _cards_html(rows, steps), f"<div class='section'><h2>Quick Summary</h2>{_summary_table_html(rows, steps)}</div>"]
    for g in current:
        evo = fr["flag4"] if g == "status" else fr["evolution"][g]
        html.append(f"<div class='section'><h2>{_section_title(g)}</h2><div class='row'>"
                    + img(plot_compare(current[g], predicted[g], f"Current vs predicted ({TIME_UNIT} {steps})"))
                    + img(plot_series(evo, _evo_title(g), xlabel, stacked=(g == "status"))) + "</div></div>")
    html.append(f"<div class='section'><h2>Replacement-Aware Distribution</h2><p class='note'>{_REPL_NOTE}</p><div class='row'>"
                + img(plot_series(fr["regular"], "Original herd by status", xlabel, stacked=True))
                + img(plot_series(fr["replaced"], "Replacement herd by status", xlabel, stacked=True))
                + img(plot_overlay(fr["regular"], fr["replaced"], xlabel)) + "</div></div>")
    html += [_values_html(values_df), "</div></body></html>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


# ---------------- interactive (Plotly) ----------------
def write_plotly(path, meta, current, predicted, evolution, regular, replaced, values_df, mode, steps):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return False
    xlabel = f"{TIME_UNIT.capitalize()}s elapsed"
    fr = _dash_frames(evolution, regular, replaced)
    rows = _summary_rows(current, predicted)
    first = [True]

    def div(fig, title, xaxis=None, unified=False):
        fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title="Number of cows", template="plotly_white", height=400,
                          margin=dict(l=50, r=20, t=50, b=50), legend=dict(font=dict(size=10)))
        if unified:      # time-series charts: one vertical line across the chart and one box listing every series at that step (presentation only)
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor="#888")
        out = pio.to_html(fig, full_html=False, include_plotlyjs=True if first[0] else False, config={"responsive": True})    # Plotly embedded once -> works offline
        first[0] = False
        return f"<div>{out}</div>"

    def compare(cur, pred, title):
        idx = sorted(set(cur.index) | set(pred.index))
        fig = go.Figure([go.Bar(x=idx, y=[cur.get(i, 0) for i in idx], name="Current", marker_color="#95a5a6"),
                         go.Bar(x=idx, y=[pred.get(i, 0) for i in idx], name="Predicted", marker_color="#3498db")])
        fig.update_layout(barmode="group")
        return div(fig, title)

    def lines(df, title, stacked=False):
        fig = go.Figure()
        for i, c in enumerate(df.columns):
            fig.add_trace(go.Scatter(x=list(df.index), y=df[c], name=str(c), mode="lines", line=dict(color=_color(c, i)),
                                     stackgroup="one" if stacked else None,
                                     hovertemplate=f"{TIME_UNIT.capitalize()} %{{x}}<br>{c}: %{{y:.1f}}<extra></extra>"))
        return div(fig, title, xaxis=xlabel, unified=True)

    overlay = go.Figure()
    for i, c in enumerate(fr["regular"].columns):
        overlay.add_trace(go.Scatter(x=list(fr["regular"].index), y=fr["regular"][c], name=f"{c} (original)", line=dict(color=_color(c, i))))
        overlay.add_trace(go.Scatter(x=list(fr["replaced"].index), y=fr["replaced"][c], name=f"{c} (replacement)",
                                     line=dict(color=_color(c, i), dash="dash")))

    html = [f"<html><head><meta charset='utf-8'><title>Farm projection (interactive)</title><style>{PAGE_CSS}</style></head><body>",
            _header_html("Farm Herd Dashboard (Interactive)", meta, mode, steps, "<p>Hover for details. Click a legend entry to toggle it.</p>"),
            "<div class='container'>", _cards_html(rows, steps)]
    for g in current:
        evo = fr["flag4"] if g == "status" else fr["evolution"][g]
        html.append(f"<div class='section'><h2>{_section_title(g)}</h2><div class='row'>"
                    + compare(current[g], predicted[g], f"Current vs predicted ({TIME_UNIT} {steps})")
                    + lines(evo, _evo_title(g), stacked=(g == "status")) + "</div></div>")
    html.append(f"<div class='section'><h2>Replacement-Aware Distribution</h2><p class='note'>{_REPL_NOTE}</p><div class='row'>"
                + lines(fr["regular"], "Original herd by status", stacked=True)
                + lines(fr["replaced"], "Replacement herd by status", stacked=True)
                + div(overlay, "Original vs replacement herd (overlaid)", xaxis=xlabel, unified=True) + "</div></div>")
    fit = ("<script>(function(){function fit(){document.querySelectorAll('.plotly-graph-div').forEach(function(d){Plotly.Plots.resize(d);});}"
           "window.addEventListener('load',function(){fit();setTimeout(fit,300);});window.addEventListener('resize',fit);})();</script>")   # charts are drawn before the 2-column layout settles
    html += [_values_html(values_df), "</div>", fit, "</body></html>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return True


def write_farm_outputs(result, out_dir, dashboards=True):
    """Write the result files of project_herd into out_dir (created). Returns the list of files written."""
    if plt is None and dashboards:
        raise ImportError("matplotlib is required for the dashboards (pip install cowmrp[plot]), or pass dashboards=False")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def csv(df, name, **kw):
        path = os.path.join(out_dir, name)
        df.to_csv(path, **kw)
        written.append(path)

    r = result
    if r["method_comparison"] is not None:
        csv(r["method_comparison"], "method_comparison.csv", index=False)
    if r["trajectory"] is not None:
        csv(r["trajectory"], "trajectory.csv", index=False)
    if r["monte_carlo"] is not None:
        csv(r["monte_carlo"], "monte_carlo_summary.csv", index=False)
    csv(r["distribution_summary"], "distribution_summary.csv", index=False)
    regular, replaced = r["regular"], r["replaced"]
    csv(pd.DataFrame({f"step_{s}": pd.Series(regular[s]) for s in range(len(regular))}), "replacement_tracking_regular.csv")
    csv(pd.DataFrame({f"step_{s}": pd.Series(replaced[s]) for s in range(len(replaced))}), "replacement_tracking_replaced.csv")
    if r["values"] is not None:
        csv(r["values"], "cow_values.csv", index=False)
    if dashboards:
        args = (r["meta"], r["current"], r["predicted"], r["evolution"], regular, replaced, r["values"], r["mode"], r["steps"])
        path = os.path.join(out_dir, "dashboard.html")
        write_dashboard(path, *args)
        written.append(path)
        path = os.path.join(out_dir, "dashboard_plotly.html")
        if write_plotly(path, *args):
            written.append(path)
    return written

"""
runners/_shared/scale_report.py
Pages for the scale benchmark (runners/_shared/scale_benchmark.py). Drawing only: it reads timing.csv, accuracy.csv, skipped.csv and
manifest.json written by the benchmark and decides nothing about the model. Offline (plotly.min.js written beside the page).

    write_subject_page(subject, module, error_limit)  -> runners/<module>/outputs/benchmark_scale/index.html
    write_summary(subjects, error_limit)              -> runners/_shared/outputs/benchmark_scale/index.html
The "cheapest accurate method" rule is the one stated on the pages: among the methods that ran and whose largest deciding error_pct
is <= the error limit, the one with the smallest wall time (exact methods have error 0 and are always eligible).
"""
import html
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners._shared import common
from runners._shared.review import COLORS, EXTRA_CSS

METHOD_COLOR = {"sim": COLORS[1], "sampled": COLORS[1], "mrp_distinct": COLORS[0], "mrp_all": COLORS[2], "exact": COLORS[2], "expected": COLORS[2]}
EXACT = {"mrp_all", "exact", "expected"}
CLIP = 1e-10
TITLES = {"cow": "Cow", "victor": "Victor", "youngstock": "Youngstock", "herd": "Herd (cow + youngstock)"}
EXTRA = (".tbl{border-collapse:collapse;font-size:.88rem;margin:.5rem 0 1.5rem 0}.tbl td,.tbl th{border:1px solid #d5dbdf;padding:4px 9px;text-align:right}"
         ".tbl th{background:#eef2f5}.tbl td.l,.tbl th.l{text-align:left}.fig{background:white;border-radius:10px;padding:8px;margin:0 0 1.2rem 0;box-shadow:0 2px 8px rgba(0,0,0,.07)}"
         ".read{color:#555;font-size:.9rem;margin:.2rem 0 .6rem 0}")


def _read(out, name):
    p = os.path.join(out, name)
    if os.path.exists(p) and os.path.getsize(p) > 2:
        return pd.read_csv(p)
    return pd.DataFrame()


def load(module):
    out = common.module_outputs(module, "benchmark_scale")
    man = {}
    if os.path.exists(os.path.join(out, "manifest.json")):
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
    return out, _read(out, "timing.csv"), _read(out, "accuracy.csv"), _read(out, "skipped.csv"), man


def decision_table(acc):
    """Largest deciding error_pct per (time_unit, n_entities, method). Exact methods are 0 by construction."""
    if acc.empty:
        return pd.DataFrame(columns=["time_unit", "n_entities", "method", "error_pct"])
    d = acc[acc["decides"].astype(bool)]
    return d.groupby(["time_unit", "n_entities", "method"], as_index=False)["error_pct"].max()


def cheapest(timing, acc, limit):
    """Rows (time_unit, n_entities) -> cheapest eligible method and its wall time (None if nothing eligible)."""
    if timing.empty:
        return pd.DataFrame()
    t = timing[(timing["n_entities"] > 0) & timing["method"].notna() & (timing["method"] != "")]
    dec = decision_table(acc)
    m = t.merge(dec, on=["time_unit", "n_entities", "method"], how="left")
    m["error_pct"] = np.where(m["method"].isin(EXACT), 0.0, m["error_pct"])
    ok = m[m["error_pct"] <= limit]
    rows = []
    for (tu, n), g in t.groupby(["time_unit", "n_entities"]):
        e = ok[(ok["time_unit"] == tu) & (ok["n_entities"] == n)]
        if e.empty:
            rows.append(dict(time_unit=tu, n_entities=n, method=None, wall_s=np.nan))
        else:
            b = e.loc[e["wall_s"].idxmin()]
            rows.append(dict(time_unit=tu, n_entities=n, method=b["method"], wall_s=float(b["wall_s"])))
    return pd.DataFrame(rows)


def _fig_html(fig, title, read):
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=60, r=20, t=40, b=50), title=title, legend=dict(orientation="h", y=-0.2))
    import plotly.io as pio
    return f"<div class='fig'>{pio.to_html(fig, full_html=False, include_plotlyjs=False, config={'responsive': True})}<div class='read'>What to read: {html.escape(read)}</div></div>"


def _lines(df, ycol, ytitle, units, title, read, logy=True, ref=None, limit=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(units), subplot_titles=[u for u in units], shared_yaxes=True)
    for c, u in enumerate(units, 1):
        for m, g in df[df["time_unit"] == u].groupby("method"):
            g = g.sort_values("n_entities")
            fig.add_trace(go.Scatter(x=g["n_entities"], y=g[ycol], mode="lines+markers", name=m, legendgroup=m, showlegend=(c == 1),
                                     line=dict(color=METHOD_COLOR.get(m, "#555"))), row=1, col=c)
        if limit is not None:
            fig.add_hline(y=limit, line_dash="dash", line_color="#c0392b", row=1, col=c, annotation_text=f"limit {limit:g} %" if c == 1 else " ")
        fig.update_xaxes(type="log", title_text="entities (N)", tickvals=sorted(set(df["n_entities"])), ticktext=[f"{int(n):,}" for n in sorted(set(df["n_entities"]))], row=1, col=c)
        fig.update_yaxes(type="log" if logy else "linear", row=1, col=c)
    fig.update_yaxes(title_text=ytitle, col=1)
    return _fig_html(fig, title, read)


def _table(df, cls_left=()):
    h = "<table class='tbl'><tr>" + "".join(f"<th class='{'l' if c in cls_left else ''}'>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td class='{'l' if c in cls_left else ''}'>{html.escape(_f(r[c]))}</td>" for c in df.columns) + "</tr>"
    return h + "</table>"


def _f(x):
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else (f"{x:,.3g}" if abs(x) < 1e4 else f"{x:,.0f}")
    return str(x)


def _page(title, body):
    return (f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{common.PAGE_CSS}{EXTRA_CSS}{EXTRA}</style>"
            f"<script src='plotly.min.js'></script></head><body>{body}</body></html>")


def _write_plotly(out):
    from plotly.offline import get_plotlyjs
    with open(os.path.join(out, "plotly.min.js"), "w", encoding="utf-8") as f:
        f.write(get_plotlyjs())


def write_subject_page(subject, module, limit):
    out, timing, acc, skipped, man = load(module)
    _write_plotly(out)
    name = TITLES.get(subject, subject)
    if timing.empty:
        body = f"<h1>Scale benchmark - {name}</h1><p>No timing results yet. Run <code>python runners/{module}/run.py --scale</code>.</p>"
        path = os.path.join(out, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_page(f"Scale benchmark {name}", body))
        return path
    units = [u for u in ["month", "week", "day"] if u in set(timing["time_unit"])]
    run = timing[(timing["n_entities"] > 0) & timing["method"].notna() & (timing["method"] != "")]
    build = timing[timing["phase"] == "build_model"]
    mach = man.get("machine", {})
    import importlib
    try:
        NOTES = importlib.import_module(f"runners.{module}.scale_inputs").NOTES
    except Exception:
        NOTES = ""
    head = (f"<h1>Scale benchmark - {name}</h1><div class='notes'>{html.escape(NOTES)}<br>Horizon {man.get('horizon_years', '?')} years | "
            f"error limit {limit:g} % | cap {man.get('cap_minutes', '?')} min per cell | seed {man.get('seed', '?')} | run {man.get('created_utc', '?')}<br>"
            f"Machine: {html.escape(str(mach.get('processor', '')))}, {mach.get('cpu_cores_physical', '?')} cores / {mach.get('cpu_threads', '?')} threads, "
            f"{mach.get('ram_total_gb', '?')} GB RAM, Python {mach.get('python', '?')}, numpy {mach.get('numpy', '?')}, no GPU used.</div>"
            "<div class='notes'><b>Accuracy rule.</b> error % = 100 x |estimate - exact| / max(|exact|, floor), floor = 1 % of the quantity's natural scale "
            "(portfolio of cows: sum of |exact values|; a cohort share: 1; animal count in the herd: cows at the start). Deciding metrics: portfolio value (value subjects), "
            "every outcome share (youngstock), bias and spread of every quantity (herd). Spread = half of the 5-95 % band of single sampled runs. "
            f"The cheapest accurate method is the one with the smallest wall time among those whose largest deciding error is at most {limit:g} %.</div>")
    figs = []
    figs.append(_lines(run, "wall_s", "seconds", units, "Run time against N",
                       "Lines that rise with N pay per animal; a flat line (exact methods) is a one-off cost. Both axes logarithmic.", ))
    figs.append(_lines(run, "per_entity_ms", "ms per entity", units, "Milliseconds per entity",
                       "A falling line means the method gets cheaper per animal as the population grows."))
    figs.append(_lines(run, "peak_ram_mb", "MB", units, "Peak memory", "Resident memory of the process during the method (MB).", logy=True))
    dec = decision_table(acc)
    if not dec.empty:
        d2 = dec.copy()
        d2["error_plot"] = d2["error_pct"].clip(lower=CLIP)
        figs.append(_lines(d2, "error_plot", "largest deciding error, %", units, "Accuracy against N",
                           f"Lower is better. Dashed line = error limit ({limit:g} %). Exact methods sit at the floor of the axis (error 0). Sampled errors fall roughly like 1/sqrt(N); a flat sampled line means a model gap that more samples do not remove.",
                           limit=limit))
    # speed-up of the exact method over sampling
    ex_m = [m for m in EXACT if m in set(run["method"])]
    sm = [m for m in ("sim", "sampled") if m in set(run["method"])]
    if ex_m and sm:
        e = run[run["method"] == ex_m[0]][["time_unit", "n_entities", "wall_s"]].rename(columns={"wall_s": "exact_s"})
        s = run[run["method"] == sm[0]][["time_unit", "n_entities", "wall_s"]].rename(columns={"wall_s": "sample_s"})
        j = e.merge(s, on=["time_unit", "n_entities"])
        if not j.empty:
            j["speedup"] = j["sample_s"] / j["exact_s"]
            j["method"] = f"{sm[0]} / {ex_m[0]}"
            figs.append(_lines(j, "speedup", "times faster", units, "Speed-up of the exact method over sampling",
                               "Above 1 the exact method is faster. (Herd: time of ONE sampled run against one expected run.)"))
    # model build
    if not build.empty:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=build["time_unit"], y=build["wall_s"], text=[f"{int(s):,} states<br>{int(z):,} nnz" for s, z in zip(build["n_states"], build["nnz"])],
                               marker_color=COLORS[0]))
        fig.update_yaxes(title_text="seconds")
        figs.append(_fig_html(fig, "One-time model build", "Time to build the model once per resolution; bar labels give the number of states and non-zero transitions."))
    # cheapest table and skipped
    ch = cheapest(timing, acc, limit)
    body = head + "".join(figs)
    if not ch.empty:
        t = ch.pivot(index="n_entities", columns="time_unit", values="method").fillna("none within limit")
        t.insert(0, "N", t.index)
        body += f"<h2>Cheapest accurate method (limit {limit:g} %)</h2><div class='read'>Smallest wall time among the methods within the limit.</div>" + _table(t, cls_left=tuple(t.columns))
    if len(skipped):
        body += "<h2>Skipped or failed runs</h2><div class='read'>Nothing is dropped silently: every run not made is listed with the reason.</div>" + _table(skipped.fillna(""), cls_left=("reason",))
    else:
        body += "<h2>Skipped or failed runs</h2><p>None.</p>"
    cols = [c for c in ("time_unit", "n_entities", "method", "wall_s", "cpu_pct", "peak_ram_mb", "per_entity_ms") if c in run.columns]
    body += "<h2>Timing table</h2>" + _table(run[cols].sort_values(["time_unit", "n_entities", "method"]), cls_left=("time_unit", "method"))
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"Scale benchmark {name}", body))
    return path


def write_summary(subjects, limit):
    out = common.module_outputs("_shared", "benchmark_scale")
    _write_plotly(out)
    body = f"<h1>Scale benchmark - all subjects</h1><div class='notes'>Cheapest method whose largest deciding error is at most {limit:g} % (smallest wall time; exact methods always qualify). Grey = skipped (reason in the subject page). Colour: green = exact method, orange = sampling, blue = exact from distinct start states.</div>"
    cell_color = {"sim": "#f6d9b8", "sampled": "#f6d9b8", "mrp_distinct": "#c9dff3", "mrp_all": "#cdebd6", "exact": "#cdebd6", "expected": "#cdebd6"}
    for s in subjects:
        module = s if ":" not in s else "_shared"
        _, timing, acc, skipped, man = load(module)
        body += f"<h2><a href='../../../{module}/outputs/benchmark_scale/index.html'>{TITLES.get(s, s)}</a></h2>"
        if timing.empty:
            body += "<p>No results.</p>"; continue
        ch = cheapest(timing, acc, limit)
        sk = skipped.dropna(subset=["n_entities"]) if len(skipped) else skipped
        h = "<table class='tbl'><tr><th>N</th>" + "".join(f"<th>{u}</th>" for u in sorted(set(ch["time_unit"]), key=["month", "week", "day"].index)) + "</tr>"
        for n in sorted(set(ch["n_entities"])):
            h += f"<tr><td>{int(n):,}</td>"
            for u in sorted(set(ch["time_unit"]), key=["month", "week", "day"].index):
                r = ch[(ch["time_unit"] == u) & (ch["n_entities"] == n)]
                if r.empty or r.iloc[0]["method"] is None or pd.isna(r.iloc[0]["method"]):
                    h += "<td style='background:#e3e6e8'>none within limit</td>"
                else:
                    m, w = r.iloc[0]["method"], r.iloc[0]["wall_s"]
                    h += f"<td style='background:{cell_color.get(m, '#fff')}'>{html.escape(m)}<br><small>{w:,.2f} s</small></td>"
            h += "</tr>"
        body += h + "</table>"
        if len(skipped):
            body += f"<div class='read'>{len(skipped)} run(s) skipped or failed - see the subject page.</div>"
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page("Scale benchmark summary", body))
    return path

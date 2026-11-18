"""
runners/_shared/review.py
=========================
Review pages for a domain expert: ONE overview page that lists every probability and reward of a domain (a small picture, the
range and the unit of each), and one detail page per item (large interactive figure, a "what to check" line). The expert can
judge from figures whether the probabilities and rewards make sense without opening any csv.

    python runners/_shared/review.py <domain> [--time-unit month|week|day]       (or: python runners/<module>/run.py --review)

OUTPUT  runners/<domain>/outputs/review/
    index.html            the overview (open this one)
    items/<id>.html       one detail page per item
    plotly.min.js         Plotly, written once so the pages work offline

The domain declares WHAT to show in domains/<domain>/review.py:
    NOTES                    optional list of plain sentences shown at the top of the overview (what the domain does not have, ...)
    build_items(model)       list of items; `model` is the engine_run.Model(states, probs, rewards, P) of the domain
An item is a dict:
    id, group ("Probabilities" | "Rewards" | "Values" | "Population" | ...), title, unit, check (the "what to check" sentence),
    panels: list of {title, kind, df, xlabel, ylabel, overlay=None}
      kind "lines"    one line per column of df (index = x); overlay = df of the same shape drawn dashed (a source table)
      kind "area"     stacked area, one band per column
      kind "bars"     grouped bars: x = index, one bar per column
      kind "heatmap"  rows = index (y), columns = x, cell colour = value
No model logic here: every value comes from the domain's own arrays. The builder only draws.
"""
import html
import importlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners._shared import common

EXTRA_CSS = (".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}"
             ".thumb{display:block;background:white;border-radius:10px;padding:10px 12px;box-shadow:0 2px 8px rgba(0,0,0,.07);text-decoration:none;color:#2c3e50;border:1px solid transparent}"
             ".thumb:hover{border-color:#3498db}.thumb h3{margin:0 0 4px 0;font-size:.95rem}.thumb svg{width:100%;height:64px;background:#f4f6f8;border-radius:6px}"
             ".thumb .rng{font-size:.82rem;color:#555;margin-top:4px}.check{background:#fff8e1;border-left:4px solid #f1c40f;padding:10px 14px;margin:0 0 1.2rem 0;border-radius:4px}"
             ".nav a{margin-right:14px}.notes{background:#eef6fc;border-left:4px solid #3498db;padding:10px 14px;margin-bottom:1.5rem;border-radius:4px}"
             ".panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}.panels>div{min-width:0;overflow:hidden}")
COLORS = ["#3498db", "#e67e22", "#27ae60", "#c0392b", "#8e44ad", "#16a085", "#7f8c8d", "#d35400", "#2c3e50", "#f1c40f"]


def _vals(item):
    arrs = [np.asarray(p["df"].values, dtype=float).ravel() for p in item["panels"]]
    v = np.concatenate(arrs) if arrs else np.array([])
    return v[np.isfinite(v)]


def _fmt(x):
    if x == 0:
        return "0"
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def _range_text(item):
    v = _vals(item)
    return f"{_fmt(v.min())} to {_fmt(v.max())} {html.escape(item['unit'])}" if v.size else "no values"


def _thumb_panel(item):
    """The panel with the widest spread of values (a panel that is all zero, e.g. a reward only pregnant cows get, says nothing)."""
    def spread(p):
        v = np.asarray(p["df"].values, dtype=float)
        v = v[np.isfinite(v)]
        return float(v.max() - v.min()) if v.size else 0.0
    return max(item["panels"], key=spread)


def _sparkline(panel):
    """Tiny inline SVG of the item's first panel (no axes): lines, bands, bars or a heat grid."""
    df, kind = panel["df"], panel["kind"]
    W, H = 200, 64
    v = np.asarray(df.values, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return f"<svg viewBox='0 0 {W} {H}'></svg>"
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = (hi - lo) or 1.0
    out = [f"<svg viewBox='0 0 {W} {H}' preserveAspectRatio='none'>"]
    if kind == "heatmap":
        r, c = v.shape
        cw, ch = W / c, H / r
        for i in range(r):
            for j in range(c):
                t = 0 if not np.isfinite(v[i, j]) else (v[i, j] - lo) / span
                out.append(f"<rect x='{j*cw:.1f}' y='{(r-1-i)*ch:.1f}' width='{cw+.5:.1f}' height='{ch+.5:.1f}' fill='rgb({int(240-200*t)},{int(240-120*t)},{int(250-60*t)})'/>")
    elif kind == "bars":
        flat = v.ravel()
        bw = W / max(flat.size, 1)
        base = 0 if lo >= 0 else lo
        top = hi if hi > 0 else 0
        sp = (top - base) or 1.0
        ncol = v.shape[1] if v.ndim > 1 else 1
        for k, y in enumerate(flat):
            if not np.isfinite(y):
                continue
            h = max((y - base) / sp * (H - 4), 1)
            out.append(f"<rect x='{k*bw+1:.1f}' y='{H-2-h:.1f}' width='{max(bw-2,1):.1f}' height='{h:.1f}' fill='{COLORS[(k % ncol) % len(COLORS)]}'/>")
    else:
        x = np.arange(len(df))
        xs = (x - x.min()) / max(x.max() - x.min(), 1) * (W - 4) + 2
        cols = list(df.columns)[:8]
        stacked = kind == "area"
        cum = np.zeros(len(df))
        if stacked:
            lo, hi = 0.0, float(np.nanmax(np.nansum(np.nan_to_num(v), axis=1))) or 1.0
            span = (hi - lo) or 1.0
        for k, c in enumerate(cols):
            y = np.nan_to_num(df[c].values.astype(float))
            if stacked:
                cum = cum + y
                y = cum
            ys = H - 3 - (y - lo) / span * (H - 6)
            pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
            out.append(f"<polyline fill='none' stroke='{COLORS[k % len(COLORS)]}' stroke-width='1.6' points='{pts}'/>")
    out.append("</svg>")
    return "".join(out)


def _figure(panel, unit):
    import plotly.graph_objects as go
    import plotly.io as pio
    df, kind = panel["df"], panel["kind"]
    fig = go.Figure()
    if kind == "heatmap":
        fig.add_trace(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale="Viridis", colorbar=dict(title=unit)))
    elif kind == "bars":
        for k, c in enumerate(df.columns):
            fig.add_trace(go.Bar(x=[str(i) for i in df.index], y=df[c], name=str(c), marker_color=COLORS[k % len(COLORS)]))
        fig.update_layout(barmode="group")
    else:
        for k, c in enumerate(df.columns):
            kw = {"stackgroup": "one"} if kind == "area" else {}
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c), line=dict(color=COLORS[k % len(COLORS)]), **kw))
            ov = panel.get("overlay")
            if ov is not None and c in ov.columns:
                fig.add_trace(go.Scatter(x=ov.index, y=ov[c], mode="lines", name=f"{c} (source)", line=dict(color=COLORS[k % len(COLORS)], dash="dash"), opacity=.7))
    fig.update_layout(title=panel.get("title", ""), xaxis_title=panel.get("xlabel", ""), yaxis_title=panel.get("ylabel", unit),
                      template="plotly_white", height=420, margin=dict(l=55, r=20, t=50, b=45))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def _page(title, body, head=""):
    return (f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{common.PAGE_CSS}{EXTRA_CSS}</style>{head}</head>"
            f"<body>{body}</body></html>")


def _groups_html(items, collapsible=False):
    """The grid of tiles, one section per group, in order of first appearance. collapsible=True (the central page of views/) wraps each
    group in an open <details>; the review's own index.html uses plain sections."""
    groups = []
    for it in items:
        if it["group"] not in groups:
            groups.append(it["group"])
    out = []
    for g in groups:
        cards = "".join(f"<a class='thumb' href='items/{it['id']}.html'><h3>{html.escape(it['title'])}</h3>{_sparkline(_thumb_panel(it))}"
                        f"<div class='rng'>{_range_text(it)}</div></a>" for it in items if it["group"] == g)
        grid = f"<div class='grid'>{cards}</div>"
        out.append(f"<details class='section' open><summary><h2>{html.escape(g)}</h2></summary>{grid}</details>" if collapsible
                   else f"<div class='section'><h2>{html.escape(g)}</h2>{grid}</div>")
    return "".join(out)


def _intro_html(domain, model, params, items, rev):
    """(the two header lines, the notes box): what the overview says about the domain."""
    notes = getattr(rev, "NOTES", [])
    lines = (f"<p>{len(model.states):,} states | {html.escape(str(params.TIME_UNIT))} steps ({params.UNIT_DAYS} d) | {len(items)} items</p>"
             "<p>Every picture is drawn from the model's own probabilities and rewards. Click one for the full figure.</p>")
    return lines, (f"<div class='notes'>{'<br>'.join(html.escape(n) for n in notes)}</div>" if notes else "")


def build(domain, model=None, out_dir=None, home_href=None):
    """Write the review pages of `domain` (into `out_dir`, default runners/<domain>/outputs/review); returns the path of index.html.
    home_href (given by the views/ build, runners/_shared/views.py): adds an "all reports" link to every detail page and writes _overview.html, the grid of
    tiles as a fragment that the central page embeds. Without it the output is exactly what --review always wrote."""
    from runners._shared import engine_run
    if importlib.util.find_spec(f"cowmrp.domains.{domain}.review") is None:
        raise SystemExit(f"domain {domain!r} has no domains/{domain}/review.py (it lists what to show; see runners/_shared/review.py)")
    rev = importlib.import_module(f"cowmrp.domains.{domain}.review")
    hooks = importlib.import_module(f"cowmrp.domains.{domain}.notebook_hooks") if importlib.util.find_spec(f"cowmrp.domains.{domain}.notebook_hooks") else None
    if model is None:
        model = engine_run.build(domain, strict=getattr(hooks, "STRICT", True))
    params = importlib.import_module(f"cowmrp.domains.{domain}.parameters")
    items = rev.build_items(model)
    out = os.path.abspath(out_dir) if out_dir else common.module_outputs(domain, "review")
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    from plotly.offline import get_plotlyjs
    with open(os.path.join(out, "plotly.min.js"), "w", encoding="utf-8") as f:
        f.write(get_plotlyjs())
    ids = [it["id"] for it in items]
    assert len(set(ids)) == len(ids), "duplicate item ids"

    for k, it in enumerate(items):
        nav = ([f"<a href='{home_href}'>&larr; all reports</a>"] if home_href else []) + ["<a href='../index.html'>&larr; all items</a>"]
        if k > 0:
            nav.append(f"<a href='{ids[k-1]}.html'>&larr; {html.escape(items[k-1]['title'])}</a>")
        if k < len(items) - 1:
            nav.append(f"<a href='{ids[k+1]}.html'>{html.escape(items[k+1]['title'])} &rarr;</a>")
        panels = "".join(f"<div class='section'>{_figure(p, it['unit'])}</div>" for p in it["panels"])
        body = (f"<div class='header'><h1>{html.escape(it['title'])}</h1><p>{html.escape(domain)} | {html.escape(it['group'])} | range {_range_text(it)}</p></div>"
                f"<div class='container'><p class='nav'>{' '.join(nav)}</p>"
                f"<div class='check'><b>What to check.</b> {html.escape(it['check'])}</div><div class='panels'>{panels}</div></div>")
        body += "<script>window.addEventListener('load',function(){setTimeout(function(){window.dispatchEvent(new Event('resize'))},60)})</script>"   # Plotly refits to the final grid width
        with open(os.path.join(out, "items", f"{it['id']}.html"), "w", encoding="utf-8") as f:
            f.write(_page(it["title"], body, head="<script src='../plotly.min.js'></script>"))

    lines, notes_html = _intro_html(domain, model, params, items, rev)
    body = (f"<div class='header'><h1>{html.escape(domain.capitalize())} model review</h1>{lines}</div>"
            f"<div class='container'>{notes_html}{_groups_html(items)}</div>")
    if home_href:        # the views/ build only: the fragment its central page embeds (the default --review output is unchanged)
        with open(os.path.join(out, "_overview.html"), "w", encoding="utf-8") as f:
            f.write(f"<div class='ovsum'>{lines}</div>{notes_html}{_groups_html(items, collapsible=True)}")
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"{domain} model review", body))
    return path




if __name__ == "__main__":
    args = sys.argv[1:]
    common.set_time_unit_from_argv()
    names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--time-unit")]
    if not names:
        sys.exit(__doc__)
    print(f"review: {build(names[0])}")

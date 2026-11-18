"""
runners/_shared/common.py
=========================
The few things every module runner shares. No model logic and no domain import lives here (a domain must be importable
only AFTER the time unit is set, so `set_time_unit_from_argv()` has to run before any `domains.*` import).

    project_root()                   folder that holds mrp/
    set_time_unit_from_argv()        --time-unit month|week|day  ->  COW_TIME_UNIT (read once by cowmrp.domains.cow.parameters)
    module_outputs(module, kind, *parts)   runners/<module>/outputs/<kind>/<parts...>   (created; kind must be in KINDS)
    write_manifest(out_dir, module, **fields)   outputs/.../manifest.json: what was produced, when, with which settings
    area_chart / section / write_page   the dashboard page every module writes (one layout: header, section cards, Plotly charts)
"""
import json
import os
import sys
from datetime import datetime, timezone


def project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while not (os.path.isfile(os.path.join(d, "pyproject.toml")) and os.path.isdir(os.path.join(d, "runners"))):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root (the folder holding pyproject.toml and runners/) not found")
        d = parent
    return d


def set_time_unit_from_argv(argv=None):
    argv = sys.argv if argv is None else argv
    if "--time-unit" in argv:
        os.environ["COW_TIME_UNIT"] = argv[argv.index("--time-unit") + 1]


# The kinds of output a module may write: runners/<module>/outputs/<kind>/[<time unit>/]...
# Any other first folder is refused, so a script can never leave a stray folder behind.
KINDS = {
    "model":          "states, transition matrix, reward vector and values (per time unit)",
    "validation":     "figures and the validation HTML pages",
    "dashboard":      "the module's dashboard page(s): html with the charts of its model/farm results",
    "audit":          "audit and check results (json, csv, snapshots)",
    "lookup_tables":  "exported probability and reward lookup tables",
    "benchmark":      "benchmark runs (summary, comparison plots, per-model results)",
    "baseline":       "saved reference results the parity check compares against",
    "farm":           "farm files and farm simulations",
    "projection":     "projection of chosen cows over years",
    "notebook":       "files written by the runner notebook",
    "review":         "review pages for a domain expert: overview of every probability/reward + one figure each (index.html)",
    "population":     "whole-farm population projections (herd module)",
    "profile_build":  "build-phase timing",
    "benchmark_scale": "cost of valuing N entities",
}


def module_outputs(module, kind, *parts):
    """runners/<module>/outputs/<kind>/<parts...>. A last part with a file extension is a file name: its folder is created
    and the file path returned; otherwise the folder itself is created and returned."""
    if kind not in KINDS:
        raise ValueError(f"unknown output kind {kind!r}; choose one of {sorted(KINDS)} (or add it to KINDS in runners/_shared/common.py)")
    mod_dir = os.path.join(project_root(), "runners", module)
    if not os.path.isdir(mod_dir):
        raise ValueError(f"no runner module {module!r}; create it with: python runners/_shared/new_domain.py {module}")
    path = os.path.join(mod_dir, "outputs", kind, *parts)
    is_file = bool(parts) and os.path.splitext(parts[-1])[1] != ""
    os.makedirs(os.path.dirname(path) if is_file else path, exist_ok=True)
    return path


def write_manifest(out_dir, module, **fields):
    """Write manifest.json next to a module's outputs. `fields` are plain JSON values (time unit, state count, files, checks)."""
    manifest = {"module": module, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)
    return path


# ---------------------------------------------------------------------------------------------
# Dashboard page (one layout for every module). Plotly is embedded once, so the page works offline.
# ---------------------------------------------------------------------------------------------
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
_FIRST_CHART = [True]


def area_chart(df, title, xlabel, ylabel, stacked=True, colors=None, height=420):
    """Plotly chart, one trace per column of `df` (index = x). Stacked area by default, plain lines with stacked=False."""
    import plotly.graph_objects as go
    import plotly.io as pio
    fig = go.Figure()
    for c in df.columns:
        kw = {"stackgroup": "one"} if stacked else {}
        if colors and c in colors:
            kw["line"] = {"color": colors[c]}
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c), **kw))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white", height=height, margin=dict(l=50, r=20, t=50, b=40))
    out = pio.to_html(fig, full_html=False, include_plotlyjs=_FIRST_CHART[0], config={"responsive": True})
    _FIRST_CHART[0] = False
    return out


def section(title, note, *blocks):
    """A white card with a heading, a one-line note and one block per grid column (a chart's html or a table's html)."""
    return (f"<div class='section'><h2>{title}</h2><p class='note'>{note}</p><div class='row'>"
            + "".join(f"<div>{b}</div>" for b in blocks) + "</div></div>")


def write_page(path, title, header_lines, sections, tab_title=None):
    """Header (title + lines), then the section cards. Returns path."""
    head = f"<div class='header'><h1>{title}</h1>" + "".join(f"<p>{l}</p>" for l in header_lines) + "</div>"
    html = (f"<html><head><meta charset='utf-8'><title>{tab_title or title}</title><style>{PAGE_CSS}</style></head><body>{head}"
            f"<div class='container'>" + "".join(sections) + "</div></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    _FIRST_CHART[0] = True          # the next page embeds Plotly again
    return path

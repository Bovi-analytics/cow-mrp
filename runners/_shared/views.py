"""
runners/_shared/views.py
========================
The local, Git-ignored report folder `views/` and its central page `views/index.html`.

    python runners/run_all.py --views [--only cow,youngstock] [--steps 24] [--n-cows 10000] [--skip validation,farm,...]
    python runners/<cow|youngstock>/run.py --views
    python runners/_shared/views.py --index-only          rebuild views/index.html from what is on disk (no report is regenerated)

Every report is built from the current package (build_model, the existing report builders); no model calculation is written here. The inspection pages (states, reward vectors, transition matrices) read the model's own arrays.

THE REGISTRY  (REPORTS below) is the one list of reports. The central page is generated from it: one row per report with the domain, the
time resolution it covers, the type, a link when the files exist, and otherwise a plain status -- "not generated yet", "FAILED (reason)"
or "not available for this domain (reason)". A report is never silently left out. TO ADD A REPORT: add one `Report(...)` entry (and a
builder function if it needs one); it then appears on the central page with no other change.

LAYOUT
    views/index.html            the central page
    views/_assets/plotly.min.js ONE shared copy of Plotly; every page that needs it points here (see localize_plotly)
    views/_status.json          result of the last attempt per report (ok / failed + reason), used by the index
    views/<domain>/data/month/  states.csv, reward_vector_<domain>.csv, transition_matrix_<domain>_long.csv, ...npz (the data behind the pages)
    views/<domain>/month/       states.html, reward_vectors.html, transition_matrices.html, review/, farm/ (cow), dashboard/
    views/cow/validation/       P00-P14 + validation_all.html + index.html (month, week AND day: several figures compare resolutions)
    views/cow/projection/       the individual-cow projection (the resolution(s) are set by SCENARIOS in runners/cow/projection.py)

Reports are built at MONTH resolution (the validation set also builds week and day, as its figures compare them). One process = one time
resolution (cowmrp reads COW_TIME_UNIT once), so the heavy builders run in their own processes.
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("COW_TIME_UNIT", "month")       # the inspection pages are month-resolution; builders that need another set it themselves
VIEWS = os.path.join(ROOT, "views")
ASSETS = os.path.join(VIEWS, "_assets", "plotly.min.js")
STATUS = os.path.join(VIEWS, "_status.json")
DOMAINS = ("cow", "youngstock")


# =============================================================================================================================
# The registry
# =============================================================================================================================
@dataclass
class Report:
    key: str                        # unique, also the name used by --skip
    domain: str
    rtype: str                      # grouping on the central page
    title: str
    resolution: str                 # what the report covers, shown on the central page
    entry: str = ""                 # main file, relative to views/ ("" = not available for this domain)
    pages: str = ""                 # glob (relative to views/) of the individual pages listed under the report
    data: list = field(default_factory=list)       # underlying data files: paths or globs, relative to views/
    unavailable: str = ""           # why this report does not exist for this domain (shown instead of a link)
    note: str = ""


REPORTS = [
    # ---------------- cow ----------------
    Report("cow-validation", "cow", "Validation", "Validation figures P00-P14 (index)", "month + week + day", "cow/validation/index.html",
           "cow/validation/P*.html", note="figures compare the three resolutions, so all three are built"),
    Report("cow-validation-all", "cow", "Validation", "validation_all.html (every figure on one page)", "month + week + day", "cow/validation/validation_all.html"),
    Report("cow-states", "cow", "Model inspection", "States", "month", "cow/month/states.html", data=["cow/data/month/states.csv"]),
    Report("cow-rewards", "cow", "Model inspection", "Reward vectors (every component, every state)", "month", "cow/month/reward_vectors.html", data=["cow/data/month/reward_vector_cow.csv"]),
    Report("cow-transitions", "cow", "Model inspection", "Transition matrix (heatmap + every row)", "month", "cow/month/transition_matrices.html", data=["cow/data/month/transition_matrix_cow*"]),
    Report("cow-farm", "cow", "Farm herd", "Farm Herd Dashboard (Interactive)", "month", "cow/month/farm/dashboard_plotly.html", data=["cow/month/farm/*.csv", "cow/month/farm/*.json"]),
    Report("cow-projection", "cow", "Individual cow", "Individual-cow projection", "as set in SCENARIOS (runners/cow/projection.py)", "cow/projection/index.html",
           "cow/projection/*.html", data=["cow/projection/*.csv"]),
    Report("cow-review", "cow", "Review", "Review pages (the grid is also on this page, above)", "month", "cow/month/review/index.html", "cow/month/review/items/*.html"),
    # ---------------- youngstock ----------------
    Report("youngstock-validation", "youngstock", "Validation", "Validation figures", "-", unavailable="no validation figures exist for youngstock"),
    Report("youngstock-states", "youngstock", "Model inspection", "States", "month", "youngstock/month/states.html", data=["youngstock/data/month/states.csv"]),
    Report("youngstock-rewards", "youngstock", "Model inspection", "Reward vectors", "month", "youngstock/month/reward_vectors.html",
           unavailable="youngstock defines no reward vector"),
    Report("youngstock-transitions", "youngstock", "Model inspection", "Transition matrix (heatmap + every row)", "month", "youngstock/month/transition_matrices.html", data=["youngstock/data/month/transition_matrix_youngstock*"]),
    Report("youngstock-farm", "youngstock", "Farm herd", "Farm herd dashboard", "-", unavailable="the farm herd dashboard is a cow-domain report"),
    Report("youngstock-dashboard", "youngstock", "Dashboard", "Youngstock cohort projection (interactive)", "month", "youngstock/month/dashboard/index.html"),
    Report("youngstock-review", "youngstock", "Review", "Review pages (the grid is also on this page, above)", "month", "youngstock/month/review/index.html", "youngstock/month/review/items/*.html"),
]
BY_KEY = {r.key: r for r in REPORTS}


# =============================================================================================================================
# Status file
# =============================================================================================================================
def _load_status():
    try:
        with open(STATUS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _record(key, ok, detail=""):
    st = _load_status()
    st[key] = {"ok": bool(ok), "detail": detail[-600:], "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(VIEWS, exist_ok=True)
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


# =============================================================================================================================
# Shared Plotly: one copy for the whole folder
# =============================================================================================================================
def ensure_assets():
    if not os.path.isfile(ASSETS):
        from plotly.offline import get_plotlyjs
        os.makedirs(os.path.dirname(ASSETS), exist_ok=True)
        with open(ASSETS, "w", encoding="utf-8") as f:
            f.write(get_plotlyjs())


_SRC = re.compile(r"""src=(['"])((?:\.\./)*)plotly\.min\.js\1""")
_INLINE = re.compile(r"<script[^>]*>\s*/\*\*\s*\*\s*plotly\.js.*?</script>", re.S)       # Plotly written into a page by include_plotlyjs=True


def localize_plotly(folder):
    """Make every page under `folder` use the one shared Plotly file (views/_assets/plotly.min.js): a page-local plotly.min.js is deleted,
    `src='plotly.min.js'` is repointed and a copy written inline into a page (several MB) is replaced by a link. Only the script tag changes."""
    ensure_assets()
    for dp, _, files in os.walk(folder):
        for fn in files:
            path = os.path.join(dp, fn)
            if fn == "plotly.min.js":
                os.remove(path)
            elif fn.endswith(".html"):
                with open(path, encoding="utf-8") as f:
                    s = f.read()
                rel = os.path.relpath(ASSETS, dp).replace("\\", "/")
                new = _SRC.sub(lambda m: f"src={m.group(1)}{rel}{m.group(1)}", s)
                new = _INLINE.sub(lambda m: f"<script charset='utf-8' src='{rel}'></script>", new)
                if new != s:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new)


# =============================================================================================================================
# Model inspection pages: states, reward vectors, transition matrix (read from the built model; nothing is computed here)
# =============================================================================================================================
_TABLE_JS = r"""
<script>
(function(){
  var D = JSON.parse(document.getElementById('data').textContent), cols = D.columns, rows = D.rows, num = D.numeric;
  var page = 0, size = 100, sortCol = -1, sortAsc = true, filt = [], view = rows.slice();
  var head = document.getElementById('thead'), body = document.getElementById('tbody'), info = document.getElementById('info');
  var h1 = '<tr>', h2 = '<tr class="f">';
  cols.forEach(function(c, i){ h1 += '<th data-i="'+i+'" title="click to sort">'+c+'</th>'; h2 += '<th><input data-i="'+i+'" placeholder="'+(num[i]?'e.g. >0.1, <5, =3':'contains')+'"></th>'; });
  head.innerHTML = h1 + '</tr>' + h2 + '</tr>';
  function ok(v, q, isNum){
    if(q==='') return true;
    if(isNum){ var m=/^\s*(>=|<=|>|<|=)?\s*(-?[0-9.eE+-]+)\s*$/.exec(q); if(m){ var x=parseFloat(m[2]), o=m[1]||'=';
      return o==='>'?v>x:o==='<'?v<x:o==='>='?v>=x:o==='<='?v<=x:Math.abs(v-x)<1e-12; } }
    return String(v).toLowerCase().indexOf(q.toLowerCase())>=0;
  }
  function apply(){
    view = rows.filter(function(r){ for(var i=0;i<cols.length;i++){ if(filt[i] && !ok(r[i], filt[i], num[i])) return false; } return true; });
    if(sortCol>=0) view.sort(function(a,b){ var x=a[sortCol], y=b[sortCol]; return (x<y?-1:x>y?1:0)*(sortAsc?1:-1); });
    page = 0; draw();
  }
  function fmt(v,i){ return (num[i] && typeof v==='number' && !Number.isInteger(v)) ? Number(v.toPrecision(7)) : v; }
  function draw(){
    var s = page*size, e = Math.min(view.length, s+size), out = '';
    for(var k=s;k<e;k++){ out += '<tr>'; for(var i=0;i<cols.length;i++) out += '<td>'+fmt(view[k][i],i)+'</td>'; out += '</tr>'; }
    body.innerHTML = out;
    info.textContent = (view.length?s+1:0)+'-'+e+' of '+view.length.toLocaleString()+' rows'+(view.length!==rows.length?' (filtered from '+rows.length.toLocaleString()+')':'');
  }
  head.addEventListener('click', function(ev){ var t=ev.target; if(t.tagName==='TH' && t.dataset.i!==undefined){ var i=+t.dataset.i; sortAsc=(sortCol===i)?!sortAsc:true; sortCol=i; apply(); }});
  head.addEventListener('input', function(ev){ var t=ev.target; if(t.tagName==='INPUT'){ filt[+t.dataset.i]=t.value; apply(); }});
  document.getElementById('prev').onclick=function(){ if(page>0){page--;draw();} };
  document.getElementById('next').onclick=function(){ if((page+1)*size<view.length){page++;draw();} };
  window.viewRows=function(){return view;}; window.viewCols=cols;
  document.getElementById('dl').onclick=function(){ var t=[cols.join(',')]; view.forEach(function(r){ t.push(r.map(function(v){return '"'+String(v).replace(/"/g,'""')+'"';}).join(',')); });
    var a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([t.join('\n')],{type:'text/csv'})); a.download='filtered_rows.csv'; a.click(); };
  apply();
})();
</script>
"""

_CSS = ("body{font-family:'Segoe UI',sans-serif;margin:0;background:#f8f9fa;color:#2c3e50}.header{background:linear-gradient(135deg,#2c3e50,#34495e);color:#fff;padding:1.2rem 2rem}"
        ".header h1{margin:0 0 .3rem}.wrap{max-width:1500px;margin:0 auto;padding:1rem 2rem}.box{background:#fff;border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,.06)}"
        "table{border-collapse:collapse;font-size:.85rem;width:100%}td,th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#eef2f5;cursor:pointer;position:sticky;top:0}"
        "tr.f th{cursor:default;background:#f8f9fa}tr.f input{width:100%;box-sizing:border-box;font-size:.8rem}.scroll{overflow-x:auto}.muted{color:#666;font-size:.85rem}"
        "button{padding:3px 10px;margin-right:6px}a{color:#2471a3}")


def _table_page(path, title, subtitle, columns, rows, numeric, csv_rel, extra_top="", head="", body_end=""):
    """One self-contained page: a sortable, filterable, paged table of `rows` (every row is embedded) + a link to the CSV with the same data."""
    payload = json.dumps({"columns": columns, "rows": rows, "numeric": numeric}, separators=(",", ":"), default=float).replace("</", "<\\/")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_CSS}</style>{head}</head><body>"
            f"<div class='header'><h1>{html.escape(title)}</h1><div>{subtitle}</div></div><div class='wrap'>"
            f"<p class='muted'><a href='../../index.html'>&larr; all reports</a> | underlying data: <a href='{csv_rel}'>{os.path.basename(csv_rel)}</a></p>{extra_top}"
            f"<div class='box'><p class='muted'>Click a column name to sort. Type in the box under a column to filter: text = contains; numeric columns accept "
            f"<code>&gt;0.1</code>, <code>&lt;=5</code>, <code>=3</code>. <span id='info'></span></p>"
            f"<p><button id='prev'>&larr; previous 100</button><button id='next'>next 100 &rarr;</button><button id='dl'>download the filtered rows (CSV)</button></p>"
            f"<div class='scroll'><table><thead id='thead'></thead><tbody id='tbody'></tbody></table></div></div></div>"
            f"<script id='data' type='application/json'>{payload}</script>{_TABLE_JS}{body_end}</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def _describe_states(domain, states):
    """A readable label per state, from the domain's own state columns (labels only; no model quantity is computed)."""
    if domain == "cow":
        from cowmrp.domains.cow.parameters import UNIT_DAYS
        from cowmrp.domains.cow.states import repro_status
        mim = states["dim"].to_numpy() * UNIT_DAYS // 30 + 1                   # month in milk, as the cow run page labels it
        st = repro_status(states["dp"].to_numpy())
        return [f"L{l}, MIM {m}, {s.lower()}" for l, m, s in zip(states["lactation"], mim, st)]
    cols = [c for c in states.columns if c not in ("state_id", "absorbing", "alive", "age_days")]
    return [", ".join(f"{c}={r[c]}" for c in cols) for _, r in states.iterrows()]


def _records(df):
    return [[(v.item() if hasattr(v, "item") else v) for v in row] for row in df.itertuples(index=False, name=None)]


def build_inspection(domain):
    """states.html, reward_vectors.html, transition_matrices.html and the data files for `domain` at month resolution. Returns the registry keys written."""
    import numpy as np
    import pandas as pd
    from runners._shared import engine_run
    from cowmrp.domain import load_domain
    from cowmrp.io import export_model
    from cowmrp.mrp import core

    dom = load_domain(domain)
    model = engine_run.build(domain, strict=True)
    out = os.path.join(VIEWS, domain, "month")
    data = os.path.join(VIEWS, domain, "data", "month")
    os.makedirs(out, exist_ok=True)
    state_cols = ["line", "stage", "age", "dp", "state_id"] if domain == "youngstock" else None
    export_model(model, dom, data, state_cols=state_cols, parquet=False)                    # the same files the runners write (states, long matrix, npz, reward vector)
    states = model.states.copy()
    states.insert(0, "description", _describe_states(domain, states))
    n, nnz = len(states), int(model.P.nnz)
    created = []                  # registry keys of the pages written

    # ---- states
    cols = list(states.columns)
    num = [pd.api.types.is_numeric_dtype(states[c]) and not pd.api.types.is_bool_dtype(states[c]) for c in cols]
    _table_page(os.path.join(out, "states.html"), f"{domain}: states", f"{n:,} states, month resolution. <code>description</code> is a label built from the state columns.",
                cols, _records(states), num, "../data/month/states.csv")
    created.append(f"{domain}-states")

    # ---- reward vector (the exported table: every component of every state)
    if model.rewards is not None:
        rw = core.export_rewards(model.states, model.rewards, column_names=dom.export_column_names)
        rw.insert(0, "state_id", model.states["state_id"].to_numpy())
        rw.insert(1, "description", _describe_states(domain, model.states))
        rcols = list(rw.columns)
        rnum = [pd.api.types.is_numeric_dtype(rw[c]) for c in rcols]
        _table_page(os.path.join(out, "reward_vectors.html"), f"{domain}: reward vectors",
                    f"{n:,} states x {len(rcols) - 3} columns: the reward components (<code>total</code> = milk_income + pregnancy_reward + cull_reward - mortality_cost - feed_cost - reproduction_cost), "
                    "the environmental rewards and the diagnostic quantities, exactly as in the exported reward vector. Costs are stored as positive numbers; <code>total</code> applies the signs.",
                    rcols, _records(rw), rnum, f"../data/month/reward_vector_{domain}.csv")
        created.append(f"{domain}-rewards")

    # ---- transition matrix: heatmap of every non-zero entry + a table of every row
    coo = model.P.tocoo()
    order = np.lexsort((coo.col, coo.row))
    src, tgt, prob = coo.row[order], coo.col[order], coo.data[order]
    desc = np.array(_describe_states(domain, model.states), dtype=object)
    tab = pd.DataFrame({"src_state_id": src, "source": desc[src], "tgt_state_id": tgt, "target": desc[tgt], "probability": prob})
    tcols = list(tab.columns)
    tnum = [c in ("src_state_id", "tgt_state_id", "probability") for c in tcols]
    heat = (f"<div class='box'><p class='muted'>Every non-zero entry of the {n:,} x {n:,} matrix is one point: x = target state id, y = source state id (top = 0), colour = probability. "
            f"{nnz:,} entries; zoom with the mouse, hover for the two states. Rows sum to 1 (checked when the model is built).</p><div id='heat' style='height:640px'></div></div>")
    heat_js = ("<script>(function(){var T=JSON.parse(document.getElementById('hd').textContent);"
               "var tr={type:'scattergl',mode:'markers',x:T.x,y:T.y,marker:{size:4,color:T.p,colorscale:'Viridis',cmin:0,cmax:1,colorbar:{title:'probability'}},"
               "text:T.t,hovertemplate:'%{text}<br>p = %{marker.color:.4f}<extra></extra>'};"
               "Plotly.newPlot('heat',[tr],{xaxis:{title:'target state id'},yaxis:{title:'source state id',autorange:'reversed'},margin:{l:60,r:10,t:10,b:50},template:'plotly_white'},{responsive:true});})();</script>")
    hd = json.dumps({"x": tgt.tolist(), "y": src.tolist(), "p": [float(v) for v in prob], "t": [f"{desc[s]}  &rarr;  {desc[t]}" for s, t in zip(src, tgt)]}, separators=(",", ":")).replace("</", "<\\/")
    _table_page(os.path.join(out, "transition_matrices.html"), f"{domain}: transition matrix",
                f"{n:,} states, {nnz:,} non-zero transitions (month resolution). Each row of the table is one transition: where a state goes and with what probability.",
                tcols, _records(tab), tnum, f"../data/month/transition_matrix_{domain}_long.csv",
                extra_top=heat + f"<script id='hd' type='application/json'>{hd}</script>", head="<script src='../../_assets/plotly.min.js'></script>", body_end=heat_js)
    created.append(f"{domain}-transitions")
    return created


# =============================================================================================================================
# Builders that run an existing report script (each in its own process)
# =============================================================================================================================
def _run(cmd, env_extra=None, timeout=3600):
    env = dict(os.environ, **(env_extra or {}))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout + "\n" + r.stderr)[-3000:]


def _py(code, *args):
    return [sys.executable, "-X", "utf8", "-c", code, *args]


def build_validation():
    out = os.path.join(VIEWS, "cow", "validation")
    ok, log = _run([sys.executable, "-X", "utf8", os.path.join(ROOT, "runners", "cow", "plots", "plot_validation_plotly.py"), "--out", out])
    if ok:
        localize_plotly(out)
    return ok, log


def build_review(domain):
    out = os.path.join(VIEWS, domain, "month", "review")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from runners._shared import review; "
            "print(review.build(sys.argv[2], out_dir=sys.argv[3], home_href='../../../../index.html'))")       # detail pages sit at views/<domain>/month/review/items/
    ok, log = _run(_py(code, ROOT, domain, out), {"COW_TIME_UNIT": "month"})
    if ok:
        localize_plotly(out)
    return ok, log


def build_projection():
    out = os.path.join(VIEWS, "cow", "projection")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import runners.cow.projection as p; p.OUTPUT_FOLDER = sys.argv[2]; "
            "print('RESOLUTIONS', sorted({s.get('time_unit') for s in p.SCENARIOS})); p.main()")
    ok, log = _run(_py(code, ROOT, out))
    if ok:
        localize_plotly(out)
    return ok, log


def build_farm(steps, n_cows):
    out = os.path.join(VIEWS, "cow", "month", "farm")
    os.makedirs(out, exist_ok=True)
    herd = os.path.join(out, "farm_data.csv")
    farm = os.path.join(ROOT, "runners", "cow", "farm")
    ok, log = _run([sys.executable, "-X", "utf8", os.path.join(farm, "generate.py"), "--n", str(n_cows), "--time-unit", "month", "--out", herd])
    if not ok:
        return ok, log
    ok2, log2 = _run([sys.executable, "-X", "utf8", os.path.join(farm, "simulate.py"), "--mode", "expected", "--steps", str(steps), "--time-unit", "month",
                      "--farm-csv", herd, "--out-dir", out])
    if ok2:
        static = os.path.join(out, "dashboard.html")           # the static copy of the same dashboard is not part of the views
        if os.path.exists(static):
            os.remove(static)
        localize_plotly(out)
    return ok2, log + log2


def build_youngstock_dashboard():
    ok, log = _run([sys.executable, "-X", "utf8", os.path.join(ROOT, "runners", "youngstock", "run.py"), "--time-unit", "month"])
    if not ok:
        return ok, log
    src = os.path.join(ROOT, "runners", "youngstock", "outputs", "dashboard", "youngstock_dashboard.html")       # written just now by the runner
    out = os.path.join(VIEWS, "youngstock", "month", "dashboard")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(src, os.path.join(out, "index.html"))
    localize_plotly(out)
    return True, log


# =============================================================================================================================
# Central page
# =============================================================================================================================
def _title_of(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(r"<title>(.*?)</title>", f.read(4000), re.S)
        return html.unescape(m.group(1).strip()) if m and m.group(1).strip() else ""
    except OSError:
        return ""


def _glob(pattern):
    import glob
    full = sorted(glob.glob(os.path.join(VIEWS, pattern.replace("/", os.sep))))
    return [os.path.relpath(p, VIEWS).replace("\\", "/") for p in full]


def _cell(rep, status):
    """(status class, html of the status/link cell, html of the pages list) for one report."""
    if rep.unavailable and not (rep.entry and os.path.isfile(os.path.join(VIEWS, rep.entry))):
        return "na", f"not available for this domain<br><span class='why'>{html.escape(rep.unavailable)}</span>", ""
    ok = os.path.isfile(os.path.join(VIEWS, rep.entry))
    st = status.get(rep.key)
    if not ok:
        if st and not st["ok"]:
            return "bad", f"FAILED at {html.escape(st['utc'])}<br><span class='why'>{html.escape(st['detail'][-300:])}</span>", ""
        return "todo", "not generated yet<br><span class='why'>python runners/run_all.py --views</span>", ""
    mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(VIEWS, rep.entry))).strftime("%Y-%m-%d %H:%M")
    cell = f"<a href='{rep.entry}'>open</a> <span class='why'>built {mtime}</span>"
    if st and not st["ok"]:
        cell += f"<br><span class='bad'>the last rebuild FAILED ({html.escape(st['utc'])}); this is an older version</span>"
    pages = ""
    if rep.pages:
        files = [f for f in _glob(rep.pages) if f != rep.entry]
        if files:
            items = "".join(f"<li><a href='{f}'>{html.escape(_title_of(os.path.join(VIEWS, f)) or os.path.basename(f))}</a> <span class='why'>{os.path.basename(f)}</span></li>" for f in files)
            pages = f"<details><summary>{len(files)} individual pages</summary><ul>{items}</ul></details>"
    return "ok", cell, pages


def _overview_section(domain, status):
    """The review grid of `domain` for the central page: the fragment the review builder wrote (runners/_shared/review.py: the same tiles and
    groups as the standalone review page), with its links pointed at the review folder. Nothing is drawn here."""
    rel = f"{domain}/month/review"
    frag = os.path.join(VIEWS, rel, "_overview.html")
    head = f"<h2 id='{domain}'>{domain.capitalize()} model review <span class='why'><a href='{rel}/index.html'>standalone page</a></span></h2>"
    if not os.path.isfile(frag):
        st = status.get(f"{domain}-review")
        why = f"FAILED at {html.escape(st['utc'])}: {html.escape(st['detail'][-300:])}" if st and not st["ok"] else "not generated yet (python runners/run_all.py --views)"
        return f"{head}<p class='why'>Review grid {why}</p>"
    with open(frag, encoding="utf-8") as f:
        body = f.read().replace("href='items/", f"href='{rel}/items/")
    st = status.get(f"{domain}-review")
    stale = f"<p class='bad'>the last rebuild of this review FAILED ({html.escape(st['utc'])}); the grid below is an older version</p>" if st and not st["ok"] else ""
    return head + stale + body


def write_index():
    from runners._shared import common, review
    os.makedirs(VIEWS, exist_ok=True)
    status = _load_status()
    blocks, counts = [], {"ok": 0, "todo": 0, "bad": 0, "na": 0}
    for domain in DOMAINS:
        rows, last = [], None
        for rep in (r for r in REPORTS if r.domain == domain):
            cls, cell, pages = _cell(rep, status)
            counts[cls] += 1
            data = " ".join(f"<a href='{d}'>{os.path.basename(d)}</a>" for pat in rep.data for d in _glob(pat))
            note = f"<br><span class='why'>{html.escape(rep.note)}</span>" if rep.note else ""
            kind = f"<b>{html.escape(rep.rtype)}</b>" if rep.rtype != last else ""
            last = rep.rtype
            rows.append(f"<tr class='{cls}'><td>{kind}</td><td>{html.escape(rep.title)}{note}{pages}</td><td>{html.escape(rep.resolution)}</td><td>{cell}</td><td>{data}</td></tr>")
        blocks.append(f"<h3>{domain}</h3><table><tr><th>Report type</th><th>Report</th><th>Resolution</th><th>Status</th><th>Data (CSV)</th></tr>{''.join(rows)}</table>")
    css = (common.PAGE_CSS + review.EXTRA_CSS +
           ".header{text-align:left;padding:1.5rem 2rem}.header h1{margin:0}.wrap{max-width:1400px;margin:0 auto;padding:1rem 2rem}"
           "table{width:100%;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:1.5rem}td,th{padding:8px 10px;text-align:left;vertical-align:top}th{background:#eef2f5}"
           ".why{color:#777;font-size:.8rem;font-weight:400}tr.todo td{background:#fff8e1}tr.bad td{background:#fdecea}tr.na td{background:#f2f2f2;color:#666}.bad{color:#b03a2e;font-size:.8rem}"
           "a{color:#2471a3}details ul{margin:.3rem 0;padding-left:1.2rem;font-size:.9rem}.bar a{margin-right:18px;font-weight:600}"
           "details.section>summary{cursor:pointer;list-style:none}details.section>summary h2{display:inline-block;margin:0 0 .5rem 0}details.section>summary::before{content:'▾ ';color:#3498db}"
           "details.section:not([open])>summary::before{content:'▸ '}.ovsum p{margin:.2rem 0;color:#555}h2{margin-top:2rem}")
    grids = "".join(_overview_section(d, status) for d in DOMAINS)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>CowMRP reports</title><style>{css}</style></head><body>"
            f"<div class='header'><h1>CowMRP: local reports</h1><div>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by runners/_shared/views.py. "
            f"{counts['ok']} reports available, {counts['todo']} not generated yet, {counts['bad']} failed, {counts['na']} not available for the domain.</div></div><div class='wrap'>"
            f"<p class='bar'>{''.join(f'<a href=#{d}>{d.capitalize()}</a>' for d in DOMAINS)}<a href='#other'>Other reports</a></p>"
            f"{grids}<h2 id='other'>Other reports</h2>"
            f"<p class='why'>Local and Git-ignored. Rebuild everything: <code>python runners/run_all.py --views</code>. Every report of the package is listed here; "
            f"a report that is missing says why. The review grids above are written by the review builder; any other new report is added by one entry in "
            f"<code>REPORTS</code> in runners/_shared/views.py.</p>{''.join(blocks)}</div></body></html>")
    with open(os.path.join(VIEWS, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.join(VIEWS, "index.html")


# =============================================================================================================================
# Driver
# =============================================================================================================================
def build(only=None, skip=(), steps=24, n_cows=10000):
    """Build the reports of the domains in `only` (default: all) and rewrite the central page. A failing report is recorded, shown on the
    page as FAILED, and does not stop the others. Returns {report key: 'ok' | 'failed' | 'skipped'}."""
    doms = [d for d in DOMAINS if not only or d in only]
    os.makedirs(VIEWS, exist_ok=True)
    ensure_assets()
    jobs = []
    for d in doms:
        jobs.append((f"{d}-inspection", [f"{d}-states", f"{d}-rewards", f"{d}-transitions"], lambda d=d: build_inspection(d)))
        jobs.append((f"{d}-review", [f"{d}-review"], lambda d=d: build_review(d)))
    if "cow" in doms:
        jobs += [("cow-validation", ["cow-validation", "cow-validation-all"], build_validation), ("cow-projection", ["cow-projection"], build_projection),
                 ("cow-farm", ["cow-farm"], lambda: build_farm(steps, n_cows))]
    if "youngstock" in doms:
        jobs.append(("youngstock-dashboard", ["youngstock-dashboard"], build_youngstock_dashboard))
    result = {}
    for name, keys, fn in jobs:
        if {name, name.split("-", 1)[1], *keys} & set(skip):
            result[name] = "skipped"
            continue
        print(f"[views] {name} ...", flush=True)
        try:
            r = fn()
            ok, detail = (True, "") if isinstance(r, list) else r
        except Exception as e:                                   # a failing report must not stop the others; it is shown on the page
            r, ok, detail = None, False, f"{type(e).__name__}: {e}"
        for k in (r if ok and isinstance(r, list) else keys):    # an inspection job reports the pages it wrote (youngstock has no reward vector)
            _record(k, ok, "" if ok else detail)
        result[name] = "ok" if ok else "failed"
        print(f"[views] {name}: {result[name]}" + ("" if ok else "\n" + detail[-800:]), flush=True)
        write_index()
    print(f"[views] central page: {write_index()}")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated domains (cow,youngstock)")
    ap.add_argument("--skip", default="", help="comma-separated report groups to skip, e.g. validation,farm")
    ap.add_argument("--steps", type=int, default=24, help="farm projection steps (months)")
    ap.add_argument("--n-cows", type=int, default=10000, help="herd size of the farm dashboard")
    ap.add_argument("--index-only", action="store_true", help="only rebuild views/index.html from what is on disk")
    a = ap.parse_args(argv)
    if a.index_only:
        print(write_index())
        return 0
    res = build(a.only.split(",") if a.only else None, tuple(x for x in a.skip.split(",") if x), a.steps, a.n_cows)
    return 1 if "failed" in res.values() else 0


if __name__ == "__main__":
    sys.exit(main())

"""
plot_validation_plotly.py
=========================
Interactive (Plotly) validation figures for CowMRP, drawn from the model's ACTUAL tables, probability arrays and MRP solutions at
MONTH, WEEK and DAY resolution (runners/cow/plots/_plot_data.py collects them, one process per resolution). Nothing is re-implemented here.

    python runners/cow/plots/plot_validation_plotly.py [--png <dir>] [--out <dir>]      (--png: also static PNG copies; needs the optional kaleido package; --out: another output folder)

Output (runners/cow/outputs/validation/plotly/):
  P01_... .html .. P14_... .html   one standalone interactive file per figure (zoom, hover, click legend entries to hide/show)
  validation_all.html              every figure on one page
  index.html                       list with captions, evidence labels and definitions

Conventions used in every figure
  - colour = LACTATION (1 to 10, dark purple to yellow); line style = RESOLUTION (solid = day, dashed = week, dotted + markers = month)
  - one axis per biological clock: days in milk (DIM), days pregnant, lactation number
  - U = length of one model step: 30 days (MONTH), 7 days (WEEK), 1 day (DAY)
  - dots/triangles/diamonds labelled with a paper = literature values (the evidence the table was built from, or an independent check)
"""
import os, sys, json, subprocess, html
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from runners._shared import common
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
import plotly.offline as po

OUT = common.module_outputs("cow", "validation", "plotly")
if "--out" in sys.argv:       # optional: write the pages to another folder; without it the folder above is used
    OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
UNITS = ("month", "week", "day")
LACS = list(range(1, 11))
LCOL = {L: sample_colorscale("Viridis", (L - 1) / 9.0)[0] for L in LACS}
RES = {"day": dict(dash="solid", width=1.6), "week": dict(dash="dash", width=1.4), "month": dict(dash="dot", width=1.4)}
RCOL = {"month": "#d95f02", "week": "#1b9e77", "day": "#7570b3"}
FIGS = []        # (file, title, caption, fig)

os.environ.setdefault("COW_TIME_UNIT", "month")
from cowmrp.domains.cow.evidence import pinedo2010 as PIN, miller2008 as MIL        # noqa: E402  (evidence values only, no model logic)


from cowmrp.domains.cow.registry import lactation_provenance          # noqa: E402  single source of the per-lactation evidence class
PROV = lactation_provenance()
DASH = {"direct": "solid", "adapted": "solid", "grouped": "dash", "repeated": "dot", "placeholder": "dashdot", "policy": "solid", "sensitivity": "dot"}


def plab(mech, L):
    """trace name with the evidence class, e.g. 'L4 [grouped]': a populated lactation is not an independent estimate"""
    return f"L{L} [{PROV[mech][L][0]}]"


def collect_all():
    data = {}
    for unit in UNITS:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "plots", "_plot_data.py")], env=dict(os.environ, COW_TIME_UNIT=unit),
                           cwd=ROOT, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            print(p.stderr[-3000:]); sys.exit(1)
        data[unit] = json.loads(line[0][6:])
        print("collected", unit)
    return data


def per_day(p, U):
    return 1.0 - (1.0 - np.clip(np.asarray(p, float), 0, 1)) ** (1.0 / U)


def add(fig, name, title, caption):
    fig.update_annotations(font_size=12)
    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"), template="plotly_white", legend=dict(orientation="v", font=dict(size=10)),
                      margin=dict(l=60, r=20, t=70, b=55))
    FIGS.append((name, title, caption, fig))


def native_steps(arr_lm, U=30):
    """step-line coordinates for a monthly row"""
    x = np.repeat(np.arange(len(arr_lm) + 1) * U, 2)[1:-1]
    y = np.repeat(arr_lm, 2)
    return x, y


# =========================================================================================== figures
def fig_milk(data):
    d = data["day"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Wood curve (default)", "MilkBot curve (alternative)", "305-day milk by lactation"), column_widths=[.36, .36, .28])
    for c, m in ((1, "wood"), (2, "milkbot")):
        for L in LACS:
            rep = d["milk_basis"][m][str(L)] == "repeated"
            fig.add_trace(go.Scatter(x=d["curve_days"], y=d["curve"][m][L - 1], mode="lines", name=plab("milk_curve_wood" if m == "wood" else "milk_curve_milkbot", L),
                                     legendgroup=f"l{L}", showlegend=(c == 1), line=dict(color=LCOL[L], width=1.8, dash=DASH[PROV["milk_curve_wood" if m == "wood" else "milk_curve_milkbot"][L][0]])), row=1, col=c)
        fig.add_hline(y=d["floor"], line=dict(color="black", dash="dot", width=1), row=1, col=c)
        fig.add_vline(x=d["last_breeding_days"], line=dict(color="grey", dash="dash", width=1), row=1, col=c)
        fig.update_xaxes(title_text="days in milk", range=[0, 600], row=1, col=c)
    fig.update_yaxes(title_text="milk of an open cow, kg/day", row=1, col=1, range=[0, 50])
    fig.update_yaxes(range=[0, 50], row=1, col=2)
    fig.add_trace(go.Bar(x=LACS, y=d["m305"]["wood"], name="Wood", marker_color="#4c78a8", showlegend=False), row=1, col=3)
    fig.add_trace(go.Bar(x=LACS, y=d["m305"]["milkbot"], name="MilkBot", marker_color="#f58518", showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[9296, 10507, 10837], mode="markers", name="Ehrlich 2011 printed M305 (US Holstein)",
                             marker=dict(symbol="diamond", size=11, color="black")), row=1, col=3)
    fig.update_xaxes(title_text="lactation", dtick=1, row=1, col=3); fig.update_yaxes(title_text="kg per 305 days", row=1, col=3)
    add(fig, "P01_milk_curves", "Milk curves, lactations 1-10",
        "Potential milk of an open cow vs days in milk, for lactations 1-10 (read from the model's curve functions). Dotted horizontal line = the farmer's milk floor "
        "(22.7 kg/day), dashed vertical = last breeding day (300 d). The source gives parameters for lactations 1-3 only; lactations 4-10 are copies of lactation 3 "
        "(dashed, drawn on top of each other: no evidence they differ). Right: 305-day totals of both curves against the printed values of Ehrlich (2011) for US Holsteins "
        "(black diamonds). Wood's a, time-to-peak and peak come from a Brazilian herd; b and c are derived by this project; MilkBot parameters are the printed US values.")


def fig_pregfactor(data):
    d = data["day"]
    fig = go.Figure()
    nat = np.array(d["native"]["pregnancy_factor"][0])
    x, y = native_steps(nat)
    keep = x <= d["dry_off_dp_days"] + 30
    fig.add_trace(go.Scatter(x=x[keep], y=y[keep], mode="lines", name="table (30-day steps)", line=dict(color="black", shape="hv")))
    for u in UNITS:
        U = data[u]["U"]; f = np.array(data[u]["preg_factor_step"])
        xs = np.arange(len(f)) * U
        m = (np.arange(len(f)) > 0) & (xs + data[u]["U"] <= data[u]["dry_off_dp_days"] + 1e-9)    # steps that end before dry-off (a straddling step averages in dry days with zero milk)
        fig.add_trace(go.Scatter(x=xs[m], y=f[m], mode="lines+markers" if u == "month" else "lines", name=f"model, {u}", line=dict(color=RCOL[u], **RES[u])))
    fig.add_vline(x=d["dry_off_dp_days"], line=dict(color="grey", dash="dash"), annotation_text="dry-off")
    fig.update_xaxes(title_text="days pregnant", range=[0, 260]); fig.update_yaxes(title_text="milk multiplier (1 = open cow)", range=[0.6, 1.02])
    add(fig, "P02_pregnancy_milk_factor", "Effect of pregnancy on milk yield (PLACEHOLDER)",
        "Multiplier on an open cow's milk by days pregnant. This is an uncited modelling placeholder (a straight decline from 1 to 0.70 at dry-off, the same for every lactation). "
        "None of the reviewed papers gives this shape, so it is not validated. "
        "Shown only for model steps that end before dry-off; the step that contains dry-off averages in dry days with zero milk and is left out. The table is a staircase of 30-day values; model lines are averages over their step.")


def lit_points(fig, store, codes, row, col, status_months, parity_map, label, legend=True):
    first = legend
    for par in (1, 2, 3, 4):
        vals = None
        for c in codes:
            v = np.array(store[c][par][:status_months], float)
            vals = v if vals is None else vals + v
        x = np.arange(status_months) * 30 + 15
        fig.add_trace(go.Scatter(x=x, y=vals * PIN.UNIT * 1000, mode="markers", name=label, legendgroup="lit", showlegend=first,
                                 marker=dict(color=LCOL[par], size=6, symbol="circle-open", line=dict(width=1.5)),
                                 hovertext=[f"parity {par if par < 4 else '>=4'}" for _ in x]), row=row, col=col)
        first = False


def fig_hazard(data, key, codes, name, title, caption):
    d = data["day"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Open cows (days in milk)", "Pregnant cows (days pregnant, until dry-off)", "Dry cows (days pregnant, dry-off to calving)"), shared_yaxes=True)
    dry, ges = d["dry_off_dp_days"], d["gestation_days"]
    for si in range(3):
        for L in LACS:
            arr = np.array(d["native"][key][si][L - 1])
            x, y = native_steps(per_day(arr, 30) * 1000)
            if si == 0: msk = x <= 520
            elif si == 1: msk = x <= dry + 30
            else: msk = (x >= dry - 30) & (x <= ges + 30)
            mech = "mortality" if key == "mortality" else "other_reason_culling"
            fig.add_trace(go.Scatter(x=x[msk], y=y[msk], mode="lines", name=plab(mech, L), legendgroup=f"l{L}", showlegend=(si == 0),
                                     hovertext=PROV[mech][L][1], line=dict(color=LCOL[L], width=1.8, dash=DASH[PROV[mech][L][0]], shape="hv")), row=1, col=si + 1)
    lit_points(fig, PIN.OPEN, codes, 1, 1, 17, None, "Pinedo 2010 (read from graph; parity 1,2,3,>=4 = lactations 1-4)")
    lit_points(fig, PIN.PREGNANT, codes, 1, 2, 7, None, "Pinedo 2010", legend=False)
    fig.update_xaxes(title_text="days in milk", range=[0, 500], row=1, col=1)
    fig.update_xaxes(title_text="days pregnant", range=[0, dry], row=1, col=2)
    fig.update_xaxes(title_text="days pregnant", range=[dry, ges], row=1, col=3)
    fig.update_yaxes(title_text="hazard, per 1000 cow-days", row=1, col=1)
    add(fig, name, title, caption)


def fig_resolution_check(data, key, name, title):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Open", "Pregnant", "Dry"), shared_yaxes=True)
    d = data["day"]
    dry, ges = d["dry_off_dp_days"], d["gestation_days"]
    for si in range(3):
        L = 2
        nat = np.array(d["native"][key][si][L - 1]); x, y = native_steps(per_day(nat, 30) * 1000)
        msk = (x <= 500) if si == 0 else (x <= dry + 30) if si == 1 else ((x >= dry - 30) & (x <= ges + 30))
        fig.add_trace(go.Scatter(x=x[msk], y=y[msk], mode="lines", name="native monthly table", showlegend=(si == 0), line=dict(color="black", shape="hv", width=1)), row=1, col=si + 1)
        for u in UNITS:
            U = data[u]["U"]; arr = np.array(data[u]["step"][key][si][L - 1])
            if si == 0:
                xs = (np.arange(len(arr)) + 1) * U; m = xs <= 500
            else:
                xs = np.arange(len(arr)) * U
                dpi = np.arange(len(arr)); ds = data[u]["dry_off_dp"]
                m = ((dpi >= 1) & (dpi < ds)) if si == 1 else ((dpi >= ds) & (xs <= ges))
            fig.add_trace(go.Scatter(x=xs[m], y=per_day(arr[m], U) * 1000, mode="lines+markers" if u == "month" else "lines", name=f"model, {u}", showlegend=(si == 0),
                                     line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=si + 1)
    fig.update_xaxes(title_text="days in milk", row=1, col=1); fig.update_xaxes(title_text="days pregnant", row=1, col=2); fig.update_xaxes(title_text="days pregnant", row=1, col=3)
    fig.update_yaxes(title_text="hazard, per 1000 cow-days", row=1, col=1)
    add(fig, name, title,
        "Lactation 2 only. Black = the native monthly table (probability per 30 days turned into a per-day hazard). Coloured = the per-step probability the model actually uses at each resolution, turned into a per-day hazard "
        "(U = step length: 30 / 7 / 1 d). They agree on average; the monthly model line is a coarse average over 30-day steps, so it cannot reproduce within-month changes (not a unit error).")


def fig_provenance(data):
    mechs = [("mortality", "mortality"), ("other_reason_culling", "other-reason culling"), ("conception", "conception"), ("pregnancy_loss", "pregnancy loss"), ("calving", "calving (277 d)"),
             ("milk_floor_cull", "milk-floor rule"), ("pregnancy_milk_factor", "pregnancy milk factor"), ("milk_curve", "milk curve (active)")]
    cls = list(DASH)
    z = [[cls.index(PROV[m][L][0]) for L in LACS] for m, _ in mechs]
    colors = {"direct": "#66bb6a", "adapted": "#aed581", "grouped": "#fff176", "repeated": "#ffb74d", "placeholder": "#e57373", "policy": "#b39ddb", "sensitivity": "#bdbdbd"}
    fig = go.Figure(go.Heatmap(z=z, x=[f"L{L}" for L in LACS], y=[n for _, n in mechs], showscale=False,
                               colorscale=[[i / (len(cls) - 1), colors[c]] for i, c in enumerate(cls)], zmin=0, zmax=len(cls) - 1,
                               text=[[PROV[m][L][0] for L in LACS] for m, _ in mechs], texttemplate="%{text}",
                               hovertext=[[f"{PROV[m][L][1]} | {PROV[m][L][3]}" for L in LACS] for m, _ in mechs], hoverinfo="text"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(side="top")
    add(fig, "P00_evidence_by_lactation", "Evidence behind each lactation (all other figures use these labels)",
        "For every probability mechanism and every lactation 1-10: direct = the source reports that lactation/parity (graph reads carry +/-10-15 %); adapted = a documented transformation; grouped = expanded from a grouped source category (3+, >=4, >=8); "
        "repeated = no distinct estimate, another lactation's value is copied; placeholder = no evidence; policy = farmer rule. Ten populated rows are not ten independent estimates. Hover for the source category. Generated from domains.cow/registry.py.")


def fig_exits(data):
    d = data["week"]; ex = d["exits"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Death per lactation, %", "Death per cow-year, %", "Live culling per cow-year, %", "Exits per lactation, % of entrants"))
    L_ = [e["L"] for e in ex if e["entry"] > 1e-9]
    death_l = [e["death"] * 100 for e in ex if e["entry"] > 1e-9]
    fig.add_trace(go.Scatter(x=L_, y=death_l, mode="lines+markers", name="model", line=dict(color="#4c78a8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, 9)), y=[MIL.DEATH_PER_LACTATION_PCT[i] for i in range(1, 9)], mode="markers", name="Miller 2008 (lower bound: misses early and dry-period deaths)",
                             marker=dict(symbol="diamond", size=10, color="black")), row=1, col=1)
    dy = [(e["death"] * e["entry"]) / e["cow_years"] * 100 for e in ex if e["entry"] > 1e-9]
    cy = [((e["other"] + e["milk_floor"]) * e["entry"]) / e["cow_years"] * 100 for e in ex if e["entry"] > 1e-9]
    fig.add_trace(go.Scatter(x=L_, y=dy, mode="lines+markers", name="model", showlegend=False, line=dict(color="#4c78a8")), row=1, col=2)
    fig.add_trace(go.Scatter(x=L_, y=cy, mode="lines+markers", name="model", showlegend=False, line=dict(color="#4c78a8")), row=1, col=3)
    t1 = PIN.TABLE1_ANNUAL
    for col_, key in ((2, "death_by_parity"), (3, "live_cull_by_parity")):
        fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[t1[key][p] for p in (1, 2, 3, 4)], mode="markers", name="Pinedo 2010 Table 1 (parity 1,2,3,>=4)", showlegend=(col_ == 2),
                                 marker=dict(symbol="diamond", size=10, color="black")), row=1, col=col_)
        fig.add_trace(go.Scatter(x=[4, 10], y=[t1[key][4]] * 2, mode="lines", showlegend=False, line=dict(color="black", dash="dot", width=1)), row=1, col=col_)
    cols = {"death": "#e45756", "other": "#f58518", "milk_floor": "#72b7b2", "max_lac": "#9d9d9d"}
    labels = {"death": "death", "other": "other culling (Pinedo hazards)", "milk_floor": "milk-floor rule (management)", "max_lac": "forced exit at lactation 10"}
    for k in ("death", "other", "milk_floor", "max_lac"):
        fig.add_trace(go.Bar(x=L_, y=[e[k] * 100 for e in ex if e["entry"] > 1e-9], name=labels[k], marker_color=cols[k]), row=1, col=4)
    fig.update_layout(barmode="stack")
    for c in range(1, 5): fig.update_xaxes(title_text="lactation", dtick=1, row=1, col=c)
    add(fig, "P05_exits_by_lactation", "Exits by lactation: model vs literature",
        "WEEK resolution. Panels 1-3: the model (blue) against Miller et al. (2008) (death per lactation; their data miss deaths before the first test day, so they are a lower bound) and Pinedo et al. (2010) "
        "annualized rates (printed for parity 1, 2, 3 and >=4; the dotted line repeats the '>=4' value). Panel 4: what happens to cows entering each lactation. IMPORTANT: the milk-floor share is NOT a literature number; it is the outcome of the farmer rule "
        "(open cows below the milk floor after day 300 are replaced) interacting with the conception level. Death is below the literature mainly because that rule removes long-open cows before their high death hazards act "
        "(switching the rule off gives 6.6 % per year, equal to Pinedo's 6.6 %).")


def fig_conception(data):
    d = data["day"]; w = data["week"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Conception probability per 30 days", "Pregnant cows, % (all lactations)", "Pregnant cows, % (lactations 1-3 vs literature)"))
    for L in LACS:
        nat = np.array(d["native"]["conception"][L - 1]); x, y = native_steps(nat)
        keep = (x > d["vwp"] - 30) & (x < d["last_breeding_days"] + 30)       # eligible window only: the model applies the waiting period and last breeding day by day
        x = np.clip(x[keep], d["vwp"], d["last_breeding_days"]); y = y[keep]; m = np.ones(len(x), bool)
        rep = d["basis"]["conception_full"][str(L)] == ["repeated"]
        fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="lines", name=plab("conception", L), legendgroup=f"l{L}",
                                 line=dict(color=LCOL[L], shape="hv", width=1.8, dash=DASH[PROV["conception"][L][0]])), row=1, col=1)
        pb = d["pregnant_by_dim"].get(str(L))
        if pb:
            xs_ = np.array(pb["dim_days"]); ys_ = np.array(pb["preg_share"]) * 100; mk = xs_ <= 300
            fig.add_trace(go.Scatter(x=xs_[mk], y=ys_[mk], mode="lines", name=f"lactation {L}", legendgroup=f"l{L}", showlegend=False,
                                     line=dict(color=LCOL[L], width=1.8)), row=1, col=2)
    fig.add_vline(x=d["vwp"], line=dict(color="grey", dash="dash"), row=1, col=1, annotation_text="waiting period 50 d")
    fig.add_vline(x=d["last_breeding_days"], line=dict(color="grey", dash="dash"), row=1, col=1, annotation_text="last breeding 300 d")
    for L in (1, 2, 3):
        pb = d["pregnant_by_dim"][str(L)]
        xs_ = np.array(pb["dim_days"]); ys_ = np.array(pb["preg_share"]) * 100; mk = xs_ <= 300
        fig.add_trace(go.Scatter(x=xs_[mk], y=ys_[mk], mode="lines", name=f"model, lactation {L}", line=dict(color=LCOL[L], width=2)), row=1, col=3)
    fig.add_trace(go.Scatter(x=[200, 200], y=[65.2, 55.4], mode="markers", name="Fodor 2019: pregnant by 200 DIM (primiparous 65.2, multiparous 55.4 %)",
                             marker=dict(symbol="diamond", size=12, color="black")), row=1, col=3)
    fig.add_trace(go.Scatter(x=[100, 150], y=[30, 46], mode="markers", name="Cook & Green 2016: conceived by 100 / 150 DIM (herds picked for POOR fertility; conception, not net pregnancy)",
                             marker=dict(symbol="triangle-up", size=11, color="grey")), row=1, col=3)
    fig.update_xaxes(title_text="days in milk", range=[0, 450])
    fig.update_yaxes(title_text="probability per 30 days", range=[0, 0.6], row=1, col=1); fig.update_yaxes(title_text="% of cows", range=[0, 100], row=1, col=2); fig.update_yaxes(range=[0, 100], row=1, col=3)
    add(fig, "P06_conception", "Conception and resulting pregnancy by lactation",
        "Curves stop at day 300, the last breeding day (after it only pregnant cows remain, which would push the share to 100 %). Left: the conception table drawn ONLY inside the eligible window (day 50 to day 300, dashed lines; one step line per lactation, the native 30-day values). Outside that window the model's conception probability is zero (waiting period and last breeding day are applied by day), so nothing is drawn. Lactation 1 is a PLACEHOLDER level; lactation 2 uses the odds ratio 0.68 and lactations 3-10 one grouped ratio 0.47 "
        "(from two studies), so lactations 3-10 lie on one line. Middle: the pregnancy share that RESULTS from conception, pregnancy loss and exits (an output, not an input). Right: lactations 1-3 against published values. The model is at or above the literature "
        "(about 80/72/62 % pregnant by 200 DIM vs 65/55 % in Fodor 2019): the conception level is the main remaining uncertainty and is tested in the sensitivity analysis (docs section 28).")


def fig_loss(data):
    from cowmrp.domains.cow.probabilities import abortion as A
    d = data["day"]
    days = np.arange(0, 290)
    h = A._albaaj_daily_hazard(days.astype(float))
    S = np.concatenate(([1.0], np.cumprod(1 - h)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pregnancy-loss hazard", "Cumulative loss"))
    for u in UNITS:
        U = data[u]["U"]; arr = np.array(data[u]["step"]["abortion"]); dpi = np.arange(len(arr)); m = (dpi >= 1) & (dpi * U <= d["gestation_days"])
        fig.add_trace(go.Scatter(x=(dpi[m] - 1) * U + U / 2, y=per_day(arr[m], U) * 1000, mode="lines+markers" if u == "month" else "lines", name=f"model, {u}", line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=1)
    fig.update_yaxes(type="log", title_text="loss per 1000 pregnant cow-days (log scale)", row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=(1 - S[:-1]) * 100, mode="lines", name="model: lost since conception", line=dict(color="black", width=2)), row=1, col=2)
    base = S[30]
    fig.add_trace(go.Scatter(x=days[30:], y=(1 - S[30:-1] / base) * 100, mode="lines", name="model: lost since day 30 (after the usual pregnancy diagnosis)", line=dict(color="#e45756", width=2)), row=1, col=2)
    def win(a, b): return (1 - S[b] / S[a]) * 100
    pts = [("Fernandez-Novo 2020 (days 28-110)", 110, 12.3, win(28, 110)), ("Corva 2025 (days ~28-210)", 210, 22.5, win(28, 210)),
           ("Lee & Kim 2007 (days 46-260)", 260, 6.9, win(46, 260)), ("Albaaj 2022 meta-analysis (days 19-90)", 90, None, win(19, 90))]
    for lab, end, val, mod in pts:
        if val is not None:
            fig.add_trace(go.Scatter(x=[end], y=[val], mode="markers", name=f"{lab}: {val:.1f} % (model, same window: {mod:.1f} %)", marker=dict(symbol="diamond", size=11)), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=[end], y=[mod], mode="markers", name=f"{lab}: model {mod:.1f} %, meta-analysis windows 27/13/7/2 %", marker=dict(symbol="x", size=11, color="black")), row=1, col=2)
    fig.update_xaxes(title_text="days pregnant", range=[0, 285]); fig.update_yaxes(title_text="% of conceptions lost", range=[0, 50], row=1, col=2)
    add(fig, "P07_pregnancy_loss", "Pregnancy loss by days pregnant",
        "Left: loss hazard (log scale) at the three resolutions; the hazard is highest at days 19-45 and falls steeply. Right: cumulative loss. The model counts loss from CONCEPTION (day 0), because its 'pregnant' state starts at conception and the "
        "Albaaj et al. (2022) meta-analysis reports 27 % loss at days 19-32, 13 % at 30-45, 7 % at 45-60 and 2 % at 60-90; so about 45 % of conceptions are lost by calving. From day 30 (when pregnancy is usually first diagnosed) the loss is the red line. "
        "Diamonds: three primary studies that report loss from a diagnosis day, each with the model's loss over the same window in the legend. HONEST FINDING: over those windows the model loses MORE than the studies "
        "(27 % vs 12.3 % for days 28-110; 29.6 % vs 22.5 % for days 28-210; 13.8 % vs 6.9 % for days 46-260), because the Albaaj et al. (2022) meta-analysis windows used for days 19-90 (13 % / 7 % / 2 % after day 30) are higher than those studies. "
        "The conception level was matched to days-open data using this same definition, so loss is not counted twice, but the post-diagnosis loss is at or above the top of the published range; its effect on results is quantified in docs section 28. "
        "Also: the abortion cost ($150) is charged on every loss, including embryonic loss a farmer never sees.")


def fig_calving(data):
    fig = go.Figure()
    g = data["day"]["gestation_days"]
    xs = np.linspace(g - 25, g + 40, 400)
    fig.add_trace(go.Scatter(x=xs, y=np.exp(-0.5 * ((xs - g) / 5.9) ** 2) / (5.9 * np.sqrt(2 * np.pi)) * 100, mode="lines", name="biological gestation length (Normal, SD 5.9 d; Mwangi 2025) - sensitivity only",
                             line=dict(color="black", dash="dot")))
    for u in UNITS:
        U = data[u]["U"]; mass = data[u]["calving_mass"]
        ks = np.array([int(k) for k in mass]); ms = np.array([mass[k] for k in mass])
        booked = ks * U + U / 2.0
        mean_b = float((booked * ms).sum())
        fig.add_trace(go.Bar(x=booked, y=ms * 100, name=f"{u}: calving booked at the END of its step (mean {mean_b:.0f} d)", marker_color=RCOL[u], opacity=.6, width=max(0.8, U * 0.35)))
    fig.add_vline(x=g, line=dict(color="black", dash="dash"), annotation_text=f"true gestation length {g} d")
    fig.update_xaxes(title_text="days pregnant at which the model books the calving", range=[g - 25, g + 40]); fig.update_yaxes(title_text="% of calvings")
    add(fig, "P08_calving_timing", "Calving timing at the three resolutions",
        "The model's gestation length is 277 days (dashed line). A step is U days long (30 / 7 / 1) and an event inside a step is booked when the step ENDS, so on average calving is booked U/2 days late: 0.5 d at DAY, 3.5 d at WEEK, 15 d at MONTH. "
        "That is why the monthly bars sit at about 285 and 315 d (two thirds / one third) instead of 277: it is time-step quantization, not a 30-day biological spread. This same booking delay is what makes MONTH value lower than DAY (docs section 21). "
        "Dotted black: the real spread of gestation length (SD 5.9 d), available as the calving 'model' for sensitivity.")


def fig_floor(data):
    d = data["day"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Lactations 1-10 (DAY)", "Lactation 2 at the three resolutions"), shared_yaxes=True)
    cp = np.array(d["cull_p"]); U = d["U"]
    for L in LACS:
        x = (np.arange(cp.shape[1]) + 1) * U
        fig.add_trace(go.Scatter(x=x, y=cp[L - 1], mode="lines", name=f"L{L} [policy; curve {PROV['milk_curve'][L][0]}]", legendgroup=f"l{L}", line=dict(color=LCOL[L], width=1.8 + (0 if L > 3 else 0.5), dash="dash" if L > 3 else "solid")), row=1, col=1)
    for u in UNITS:
        c2 = np.array(data[u]["cull_p"][1]); Uu = data[u]["U"]
        fig.add_trace(go.Scatter(x=(np.arange(len(c2)) + 1) * Uu, y=c2, mode="lines+markers" if u == "month" else "lines", name=f"{u}", line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=2)
    fig.add_vline(x=d["last_breeding_days"], line=dict(color="grey", dash="dash"), row=1, col=1, annotation_text="floor checked from day 300")
    fig.update_xaxes(title_text="days in milk", range=[280, 460]); fig.update_yaxes(title_text="probability an OPEN cow is replaced in this step", range=[-0.02, 1.05], row=1, col=1)
    add(fig, "P09_milk_floor_rule", "The milk-floor culling rule (a management policy)",
        "Open cows only (pregnant and dry cows are never checked). From day 300 on, the model looks at the milk of an average open cow of her lactation; on the first day it is at or below 22.7 kg/day (the farmer's floor) every open cow of that lactation is replaced. "
        "The crossing day is 392 d for lactation 1, 344 d for lactation 2 and 333 d for lactations 3-10 (the lactation 3 curve is repeated). The rule is deterministic: there is no cow-to-cow variation, so the whole open cohort leaves at once. "
        "That produces the sudden drop in cohort survival (figure P11) and is a property of the policy, not of the biology. Month steps (30 d) spread the jump over the step that contains the crossing day, so the monthly ramp starts earlier than the crossing day.")


def fig_herd(data):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Share of cow-time by days in milk", "By reproductive status (WEEK)", "By lactation"))
    for u in UNITS:
        occ = np.array(data[u]["occ"]["all"])[:20]
        fig.add_trace(go.Scatter(x=np.arange(20) * 30 + 15, y=occ, mode="lines+markers" if u == "month" else "lines", name=u, line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=1)
    for s, c in (("Open", "#1f77b4"), ("Pregnant", "#2ca02c"), ("Dry", "#ff7f0e")):
        occ = np.array(data["week"]["occ"]["by_status"][s])[:20]
        fig.add_trace(go.Scatter(x=np.arange(20) * 30 + 15, y=occ, mode="lines", name=s, line=dict(color=c)), row=1, col=2)
    for u in UNITS:
        fig.add_trace(go.Bar(x=LACS, y=data[u]["occ"]["by_lactation"], name=u, marker_color=RCOL[u], showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="days in milk (30-day bins)", range=[0, 600], row=1, col=1); fig.update_xaxes(title_text="days in milk (30-day bins)", range=[0, 600], row=1, col=2)
    fig.update_xaxes(title_text="lactation", dtick=1, row=1, col=3); fig.update_yaxes(title_text="share of all cow-time")
    add(fig, "P10_steady_state_herd", "Steady-state herd",
        "Where an equilibrium herd (every replaced cow replaced by a heifer) spends its time, from the model's own solution. The monthly curve is lower in the first bin (0-30 d) for the same reason as the calving timing: calving is booked at the end of "
        "a step, so the first part of the new lactation is booked as dry time (docs section 21). About 34 / 26 / 17 / 10 % of cow-time is in lactations 1 / 2 / 3 / 4, close to typical US herds.")


def fig_survival(data):
    fig = go.Figure()
    for u in UNITS:
        s = data[u]["cycle_survival"]
        fig.add_trace(go.Scatter(x=s["years"], y=np.array(s["alive"]) * 100, mode="lines", name=u, line=dict(color=RCOL[u], **RES[u])))
    fig.add_annotation(x=1.05, y=85, text="open cows below the milk floor are all replaced at once<br>(deterministic management rule, figure P09)", showarrow=True, ax=90, ay=-40)
    fig.update_xaxes(title_text="years since entering the herd as a replacement heifer", range=[0, 9]); fig.update_yaxes(title_text="% of the cohort not yet replaced", range=[0, 100])
    add(fig, "P11_cohort_survival", "Survival of a replacement cohort",
        "Share of a cohort of replacement heifers still in the herd, from the model's transition matrix. The sudden drop at about one year (and a smaller one near 1.8 years) is the milk-floor rule acting on all open lactation-1 (2) cows on the same day; "
        "it is an artefact of the deterministic rule, not a biological pattern. Lines for the three resolutions nearly coincide.")


def fig_values(data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Value of a pregnancy, lactation 2", "Value of replacing an open cow, lactation 2"))
    for u in UNITS:
        pv = data[u]["pv"]; rv = data[u]["rv"]
        fig.add_trace(go.Scatter(x=pv["gest_days"], y=pv["value"], mode="lines+markers" if u == "month" else "lines", name=f"{u} (conceived at about day {pv['conceived_at_dim_days']:.0f} in milk)", line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=rv["dim_days"], y=rv["value"], mode="lines+markers" if u == "month" else "lines", name=u, showlegend=False, line=dict(color=RCOL[u], **RES[u]), marker=dict(size=5)), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="black", width=1), row=1, col=2)
    fig.update_xaxes(title_text="days pregnant", row=1, col=1); fig.update_xaxes(title_text="days in milk (open cow)", row=1, col=2)
    fig.update_yaxes(title_text="US$: value of the pregnant cow minus the same cow open", row=1, col=1); fig.update_yaxes(title_text="US$: value of keeping her minus value of a new heifer", row=1, col=2)
    add(fig, "P12_values", "Value of pregnancy and value of replacement",
        "Both come from the model's solved values. Left: a cow conceiving at about day 90 in milk (the same age at all three resolutions) is worth more the longer she has been pregnant, because her future cash flow is already secured; De Vries (2006) reports $0-551 for a NEW pregnancy and "
        "$517-525 for losing a 4-month pregnancy in lactation 2. Right: an open cow's value relative to a heifer falls with days in milk and turns negative (replace her) once her expected remaining profit is lower than a heifer's. "
        "The monthly values are lower for the booking reason in P08.")


def fig_rewards(data):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Milk income", "Feed cost", "Reproduction cost (AI, diagnosis, synchronization)"))
    for L in (1, 2, 3):
        for u in UNITS:
            r = data[u]["reward_day"][str(L)]
            for c, k in enumerate(("milk_income", "feed_cost", "reproduction_cost")):
                fig.add_trace(go.Scatter(x=r["dim_days"], y=r[k], mode="lines+markers" if u == "month" else "lines", name=f"lactation {L}, {u}", legendgroup=f"{L}{u}", showlegend=(c == 0),
                                         line=dict(color=LCOL[L], **RES[u]), marker=dict(size=4)), row=1, col=c + 1)
    for c in (1, 2, 3):
        fig.add_vline(x=data["day"]["vwp"], line=dict(color="grey", dash="dash"), row=1, col=c)
        fig.add_vline(x=data["day"]["last_breeding_days"], line=dict(color="grey", dash="dot"), row=1, col=c)
        fig.update_xaxes(title_text="days in milk", range=[0, 600], row=1, col=c)
    fig.update_yaxes(title_text="US$ per day, open cows (income positive, costs negative)", row=1, col=1)
    add(fig, "P13_rewards_per_day", "Rewards per day, open cows, lactations 1-3",
        "Expected reward per day for a cow in the open state, per cow at the START of the step. Costs are negative. Grey dashed = end of the waiting period (50 d), grey dotted = last breeding day (300 d). Milk income and feed cost agree across resolutions. "
        "Reproduction cost is lower at coarser steps (about 0.78 x DAY at MONTH) by design: a cow that conceives part-way through a step stops being charged for breeding, and the expected charge per cow at the start of a long step therefore falls "
        "(checked: (1+P(stay open))/2 is about 0.78 at MONTH and about 1 at DAY). Feed and milk are paid by the day and also by the pregnancy status; there is no separate double-counting of milk revenue and breeding cost "
        "(the days-open penalty that would have done so is switched off).")


def fig_summary(data):
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Value of a replacement heifer, US$", "Cycle length, days", "Calvings per cycle", "Number of states (log axis)"))
    for c, (k, f) in enumerate((("V", lambda v: f"{v:,.0f}"), ("cycle_days", lambda v: f"{v:,.0f}"), ("calvings_per_cycle", lambda v: f"{v:.2f}"), ("n_states", lambda v: f"{v:,.0f}"))):
        vals = [data[u]["summary"][k] for u in UNITS]
        fig.add_trace(go.Bar(x=[u.upper() for u in UNITS], y=vals, text=[f(v) for v in vals], textposition="outside", marker_color=[RCOL[u] for u in UNITS], showlegend=False), row=1, col=c + 1)
    fig.update_yaxes(range=[0, 36000], row=1, col=1); fig.update_yaxes(range=[0, 1400], row=1, col=2); fig.update_yaxes(range=[0, 2.6], row=1, col=3); fig.update_yaxes(type="log", range=[2.5, 6.3], row=1, col=4)
    add(fig, "P14_resolution_summary", "Whole-model results at the three resolutions",
        "One replacement cycle = a heifer entering the herd at the start of lactation 1 until she is replaced (death, culling or forced exit after lactation 10). 'Calvings per cycle' counts every calving AFTER that entry "
        "(the heifer's own first calving is not counted), so lactations started per cycle = calvings + 1. V is the discounted (5 % per year) value of a replacement heifer; MONTH is about 8 % lower than DAY because of the event-booking effect "
        "described in P08 and docs section 21, so WEEK or DAY is recommended for quantitative work.")


# =========================================================================================== write
def main():
    os.makedirs(OUT, exist_ok=True)
    data = collect_all()
    fig_provenance(data); fig_milk(data); fig_pregfactor(data)
    fig_hazard(data, "mortality", ("died",), "P03_mortality", "Mortality hazard by lactation, state and clock",
               "Hazard of death, per 1000 cow-days, for lactations 1-10 (colour) from the lookup table. OPEN cows use days in milk; PREGNANT cows days pregnant until dry-off (222 d); DRY cows days pregnant from dry-off to calving. "
               "Open circles = Pinedo et al. (2010) values read from their graphs (parity 1, 2, 3, >=4 plotted on lactations 1-4). Lactations 5-10 rest on the '>=4' evidence scaled by the death-per-lactation ratios of Miller et al. (2008). "
               "Legend: the evidence class of each lactation (direct = source reports that parity; grouped = expanded from the '>=4' group with Miller's lactation ratio; repeated = no distinct estimate). Hazards rise with lactation, are highest just after calving, and are about three times lower in pregnant than in open cows (as Pinedo reports). "
               "CAVEAT: Pinedo's pregnant-cow curves describe cows still in a lactation record; the DRY panel repeats them for dry cows (labelled 'repeated' in the table): no hazard was measured for dry cows. Known feature of the transcription: lactation 4 (grouped, scaled by Miller) lies below lactation 3 in places (docs section 29). The three panels share one y-axis.")
    fig_resolution_check(data, "mortality", "P03b_mortality_resolution_check", "Mortality: table vs model at each resolution (lactation 2)")
    fig_hazard(data, "culling", PIN.OTHER_CULL_CODES, "P04_other_reason_culling", "Other-reason culling hazard by lactation, state and clock",
               "Hazard of being culled for feet and legs, injury, mastitis, disease or udder problems (Pinedo et al. 2010; the sum of the five codes), per 1000 cow-days, lactations 1-10. Lactations 1-4 follow the four parity groups of the paper; "
               "lactations 5-10 repeat the '>=4' curve (dotted, no evidence they differ), so they lie on one line. Culling for low production or reproduction is NOT here: it is the milk-floor rule (P09). The Dry panel repeats Pinedo's late-pregnancy "
               "values for cows that are no longer milked; whether real dry-period culling is lower is not known from the reviewed papers (extrapolation, labelled repeated). Same axes as P03 for comparison.")
    fig_resolution_check(data, "culling", "P04b_culling_resolution_check", "Other-reason culling: table vs model at each resolution (lactation 2)")
    fig_exits(data); fig_conception(data); fig_loss(data); fig_calving(data); fig_floor(data); fig_herd(data); fig_survival(data); fig_values(data); fig_rewards(data); fig_summary(data)
    # write files
    for name, title, caption, fig in FIGS:
        fig.write_html(os.path.join(OUT, name + ".html"), include_plotlyjs="directory", full_html=True)
    with open(os.path.join(OUT, "plotly.min.js"), "w", encoding="utf-8") as f:
        f.write(po.get_plotlyjs())
    body = ["<html><head><meta charset='utf-8'><title>CowMRP validation (all figures)</title><script src='plotly.min.js'></script><style>body{font-family:sans-serif;max-width:1500px;margin:auto}p.cap{max-width:1300px;color:#333}</style></head><body><h1>CowMRP validation figures</h1>"]
    idx = ["<html><head><meta charset='utf-8'><title>CowMRP validation index</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto}li{margin:10px 0}</style></head><body><h1>CowMRP validation figures (interactive)</h1>",
           "<p>Generated by <code>runners/cow/plots/plot_validation_plotly.py</code> from the model's actual tables and solutions (<code>runners/cow/plots/_plot_data.py</code>). <b>Colour = lactation (1-10, purple to yellow); line style = resolution (solid day, dashed week, dotted+markers month).</b> "
           "U = length of one model step: 30 days (MONTH), 7 days (WEEK), 1 day (DAY). Evidence labels: source = exact printed value; derived = adapted from sources; repeated = grouped value copied to lactations the paper does not separate; placeholder = no evidence. "
           "Lookup tables for all lactations 1-10: <code>runners/cow/outputs/lookup_tables/lactation_1_to_10_overview.html</code>. <a href='validation_all.html'>All figures on one page</a>.</p><ul>"]
    for name, title, caption, fig in FIGS:
        body.append(f"<h2 id='{name}'>{html.escape(title)}</h2>" + fig.to_html(full_html=False, include_plotlyjs=False, div_id=name + "_div") + f"<p class='cap'>{html.escape(caption)}</p>")
        idx.append(f"<li><a href='{name}.html'><b>{html.escape(title)}</b></a><br><span style='color:#444'>{html.escape(caption)}</span></li>")
    body.append("</body></html>"); idx.append("</ul></body></html>")
    open(os.path.join(OUT, "validation_all.html"), "w", encoding="utf-8").write("\n".join(body))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(idx))
    if "--png" in sys.argv:       # optional static copies for quick looks (needs the kaleido package; not a project dependency)
        png = sys.argv[sys.argv.index("--png") + 1]; os.makedirs(png, exist_ok=True)
        for name, title, caption, fig in FIGS:
            fig.write_image(os.path.join(png, name + ".png"), width=1500, height=560, scale=1)
    print(f"wrote {len(FIGS)} figures to {OUT}")


if __name__ == "__main__":
    main()

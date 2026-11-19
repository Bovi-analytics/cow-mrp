"""guide/web/performance_page.py -- the "Computational performance and scalability" page.

Single source of every number on the page: content/performance_data.json (measured values of the saved benchmark runs: medians over the repeats, min / max,
CPU, peak memory, output size, model size and the environment record). The tables, the sentences that quote numbers and the four figures are all generated
from that file when the site is built; nothing is typed into the page text. content/performance.html holds the hand-written explanation.

    body()                  the HTML of the page (used by guide/web/build_web.py and, in its own layout, by guide/build.py)
    make_figures(outdir)    the four PNG figures, drawn with the site's plotting style

The build stops if the model sizes in the data file differ from the model the package builds now (the data would describe another model).
Nothing here imports or changes the model; it only reads the data file (and, for the check, the facts that guide/sync.py reads from the package).
"""
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(HERE, "content")
DATA = os.path.join(CONTENT, "performance_data.json")

UNITS = ("month", "week", "day")
SIZES = (1, 1000, 10000)
NAMES = {"month": "Monthly", "week": "Weekly", "day": "Daily"}
STEP = {"month": "30-day step", "week": "7-day step", "day": "1-day step"}
COLOR = {"month": "#d99a1e", "week": "#2a6a4c", "day": "#3b6ea5"}          # the colours of the time-resolution figure
FIXED = ["states", "probabilities", "rewards", "transition_matrix", "row_check"]
HERD = ["prepare_herd", "map_to_states", "future_distribution", "state_values", "keep_cull", "save_results"]
LABEL = {"states": "Build states", "probabilities": "Probabilities", "rewards": "Reward vector (all states)", "transition_matrix": "Transition matrix",
         "row_check": "Row-sum check", "prepare_herd": "Prepare herd", "map_to_states": "Map cows to states", "future_distribution": "Future state distribution",
         "state_values": "State values / rewards", "keep_cull": "KEEP / CULL", "save_results": "Save results"}
# the stage groups of the stacked figure: (label, stages, colour)
GROUPS = [("States and probabilities", ["states", "probabilities"], "#b7cde3"), ("Transition matrix (and row-sum check)", ["transition_matrix", "row_check"], "#3b6ea5"),
          ("Reward vector", ["rewards"], "#6f9fc7"), ("Future state distribution", ["future_distribution"], "#2a6a4c"),
          ("State values / rewards", ["state_values"], "#7a5aa6"), ("KEEP / CULL", ["keep_cull"], "#d99a1e"),
          ("Herd preparation, mapping and saving", ["prepare_herd", "map_to_states", "save_results"], "#8a949c")]


def load():
    with open(DATA, encoding="utf8") as fh:
        return json.load(fh)


def by_key(d):
    return {(s["unit"], s["n_target"]): s for s in d["scenarios"]}


# ------------------------------------------------------------------------------------------------------------------ formatting
def fT(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 0.01:
        return f"{s * 1e3:.2f} ms"
    if s < 0.1:
        return f"{s * 1e3:.1f} ms"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return f"{s:.2f} s"
    return f"{int(s // 60)} min {round(s % 60)} s"


def fTf(s):
    """fT for figure labels: the serif font of the figures has no micro sign, so it is written as mathtext."""
    return fT(s).replace("\u00b5", r"$\mu$")


def fN(n):
    return f"{round(n):,}"


def fB(b):
    return f"{b / 1e3:.1f} KB" if b < 1e6 else f"{b / 1e6:.2f} MB"


def fP(p):
    sign = "+" if p > 0 else "−" if p < 0 else ""
    return f"{sign}{abs(p):.1f} %" if abs(p) >= 99 else f"{sign}{abs(p):.0f} %"


def cows(n):
    return f"{n:,} cow" + ("" if n == 1 else "s")


def herd_t(s):
    return sum(s["stages"][k] for k in HERD)


def fixed_t(s):
    return sum(s["stages"][k] for k in FIXED)


def table(head, rows, small=False):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    style = ' style="font-size:.84rem"' if small else ""
    return f'<div style="overflow-x:auto"><table{style}><tr>{th}</tr>{body}</table></div>'


# ------------------------------------------------------------------------------------------------------------------ guard
def check_against_package(d, facts):
    """Stop if the model sizes of the measured runs differ from the model the package builds now."""
    want = {"month": (facts["COW_STATES"], facts["COW_NNZ"]), "week": (facts["WEEK_STATES"], facts["WEEK_NNZ"]), "day": (facts["DAY_STATES"], facts["DAY_NNZ"])}
    for s in d["scenarios"]:
        got = (f"{s['num_states']:,}", f"{s['nnz']:,}")
        if got != tuple(want[s["unit"]]):
            sys.exit(f"performance_data.json was measured on a model with {got[0]} states / {got[1]} transitions at {s['unit']} resolution, "
                     f"the package builds {want[s['unit']][0]} / {want[s['unit']][1]} now: the benchmark must be re-run and the data re-exported")


# ------------------------------------------------------------------------------------------------------------------ tables
def env_table(d):
    e = d["environment"]
    rows = [("Operating system", e["platform"]), ("Processor", f"{e['processor']} ({e['logical_cores']} logical cores)"), ("Memory", f"{e['ram_gb']} GB installed"),
            ("Python", e["python"]), ("Package", e["package_source"]), ("CPU and memory measured with", f"psutil {e['psutil']}"),
            ("Measured on", d["measured_on"])]
    return table(["Item", "Recorded value"], [[a, html.escape(b)] for a, b in rows])


def summary_table(S):
    rows = []
    for u in UNITS:
        for n in SIZES:
            s = S[(u, n)]
            e = s["e2e"]
            herd = "original record" if n == 1 else (f"{fN(s['n_original'])} original" + (f" + {fN(s['n_copies'])} copies" if s["n_copies"] else ""))
            runs = f"median of {s['runs']}" if s["runs"] > 1 else "single run"
            span = f'<br><span class="mut">{fT(e["min"])} – {fT(e["max"])}</span>' if s["runs"] > 1 else ""
            rows.append([f"{NAMES[u]}" + ("*" if s["fallback"] else ""), f'{fN(s["n_cows"])}<br><span class="mut">{herd}</span>', runs,
                         f"<b>{fT(e['wall'])}</b>{span}", fT(e["wall"] / s["n_cows"]), f"{fN(e['peak_mb'])}&nbsp;MB", f"{e['cpu_pct']:.0f}&nbsp;%", fB(s["output_bytes"])])
    return table(["Resolution", "Cows", "Runs", "End-to-end time (range)", "Per cow (amortised)", "Peak memory", "CPU", "Results written"], rows, small=True)


def stage_table(S):
    rows = []
    for grp, names in (("Fixed: model construction (does not depend on the herd)", FIXED), ("Herd-dependent", HERD)):
        rows.append([f'<b>{grp}</b>', "", "", ""])
        for k in names:
            cells = []
            for u in UNITS:
                s = S[(u, 1000)]
                v = fT(s["stages"][k]) + (" *" if (u == "day" and k == "future_distribution") else "")
                cells.append(v)
            rows.append([LABEL[k], *cells])
    tot = [fT(S[(u, 1000)]["e2e"]["wall"]) for u in UNITS]
    rows.append(["<b>Measured end to end</b>", *[f"<b>{t}</b>" for t in tot]])
    return table(["Stage (1,000 cows)", "Monthly", "Weekly", "Daily"], rows)


def percow_table(S):
    rows = []
    for u in UNITS:
        t = [herd_t(S[(u, n)]) for n in SIZES]
        pc = [t[i] / S[(u, n)]["n_cows"] for i, n in enumerate(SIZES)]
        rows.append([NAMES[u] + ("*" if S[(u, 1000)]["fallback"] else ""), *[f'{fT(t[i])}<br><span class="mut">{fT(pc[i])} per cow</span>' for i in range(3)],
                     fP(100 * (pc[1] / pc[0] - 1)), fP(100 * (pc[2] / pc[1] - 1))])
    return table(["Herd-dependent time", "1 cow", "1,000 cows", "10,000 cows", "Per cow, 1 → 1,000", "Per cow, 1,000 → 10,000"], rows, small=True)


def model_table(S):
    rows = []
    for u in UNITS:
        s = S[(u, 1000)]
        fx = " / ".join(fT(fixed_t(S[(u, n)])) for n in SIZES)
        rows.append([f'{NAMES[u]}<br><span class="mut">{STEP[u]}</span>', fN(s["num_states"]), fN(s["nnz"]), fx,
                     f"{fN(S[(u, 1)]['e2e']['peak_mb'])} – {fN(S[(u, 10000)]['e2e']['peak_mb'])}&nbsp;MB"])
    return table(["Resolution", "States", "Transitions", "Fixed model build (1 / 1,000 / 10,000 cows)", "Peak memory (1 to 10,000 cows)"], rows, small=True)


# ------------------------------------------------------------------------------------------------------------------ sentences quoting numbers
def sentences(d, S):
    m1, m10 = S[("month", 1)], S[("month", 10000)]
    t = {}
    ratio = lambda u: herd_t(S[(u, 10000)]) / herd_t(S[(u, 1)])
    pcm = lambda u, n: herd_t(S[(u, n)]) / S[(u, n)]["n_cows"]
    t["PF_RUNTIME_TEXT"] = (
        f"At monthly resolution the whole workflow takes {fT(S[('month', 1000)]['e2e']['wall'])} for 1,000 cows, at weekly resolution {fT(S[('week', 1000)]['e2e']['wall'])} "
        f"and at daily resolution {fT(S[('day', 1000)]['e2e']['wall'])}. Within a resolution, going from 1 cow to 10,000 cows changes the end-to-end time only a little: "
        f"monthly {fT(m1['e2e']['wall'])} → {fT(m10['e2e']['wall'])}, weekly {fT(S[('week', 1)]['e2e']['wall'])} → {fT(S[('week', 10000)]['e2e']['wall'])}, "
        f"daily {fT(S[('day', 1)]['e2e']['wall'])} → {fT(S[('day', 10000)]['e2e']['wall'])}. Over this range of herd sizes (1 to 10,000 cows) the resolution, which sets the number of states, mattered far more than the herd size.")
    s1, s2, s3 = S[("month", 1000)], S[("week", 1000)], S[("day", 1000)]
    share = lambda s, ks: 100 * sum(s["stages"][k] for k in ks) / sum(s["stages"].values())
    t["PF_STAGES_TEXT"] = (
        f"Building the model is a fixed cost: {fT(fixed_t(s1))} monthly, {fT(fixed_t(s2))} weekly and {fT(fixed_t(s3))} daily (for 1,000 cows), "
        f"that is {share(s1, FIXED):.0f} %, {share(s2, FIXED):.0f} % and {share(s3, FIXED):.0f} % of the sum of the stage times. At monthly resolution the probabilities are the "
        f"largest single stage ({fT(s1['stages']['probabilities'])}); at weekly resolution the future distribution ({fT(s2['stages']['future_distribution'])}); "
        f"at daily resolution KEEP / CULL ({fT(s3['stages']['keep_cull'])}, {share(s3, ['keep_cull']):.0f} %), followed by the state values ({fT(s3['stages']['state_values'])}). "
        f"The reward vector takes {fT(s1['stages']['rewards'])}, {fT(s2['stages']['rewards'])} and {fT(s3['stages']['rewards'])}, and the transition matrix "
        f"{fT(s1['stages']['transition_matrix'])}, {fT(s2['stages']['transition_matrix'])} and {fT(s3['stages']['transition_matrix'])}.")
    t["PF_PERCOW_TEXT"] = (
        f"From 1 cow to 10,000 cows the herd-dependent time grows by a factor of {ratio('month'):.1f} (monthly), {ratio('week'):.1f} (weekly) and {ratio('day'):.1f} (daily), "
        f"while the herd grows 10,000-fold. The amortised herd-dependent time per cow therefore falls from {fT(pcm('month', 1))} to {fT(pcm('month', 10000))} at monthly resolution, "
        f"from {fT(pcm('week', 1))} to {fT(pcm('week', 10000))} weekly and from {fT(pcm('day', 1))} to {fT(pcm('day', 10000))} daily. "
        f"The part that does grow with the herd is writing the per-cow results: saving takes {fT(m1['stages']['save_results'])} for 1 cow and {fT(m10['stages']['save_results'])} for 10,000 cows "
        f"at monthly resolution ({fB(m1['output_bytes'])} → {fB(m10['output_bytes'])}), and mapping the cows to states grows from {fT(m1['stages']['map_to_states'])} to {fT(m10['stages']['map_to_states'])}.")
    lo = min(S[(u, n)]["e2e"]["cpu_pct"] for u in UNITS for n in SIZES)
    hi = max(S[(u, n)]["e2e"]["cpu_pct"] for u in UNITS for n in SIZES)
    pk = lambda u: (min(S[(u, n)]["e2e"]["peak_mb"] for n in SIZES), max(S[(u, n)]["e2e"]["peak_mb"] for n in SIZES))
    t["PF_MEMORY_TEXT"] = (
        f"Over this range of herd sizes, peak memory was determined mainly by the resolution: {fN(pk('month')[0])} – {fN(pk('month')[1])} MB monthly, {fN(pk('week')[0])} – {fN(pk('week')[1])} MB weekly and "
        f"{fN(pk('day')[0])} – {fN(pk('day')[1])} MB daily, for 1 to 10,000 cows. CPU use was between {lo:.0f} % and {hi:.0f} % of one core in every run: the workflow is "
        f"single-threaded, so the other logical cores of the machine were not used.")
    dl = d["daily_limitation"]
    needed = dl["vectors_kept"] * dl["states"] * 8 / 1e9
    t["PF_DAILY_TEXT"] = (
        f"At daily resolution the model has {fN(s3['num_states'])} states and {fN(s3['nnz'])} transitions. The standard projection function, <code>project_herd</code>, stores the "
        f"distribution after every one of the {dl['steps'] + 1} steps (three lists: the original herd, the replaced animals and the total), that is {fN(dl['vectors_kept'])} vectors of "
        f"{fN(dl['states'])} numbers. At 8 bytes per number this is about {needed:.0f} GB, more than the {dl['available_gb_recorded']} GB that were free on the benchmark machine "
        f"({d['environment']['ram_gb']} GB installed). This figure is arithmetic from the model size, not a measurement; the benchmark recorded that the full trajectory was not attempted for this reason, "
        f"and earlier trial runs of the same case ran out of memory (those trial runs are not part of the saved results).")
    t["PF_FALLBACK_TEXT"] = (
        f"The daily future distribution was therefore computed with <code>replacement_split(…, keep_trajectory=False)</code>, which applies exactly the same step "
        f"<code>π ← π P</code> {dl['steps']} times but keeps only the final distribution. It was checked against <code>project_herd</code> wherever <code>project_herd</code> fits in memory: "
        f"monthly (24 steps), weekly (104 steps) and daily with 30 and 120 steps. In all four cases the expected number of cows by reproductive status and by lactation, and the split into animals "
        f"that were never replaced and animals that entered through the replacement state, were identical (largest absolute difference {d['checks']['fallback_vs_project_herd'][0]['max_abs_difference']:.0f}). "
        f"For the full {dl['steps']} daily steps, which <code>project_herd</code> cannot run on this machine, the final distribution was compared with an independent step-by-step calculation "
        f"(<code>π ← π P</code> applied directly to the transition matrix): the largest absolute difference over all states was {d['checks']['fallback_vs_independent_loop_day_730']['max_abs_difference']:.1e}, "
        f"which is floating-point rounding, and the total probability was {d['checks']['fallback_vs_independent_loop_day_730']['mass']:.0f}. "
        f"What remains unverified is only the literal run of <code>project_herd</code> at {dl['steps']} daily steps. The fallback does not store or group the intermediate steps, so its time is expected to "
        f"be somewhat lower than that of <code>project_herd</code> at the same resolution; the daily future-distribution time (marked *) should be read as a lower bound for that function.")
    h = d["herd_construction"]
    t["PF_HERD_TEXT"] = (
        f"The example herd file has {fN(h['file_cows'])} cows. At weekly and daily resolution {h['no_state_weekly_daily']} of them have no state in the model and are left out "
        f"(no state is invented and no record is moved to a neighbouring state): " + "; ".join(f"{r['cows']} {r['text']}" for r in h["reasons"]) + ". "
        f"At monthly resolution the coarser steps give all {fN(h['file_cows'])} records a state, so the monthly herds contain these cows and the weekly and daily herds do not. "
        f"The remaining {fN(h['distinct_valid'])} valid records are used as they are. A herd of 1 cow is the single record closest to the file's median "
        f"(lactation, days in milk, days pregnant). Larger herds are built by repeating the valid records in file order with new cow identifiers: 1,000 cows are the file itself at monthly resolution "
        f"and {fN(h['distinct_valid'])} original records plus {1000 - h['distinct_valid']} copies at weekly and daily resolution; 10,000 cows are the file repeated ten times at monthly resolution "
        f"and {fN(h['distinct_valid'])} original records plus {fN(10000 - h['distinct_valid'])} copies at weekly and daily resolution. The copies are identical to their originals. "
        f"This leaves the herd's composition almost unchanged (the largest change in the share of any lactation is {h['lactation_share_shift_pp']['1000']} percentage points at 1,000 cows and "
        f"{h['lactation_share_shift_pp']['10000']} at 10,000), but it means that a replicated herd is not 10,000 different cows. The “Herd composition” column of the summary table shows the split for every run.")
    lo_ov = min(S[k]["overhead_s"]["min"] for k in S)
    hi_ov = max(S[k]["overhead_s"]["max"] for k in S)
    t["PF_OVERHEAD_TEXT"] = f"between {lo_ov * 1e3:.0f} and {hi_ov * 1e3:.0f} ms in every run"
    t["PF_REPS_TEXT"] = (f"Monthly and weekly scenarios were run {S[('month', 1)]['runs']} times each and the table reports the median (the range is shown under the median). "
                         f"Daily scenarios were run once each, because one daily run takes about {fT(S[('day', 1000)]['e2e']['wall'])}: no spread is known for them. "
                         f"The three daily scenarios, which do almost the same work, differ from each other by {100 * (max(S[('day', n)]['e2e']['wall'] for n in SIZES) / min(S[('day', n)]['e2e']['wall'] for n in SIZES) - 1):.0f} %, "
                         f"which gives an idea of the run-to-run variation.")
    t["PF_KC_YEARS"] = ", ".join(str(y) for y in d["workflow"]["keepcull_years"])
    t["PF_HORIZON"] = str(d["workflow"]["horizon_months"])
    t["PF_STEPS"] = " / ".join(str(S[(u, 1)]["steps"]) for u in UNITS)
    t["PF_REPS_MS"] = str(S[("month", 1)]["runs"])
    return t


# ------------------------------------------------------------------------------------------------------------------ the page
def body(read, facts=None):
    """HTML of the page. ``read``: function that returns a text file's content (the site builder's own reader); ``facts``: guide/sync.facts() for the model-size guard."""
    d = load()
    if facts is not None:
        check_against_package(d, facts)
    S = by_key(d)
    tokens = {"PF_ENV_TABLE": env_table(d), "PF_SUMMARY_TABLE": summary_table(S), "PF_STAGE_TABLE": stage_table(S), "PF_PERCOW_TABLE": percow_table(S),
              "PF_MODEL_TABLE": model_table(S), **sentences(d, S)}
    text = read(os.path.join(CONTENT, "performance.html"))
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(PF_\w+)\}\}", text)
    if left:
        sys.exit(f"performance page: unfilled placeholders {left}")
    return text


# ------------------------------------------------------------------------------------------------------------------ figures
def make_figures(outdir):
    """Four figures from performance_data.json (same plotting style as the other figures of the site)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    d = load()
    S = by_key(d)
    style = {"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
             "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e9ec", "grid.linewidth": .8, "axes.axisbelow": True,
             "figure.dpi": 100, "savefig.dpi": 160}
    os.makedirs(outdir, exist_ok=True)
    sizes_lab = ["1 cow", "1,000 cows", "10,000 cows"]
    with plt.rc_context(style):
        # 1. end-to-end time by herd size and resolution
        fig, ax = plt.subplots(figsize=(7.6, 4))
        w = 0.26
        for j, u in enumerate(UNITS):
            y = np.array([S[(u, n)]["e2e"]["wall"] for n in SIZES])
            lo = np.array([S[(u, n)]["e2e"]["wall"] - S[(u, n)]["e2e"]["min"] for n in SIZES])
            hi = np.array([S[(u, n)]["e2e"]["max"] - S[(u, n)]["e2e"]["wall"] for n in SIZES])
            x = np.arange(3) + (j - 1) * w
            ax.bar(x, y, w * .92, color=COLOR[u], yerr=[lo, hi], error_kw={"lw": 1, "capsize": 2, "ecolor": "#333"},
                   label=f"{NAMES[u]} ({'median of ' + str(S[(u, 1)]['runs']) if S[(u, 1)]['runs'] > 1 else 'single run'})")
            for xi, yi, hh in zip(x, y, hi):
                ax.text(xi, (yi + hh) * 1.25, fTf(yi), ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.set_yscale("log"); ax.set_ylim(0.1, 6000); ax.set_xticks(range(3)); ax.set_xticklabels(sizes_lab)
        ax.set_ylabel("end-to-end time (seconds, log scale)"); ax.set_title("Complete workflow, 24-month projection")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_perf_runtime.png")); plt.close(fig)

        # 2. share of the stage time at 1,000 cows
        fig, ax = plt.subplots(figsize=(7.8, 3.4))
        for i, u in enumerate(UNITS):
            s = S[(u, 1000)]
            tot = sum(s["stages"].values())
            left = 0.0
            for lab, ks, col in GROUPS:
                v = 100 * sum(s["stages"][k] for k in ks) / tot
                hatch = "///" if (u == "day" and lab == "Future state distribution") else None
                ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=.6, hatch=hatch, label=lab if i == 0 else None)
                if v >= 5:
                    ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white" if col not in ("#b7cde3",) else "#222")
                left += v
        ax.set_yticks(range(3)); ax.set_yticklabels([f"{NAMES[u]}\n{fT(S[(u, 1000)]['e2e']['wall'])} end to end" for u in UNITS]); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel("share of the sum of the stage times (%), 1,000 cows"); ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.28))
        ax.set_title("Where the time goes (hatched: daily future distribution, computed without storing the trajectory)", fontsize=9.5)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_perf_stages.png")); plt.close(fig)

        # 3. herd-dependent time: total and per cow
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
        for u in UNITS:
            tot = [herd_t(S[(u, n)]) for n in SIZES]
            pc = [tot[i] / S[(u, n)]["n_cows"] for i, n in enumerate(SIZES)]
            axes[0].plot(SIZES, tot, "o-", color=COLOR[u], lw=2, label=NAMES[u])
            axes[1].plot(SIZES, pc, "o-", color=COLOR[u], lw=2, label=NAMES[u])
            for i, n in enumerate(SIZES):
                axes[0].annotate(fTf(tot[i]), (n, tot[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
                axes[1].annotate(fTf(pc[i]), (n, pc[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        for ax, t, yl in ((axes[0], "Herd-dependent time (total)", "seconds, log scale"), (axes[1], "Herd-dependent time per cow (total / cows)", "seconds per cow, log scale")):
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.35, 3e4); ax.set_xticks(SIZES); ax.set_xticklabels(["1", "1,000", "10,000"]); ax.set_xlabel("cows in the herd")
            ax.set_title(t, fontsize=10.5); ax.set_ylabel(yl); ax.margins(y=.25)
        axes[0].legend(frameon=False, fontsize=8.5)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_perf_per_cow.png")); plt.close(fig)

        # 4. peak memory
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        for j, u in enumerate(UNITS):
            y = np.array([S[(u, n)]["e2e"]["peak_mb"] for n in SIZES])
            x = np.arange(3) + (j - 1) * w
            ax.bar(x, y, w * .92, color=COLOR[u], label=f"{NAMES[u]} ({fN(S[(u, 1)]['num_states'])} states)")
            for xi, yi in zip(x, y):
                ax.text(xi, yi * 1.1, f"{fN(yi)}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log"); ax.set_ylim(50, 60000); ax.set_xticks(range(3)); ax.set_xticklabels(sizes_lab)
        ax.set_ylabel("peak resident memory (MB, log scale)"); ax.set_title("Peak memory of the complete workflow"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_perf_memory.png")); plt.close(fig)
    return ["fig_perf_runtime.png", "fig_perf_stages.png", "fig_perf_per_cow.png", "fig_perf_memory.png"]

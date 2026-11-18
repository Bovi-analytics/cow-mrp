"""
plot_validation.py
==================
Validation figures drawn from the model's ACTUAL arrays and solutions at MONTH, WEEK and DAY resolution
(one subprocess per resolution; nothing here re-implements model logic). Every biological dimension has its own
axis -- days in milk, gestation days, lactation number and time since replacement are never mixed on one axis.

    python runners/cow/plots/plot_validation.py            # -> runners/cow/outputs/validation/*.png + index.html

Figures
  F01 milk curves by lactation (DIM axis): continuous curve + the model's step-mean values at 3 resolutions; floor line
  F02 pregnancy milk factor (gestation-day axis) with the dry-off day
  F03 mortality and age/stage culling, per-day hazard vs days in milk, Open / Pregnant / Dry (table vs resolutions)
  F04 conception: per-day hazard and cumulative probability vs days in milk (waiting period and last breeding day marked)
  F05 pregnancy loss: per-day hazard and cumulative loss vs gestation days
  F06 calving: probability mass vs gestation days at 3 resolutions (mean = gestation length)
  F07 milk-floor reproductive cull: cull probability vs days in milk, per lactation
  F08 steady-state herd: share of cow-time by days in milk and by lactation, per resolution
  F09 survival of a replacement cohort vs years since entry, per resolution
  F10 pregnancy value vs gestation age, replacement value vs days in milk, per resolution
  F11 reward per day vs days in milk (milk income, feed cost, reproduction cost), lactations 1-3
  F12 resolution summary (V, cycle, calvings, exits)
Evidence labels (source / derived / repeated / placeholder) are written into each caption in index.html.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners._shared import common
SNAPSHOT_LABEL = os.environ.get('COW_SNAPSHOT_LABEL', 'final_v2')   # audit snapshot used by F12 (runners/cow/checks/audit_metrics.py run <label>)
OUT = common.module_outputs("cow", "validation")
UNITS = ("month", "week", "day")


# ------------------------------------------------------------------ collect (per resolution, in its own process)
def collect():
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    import cowmrp.domains.cow.parameters as P
    from cowmrp.mrp import core as mrp_core
    from cowmrp.mrp.engine import resolve_method
    from cowmrp.domains.cow.states import build_cow_states, repro_status
    from cowmrp.model import build_model
    from cowmrp.domains.cow.rewards import compute_rewards
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
    from cowmrp.domains.cow.tables import get_tables
    from cowmrp.domains.cow.curves import curve_kg
    from cowmrp.domains.cow.value_comparisons import pregnancy_value, replacement_value
    import cowmrp.domains.cow.probabilities.mortality as M
    import cowmrp.domains.cow.probabilities.culling as C
    import cowmrp.domains.cow.probabilities.conception as K
    import cowmrp.domains.cow.probabilities.abortion as A
    import cowmrp.domains.cow.probabilities.calving as V
    from cowmrp.domains.cow.lookup_tables import get_full_table, table_to_array, N_DIM_MONTHS, N_GEST_MONTHS, STATUSES
    from cowmrp.domains.cow.rewards.milk_income import milk_income_formula
    from cowmrp.domains.cow.rewards.feed_cost import feed_cost_formula
    from cowmrp.domains.cow.rewards.reproduction_cost import reproduction_cost_formula

    U = P.UNIT_DAYS
    t = get_tables()
    d = {"unit": P.TIME_UNIT, "U": U, "MAX_DIM": P.MAX_DIM, "MAX_LACTATION": P.MAX_LACTATION}
    lacs = [0, 1, 2, 9]
    # two clocks: (open[lac, dim], pregnant[lac, dp], dry[lac, dp])
    d["mortality"] = [a[lacs, :].tolist() for a in M._mortality_step_arrays()]
    d["culling"] = [a[lacs, :].tolist() for a in C._culling_age_stage_step_arrays()]
    d["conception"] = K._conception_step_arrays()[lacs, :].tolist()
    d["cull_p"] = t["cull_p"].tolist()
    d["cull_day"] = t["cull_day"].tolist()
    d["milk_curve"] = t["milk_curve"].tolist()
    d["preg_factor"] = t["preg_factor"][0].tolist()
    d["dim_max"], d["dp_max"] = int(t["dim_max"]), int(t["dp_max"])
    d["monthly"] = {
        "mortality": table_to_array(get_full_table("mortality_full"), N_DIM_MONTHS, STATUSES)[:, lacs, :].tolist(),
        "culling": table_to_array(get_full_table("culling_age_stage_full"), N_DIM_MONTHS, STATUSES)[:, lacs, :].tolist(),
        "conception": table_to_array(get_full_table("conception_full"), N_DIM_MONTHS)[0][lacs, :].tolist(),
        "pregnancy_factor": table_to_array(get_full_table("pregnancy_milk_factor_full"), N_GEST_MONTHS, value_col="factor")[0][0].tolist(),
    }
    dpf = pd.DataFrame({"lactation": 1, "dim": 0, "dp": np.arange(0, t["dp_max"] + 1)})
    d["abortion"] = np.asarray(resolve_method(A.abortion_spec)(dpf, {})).tolist()
    d["calving_mass"] = {str(k): v for k, v in V.calving_step_mass().items()}
    d["dry_off_dp"], d["dry_off_dp_days"] = P.DRY_OFF_DP, P.DRY_OFF_DP_DAYS
    d["gestation_days"] = P.GESTATION_LENGTH_DAYS
    d["vwp"], d["last_breeding_days"], d["floor"] = P.VOLUNTARY_WAITING_PERIOD_DAYS, P.LAST_BREEDING_DIM_DAYS, P.MILK_FLOOR_KG
    ref_days = np.arange(1, 701) + 0.5
    d["curve_days"] = ref_days.tolist()
    d["curve_continuous"] = {m: [curve_kg(ref_days, l, method=m).tolist() for l in (1, 2, 3, 10)] for m in ("wood", "milkbot")}

    # full model
    model = build_model("cow")                      # the package build (cowmrp.model): states, probabilities, rewards, validated matrix
    states, probs, rewards, matrix = model.states, model.probs, model.rewards, model.P
    gamma = mrp_core.annual_to_per_step_gamma(0.95, U)
    Vv = mrp_core.evaluate_infinite_horizon(matrix, rewards["total"], gamma)
    rid = get_replacement_state_id(states)
    n = len(states)
    Q = matrix.tocsr().copy().tolil()
    Q[:, rid] = 0
    Q = Q.tocsr()
    e = np.zeros(n)
    e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)
    x = x / x.sum()
    dim_days = (states["dim"].values + 1) * U
    status = np.array(repro_status(states["dp"].values))
    lac = states["lactation"].values
    def density30(mask):
        """Share of cow-time per 30-day bin of days in milk. A step k covers true ages [(k+.5)U, (k+1.5)U) (mean start age
        (k+.5)U, one step long); its cow-time is spread uniformly over them, so bins are not aliased by the step length."""
        xs = np.bincount(states["dim"].values[mask], weights=x[mask], minlength=int(states["dim"].max()) + 1)
        dens = np.zeros(1200)
        for k, w in enumerate(xs):
            if w:
                a0 = int(round((k + 0.5) * U)); dens[a0:a0 + U] += w / U
        return [float(dens[b * 30:(b + 1) * 30].sum()) for b in range(0, 34)]
    d["occ"] = {"bin30": density30(np.ones(n, bool)),
                "bin30_by_status": {s: density30(status == s) for s in ("Open", "Pregnant", "Dry")},
                "by_lactation": [float(x[lac == l].sum()) for l in range(1, P.MAX_LACTATION + 1)]}
    # survival of a replacement cohort (still in its first cycle)
    pi = np.zeros(n)
    pi[rid] = 1.0
    horizon = int(round(9 * 365 / U))
    stride = max(1, horizon // 90)
    surv_t, surv = [], []
    for step in range(horizon + 1):
        if step % stride == 0:
            surv_t.append(step * U / 365.0)
            surv.append(float(pi.sum()))
        pi = pi @ Q
    d["cycle_survival"] = {"years": surv_t, "alive": surv}
    # values vs gestation age / DIM
    c0 = max(0, int(round((90 - U / 2) / U)))
    pv_x, pv_y = [], []
    for dp in range(1, t["dp_max"] + 1):
        try:
            pv_y.append(float(pregnancy_value(states, Vv, lactation=2, dim=c0 + dp, dp=dp)["value"]))
            pv_x.append((dp - 1) * U + U / 2)
        except ValueError:
            pass
    d["pv"] = {"gest_days": pv_x, "value": pv_y, "conceived_at_dim_days": c0 * U + U / 2}
    rv_x, rv_y = [], []
    for k in range(0, min(int(t["cull_dim"][1]), t["dim_max"])):
        rv_x.append((k + 1) * U)
        rv_y.append(float(replacement_value(states, Vv, lactation=2, dim=k, dp=0)["value"]))
    d["rv"] = {"dim_days": rv_x, "value": rv_y}
    # rewards per day for open cows, lactations 1-3
    d["reward_day"] = {}
    for l in (1, 2, 3):
        sel = (states["lactation"].values == l) & (states["dp"].values == 0)
        sub = states[sel]
        pr = {k: v[sel] for k, v in probs.items()}
        d["reward_day"][str(l)] = {
            "dim_days": ((sub["dim"].values + 1) * U).tolist(),
            "milk_income": (milk_income_formula(sub, pr) / U).tolist(),
            "feed_cost": (-feed_cost_formula(sub, pr) / U).tolist(),
            "reproduction_cost": (-reproduction_cost_formula(sub, pr) / U).tolist(),
        }
    print("RESULT" + json.dumps(d))


# ------------------------------------------------------------------ plotting
def per_day(p, U):
    import numpy as np
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    return 1 - (1 - p) ** (1.0 / U)


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    data = {}
    for unit in UNITS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--collect"], env=dict(os.environ, COW_TIME_UNIT=unit),
                           cwd=ROOT, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            print(p.stderr[-2500:]); sys.exit(1)
        data[unit] = json.loads(line[0][6:])
        print(f"collected {unit}")
    col = {"month": "#d95f02", "week": "#1b9e77", "day": "#7570b3"}
    captions = {}

    def save(fig, name, caption):
        path = os.path.join(OUT, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        captions[name] = caption
        print("wrote", name)

    d0 = data["day"]
    # F01 milk curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, method in zip(axes, ("wood", "milkbot")):
        for i, l in enumerate((1, 2, 3, 10)):
            ax.plot(d0["curve_days"], d0["curve_continuous"][method][i], lw=1.4, label=f"lactation {l}" + (" (= 3, repeated)" if l == 10 else ""))
        for unit in (UNITS if method == "wood" else ()):
            u = data[unit]["U"]
            mc = np.array(data[unit]["milk_curve"][1])
            x = (np.arange(len(mc)) + 1) * u
            ax.plot(x[x <= 700], mc[x <= 700], ".", ms=4 if unit == "day" else 6, color=col[unit], label=f"model step-mean, lactation 2 ({unit})", alpha=.6)
        ax.axhline(d0["floor"], color="k", ls=":", lw=1); ax.axvline(d0["last_breeding_days"], color="grey", ls="--", lw=1)
        ax.set_title(f"{method.capitalize()} curve (active)" if method == "wood" else "MilkBot curve (selectable)")
        ax.set_xlabel("days in milk"); ax.set_xlim(0, 700)
    axes[0].set_ylabel("milk, kg/day (open cow)"); axes[0].legend(fontsize=7)
    save(fig, "F01_milk_curves.png", "Milk curves by lactation vs days in milk. Dotted: milk floor 22.7 kg/d (farmer setting); dashed: last breeding day. "
         "Wood: a, time-to-peak and peak from Evangelista 2024 (non-US Holstein; lactations 1-3 'source', 4-10 'repeated'); b, c derived. "
         "MilkBot: Ehrlich (2011) Table 1, Holstein median (lactations 1-3 'source', 4-10 'repeated'). Dots: the model's step-mean milk, plotted at the mean age during the step ((k+1)U).")

    # F02 pregnancy factor
    fig, ax = plt.subplots(figsize=(6.4, 4))
    mp = np.array(d0["monthly"]["pregnancy_factor"])
    ax.step(np.arange(len(mp)) * 30 + 15, mp, where="mid", color="k", lw=1, label="native monthly table (placeholder)")
    for unit in UNITS:
        u = data[unit]["U"]
        pf = np.array(data[unit]["preg_factor"])
        ax.plot((np.arange(len(pf))) * u, pf, "o-", ms=3, color=col[unit], label=f"model factor by dp ({unit})")
    ax.axvline(d0["dry_off_dp_days"], color="grey", ls="--"); ax.set_xlim(0, 290)
    ax.set_xlabel("days pregnant (mean age during the step)"); ax.set_ylabel("multiplier on open-cow milk"); ax.legend(fontsize=7)
    save(fig, "F02_pregnancy_milk_factor.png", "Pregnancy milk factor vs gestation days. Placeholder: linear decline to 0.70 at dry-off (222 d), uncited. "
         "0 once dry. Steps are plotted at dp*U (mean gestation age during step dp).")

    # F03 mortality / culling (two clocks: Open on days in milk, Pregnant/Dry on days pregnant)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for r, (name, key) in enumerate((("mortality", "mortality"), ("other-reason culling", "culling"))):
        for c, st in enumerate(("Open", "Pregnant", "Dry")):
            ax = axes[r, c]
            mon = np.array(d0["monthly"][key])[c]
            for li, lac in enumerate((1, 2, 3, 10)):
                ax.step(np.arange(mon.shape[1]) * 30 + 15, per_day(mon[li], 30) * 1000, where="mid", color="k", alpha=.2 + .25 * li, lw=1,
                        label=f"native table, lactation {lac}" if (r == 0 and c == 0) else None)
            for unit in UNITS:
                u = data[unit]["U"]
                arr = np.array(data[unit][key][c])[0]
                x = (np.arange(arr.shape[0]) + 1) * u if c == 0 else np.arange(arr.shape[0]) * u
                ax.plot(x, per_day(arr, u) * 1000, color=col[unit], lw=1.2, label=unit if (r == 0 and c == 0) else None)
            ax.set_title(f"{name}, {st}", fontsize=9); ax.set_xlim(0, 500 if c == 0 else 300)
            ax.set_xlabel("days in milk" if c == 0 else "days pregnant")
            if c == 0: ax.set_ylabel("hazard, per 1000 cow-days")
    axes[0, 0].legend(fontsize=7)
    fig.subplots_adjust(hspace=0.4)
    save(fig, "F03_mortality_culling.png", "Per-day hazard. Open: days in milk; Pregnant and Dry: days of pregnancy. Black: native monthly table for lactations 1, 2, 3, 10 (darker = later). "
         "Colours: the model's per-step probability (lactation 1) as a per-day rate at each resolution. Basis: Pinedo et al. 2010 Figs 4-7 read from the graphs "
         "(class B); lactations >= 4 repeat the '>=4' curve for culling; mortality 4-10 scaled by Miller et al. 2008.")

    # F04 conception
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    mon = np.array(d0["monthly"]["conception"])
    for li, lac in enumerate((1, 2)):
        axes[0].step(np.arange(mon.shape[1]) * 30 + 15, per_day(mon[li], 30) * 1000, where="mid", color="k", alpha=.35 + .4 * li, lw=1,
                     label=f"native table, lactation {lac}" if True else None)
    for unit in UNITS:
        u = data[unit]["U"]
        arr = np.array(data[unit]["conception"])
        for li, ls in ((0, "-"), (1, "--")):
            axes[0].plot((np.arange(arr.shape[1]) + 1) * u, per_day(arr[li], u) * 1000, ls, color=col[unit], lw=1.1)
            cum = 1 - np.cumprod(1 - arr[li])
            axes[1].plot((np.arange(arr.shape[1]) + 1) * u + u / 2, cum, ls, color=col[unit], lw=1.2, label=f"{unit}, lactation {li + 1}" if li < 2 else None)   # cumulative AFTER step k: end-of-step mean age
    for ax in axes:
        ax.axvline(d0["vwp"], color="grey", ls=":"); ax.axvline(d0["last_breeding_days"], color="grey", ls="--"); ax.set_xlim(0, 450); ax.set_xlabel("days in milk")
    axes[0].set_ylabel("conception hazard, per 1000 open cow-days"); axes[0].set_title("per-day conception hazard"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("P(conceived by this DIM) if never lost"); axes[1].set_title("cumulative"); axes[1].legend(fontsize=6)
    save(fig, "F04_conception.png", "Conception vs days in milk; dotted: end of voluntary waiting period (50 d); dashed: last breeding day (300 d). "
         "Basis: PLACEHOLDER (consistent with CDCB days-open aggregate, not a direct estimate); lactation 1 vs 2+ buckets, repeated for 2-10.")

    # F05 pregnancy loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    mon_ab = None
    for unit in UNITS:
        u = data[unit]["U"]
        ab = np.array(data[unit]["abortion"])
        g = np.arange(len(ab)) * u
        axes[0].plot(g[1:], per_day(ab[1:], u) * 1000, color=col[unit], lw=1.2, label=unit)
        axes[1].plot(g[1:] + u / 2, 1 - np.cumprod(1 - ab[1:]), color=col[unit], lw=1.4, label=unit)   # cumulative AFTER step dp: end-of-step mean gestation age
    for ax in axes:
        ax.set_xlabel("days pregnant (dp * U)"); ax.set_xlim(0, 285)
    axes[0].set_ylabel("loss hazard, per 1000 pregnant cow-days"); axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("cumulative pregnancy loss"); axes[1].axhline(0.45, color="k", ls=":", lw=1)
    save(fig, "F05_pregnancy_loss.png", "Pregnancy loss vs gestation days (active: Albaaj 2022-derived day hazard, windows 19-90 d, pooled across parities; "
         "beyond 90 d legacy uncited rates). Dotted: ~45% cumulative loss to calving. Basis: derived (<=90 d) / placeholder (>90 d).")

    # F06 calving
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for unit in UNITS:
        u = data[unit]["U"]
        mass = {int(k): v for k, v in data[unit]["calving_mass"].items()}
        ax.bar([k * u + (-3 if unit == "month" else 0 if unit == "week" else 3) for k in mass], [v for v in mass.values()], width=2.4 if unit != "day" else 1.2,
               color=col[unit], label=f"{unit}: mean {sum(k * u * v for k, v in mass.items()):.1f} d")
    ax.axvline(d0["gestation_days"], color="k", ls=":"); ax.set_xlabel("days pregnant when the cow leaves the pregnant states (k*U)"); ax.set_ylabel("share of a conception cohort")
    ax.legend(fontsize=7); ax.set_xlim(250, 310)
    save(fig, "F06_calving_timing.png", "When pregnancies end. Dotted: gestation length 277 d (Basiel 2024, Holstein-sired). The model books the calving at the end of the step; "
         "the mean of k*U equals the gestation length at every resolution.")

    # F07 cull
    fig, ax = plt.subplots(figsize=(7, 4))
    for unit in UNITS:
        u = data[unit]["U"]
        cp = np.array(data[unit]["cull_p"])
        for li in (0, 1, 2):
            ax.plot((np.arange(cp.shape[1]) + 1) * u, cp[li], color=col[unit], ls=("-", "--", ":")[li], lw=1.2,
                    label=f"{unit}, lactation {li + 1}" if unit == "day" else None)
    for li, c in enumerate(d0["cull_day"][:3]):
        ax.axvline(c, color="k", lw=.6, ls=(0, (1, 2)))
    ax.set_xlim(250, 480); ax.set_xlabel("days in milk (mean age during the step)"); ax.set_ylabel("P(culled in this step | open)"); ax.legend(fontsize=7)
    save(fig, "F07_milk_floor_cull.png", "Milk-floor reproductive cull: probability that an OPEN cow is culled in a step, per lactation (3-10 identical = repeated). "
         "Vertical lines: continuous-time floor-crossing day. The cull is spread over the adjacent steps with the calving entry-offset rule.")

    # F08 steady-state herd
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for unit in UNITS:
        o = data[unit]["occ"]
        axes[0].plot(np.arange(34) * 30 + 15, o["bin30"], color=col[unit], label=unit)
        axes[2].plot(range(1, len(o["by_lactation"]) + 1), o["by_lactation"], "o-", color=col[unit], label=unit)
    st = data["week"]["occ"]["bin30_by_status"]
    for s, c in zip(("Open", "Pregnant", "Dry"), ("tab:blue", "tab:green", "tab:orange")):
        axes[1].plot(np.arange(34) * 30 + 15, st[s], color=c, label=f"{s} (week)")
    axes[0].set_xlabel("days in milk (30-d bins)"); axes[0].set_ylabel("share of cow-time"); axes[0].set_title("steady-state herd by DIM")
    axes[1].set_xlabel("days in milk (30-d bins)"); axes[1].set_title("by reproductive status (WEEK)")
    axes[2].set_xlabel("lactation number"); axes[2].set_title("by lactation")
    for ax in axes: ax.legend(fontsize=7)
    save(fig, "F08_steady_state_herd.png", "Steady-state (stationary, replacement-closed) herd: where cow-time is spent. Lactation 10 is a forced exit (calving there = replaced). "
         "Culling repeats the '>=4' evidence for lactations 5-10 and mortality uses Miller's ratios there, so the old-parity tail rests on grouped evidence.")

    # F09 cohort survival
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for unit in UNITS:
        cs = data[unit]["cycle_survival"]
        ax.plot(cs["years"], cs["alive"], color=col[unit], lw=1.4, label=unit)
    ax.set_xlabel("years since entering as a replacement"); ax.set_ylabel("share of the cohort not yet replaced"); ax.legend(fontsize=7)
    save(fig, "F09_cohort_survival.png", "Survival of a replacement cohort (probability a cow has not yet been replaced by death, culling or the lactation-10 exit).")

    # F10 values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for unit in UNITS:
        pv, rv = data[unit]["pv"], data[unit]["rv"]
        axes[0].plot(pv["gest_days"], pv["value"], "o-", ms=3, color=col[unit], label=f"{unit} (conceived at DIM ~{pv['conceived_at_dim_days']:.0f})")
        axes[1].plot(rv["dim_days"], rv["value"], color=col[unit], label=unit)
    axes[0].set_xlabel("days pregnant at the start of the step"); axes[0].set_ylabel("pregnancy value, $  (V pregnant - V open)"); axes[0].set_title("lactation 2")
    axes[1].set_xlabel("days in milk (open cow)"); axes[1].set_ylabel("replacement value, $  (V cow - V replacement)"); axes[1].set_title("lactation 2")
    for ax in axes: ax.legend(fontsize=7)
    save(fig, "F10_values.png", "Pregnancy value vs gestation age and replacement value vs days in milk, from the model's infinite-horizon solve (read-only diagnostics). "
         "Comparison is at matched real ages; coarse steps differ by an O(U) event-timing effect.")

    # F11 rewards per day
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ci, (key, title) in enumerate((("milk_income", "milk income"), ("feed_cost", "feed cost (as +)"), ("reproduction_cost", "reproduction cost (as +)"))):
        for unit in UNITS:
            rd = data[unit]["reward_day"]
            for l, ls in (("1", "-"), ("2", "--"), ("3", ":")):
                x = np.array(rd[l]["dim_days"]); y = np.array(rd[l][key])
                o = np.argsort(x)
                axes[ci].plot(x[o], y[o], ls, color=col[unit], lw=1.1, label=f"{unit} L{l}" if ci == 0 else None)
        axes[ci].set_title(f"{title}, $/cow-day (open cows)"); axes[ci].set_xlabel("days in milk (mean age during the step)")
    axes[0].legend(fontsize=6, ncol=2)
    save(fig, "F11_rewards_per_day.png", "Per-day reward of open cows vs days in milk (state rewards divided by the step length). "
         "Milk income: Wood curve x price (formula). Reproduction cost: breeding cost + days-open penalty (open-share corrected); the days-open penalty is a flagged double-counting risk.")

    # F12 summary
    snaps = common.module_outputs("cow", "audit", "snapshots")
    rows = []
    for unit in UNITS:
        f = os.path.join(snaps, f"{SNAPSHOT_LABEL}_{unit}.json")
        if os.path.exists(f):
            rows.append(json.load(open(f, encoding="utf-8")))
    if rows:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        names = [r["unit"] for r in rows]
        for ax, key, title in zip(axes, ("V_replacement", "cycle_days", "calvings_per_cycle", "num_states"),
                                  ("V(replacement), $", "cycle length, days", "calvings per cycle", "states")):
            ax.bar(names, [r[key] for r in rows], color=[col[n] for n in names]); ax.set_title(title)
            if key == "num_states": ax.set_yscale("log")
        save(fig, "F12_resolution_summary.png", "Whole-model results per time resolution (snapshot '" + SNAPSHOT_LABEL + "'). Differences between MONTH and WEEK/DAY are event-timing quantization.")

    html = ["<html><head><meta charset='utf-8'><title>CowMRP validation figures</title><style>body{font-family:Segoe UI,sans-serif;max-width:1100px;margin:auto;padding:1rem}"
            "figure{border:1px solid #ddd;padding:.5rem;margin:1.2rem 0}img{max-width:100%}figcaption{font-size:.9rem;color:#333}</style></head><body><h1>CowMRP validation figures</h1>"
            "<p>Generated from the model's actual arrays and solutions by <code>runners/cow/plots/plot_validation.py</code>. Evidence labels: source / derived / repeated / placeholder.</p>"]
    for name, cap in captions.items():
        html.append(f"<figure><img src='{name}'><figcaption><b>{name}</b> &mdash; {cap}</figcaption></figure>")
    html.append("</body></html>")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(html))
    print("wrote index.html")


if __name__ == "__main__":
    collect() if "--collect" in sys.argv else main()

"""
run_projection.py  --  "where will this cow be, and what will she earn, over the next N years?"

THE ONLY FILE YOU NEED TO EDIT AND RUN:      python projection/run_projection.py

WHAT IT DOES
  1. You describe one or more starting states below (lactation, days in milk, days pregnant) and give each a LABEL.
  2. For each time resolution used (month / week / day) it builds the model ONCE with the existing code:
        cow.states / cow.probabilities / cow.transitions / cow.rewards   (the biology and the economics)
        mrp.core                                                         (distribution evolution, discounting, simulation)
     This file contains NO model equations: no probability, no reward, no discount formula is written here.
  3. It projects the probability distribution over states forward in time, and the expected reward of EVERY reward component
     at every future step (per day, and cumulative with discounting), and writes one interactive Plotly file per reward plus one per
     scenario for the state distribution.

OUTPUT (outputs/projection/)
  index.html                 list of everything          projection_all.html     every figure on one page
  reward_<name>.html         one figure per reward: expected reward per day (left), cumulative discounted reward (right)
  distribution_<label>.html  status / lactation / days-in-milk distribution over time, and "original cow still in herd"
  summary.csv, timeseries_<label>.csv     the numbers behind the figures (open in Excel)

HOW TIME AND DISCOUNTING WORK
  Reward at step t is collected in the state the cow occupies at step t; the cumulative curve is sum_t gamma^t * E[reward_t] with
  gamma = ANNUAL_DISCOUNT_FACTOR ** (step length / 365) (mrp.core.annual_to_per_step_gamma). The final cumulative total of
  'total' equals the finite-horizon value from mrp.core.evaluate_finite_horizon (checked on every run).
  A cow that dies or is culled is replaced by a new heifer (the model's replacement state), so the projection follows the cow SLOT:
  the replacement cost and the heifer's future milk are included. "original cow still in herd" shows when the first cow has left.
"""
import os, sys, json, subprocess, html, re

# =====================================================================================================================
# 1. USER SETTINGS
# =====================================================================================================================
YEARS = 2                       # projection horizon
ANNUAL_DISCOUNT_FACTOR = 0.95   # value today of 1 US$ received in one year (the discount convention used everywhere in CowMRP)
N_TRAJECTORIES = 0              # 0 = expected values only. >0 = also simulate that many individual cows (Monte Carlo, slower) and draw
                                #   10 % - 90 % bands on the cumulative reward ("possible futures" of a single cow)
RANDOM_SEED = 1
OUTPUT_FOLDER = "outputs/projection"

# ---- SCENARIOS --------------------------------------------------------------------------------------------------------
# Each scenario is a dict:
#   label        any text you like; used in titles, legends and file names
#   time_unit    "month" | "week" | "day"  -> the time discretization of the model for this scenario
#                (month: 1,210 states, seconds; week: 15,630 states, about a minute; day: 698,070 states, several minutes)
#   lactation    1 ... 10
#   STATE, either in CALENDAR DAYS (converted to model steps for you):
#       dim_days = days in milk (days since calving), dp_days = days pregnant (0 = open cow)
#   or directly in MODEL STEPS of the chosen time_unit:
#       dim = step index, dp = step index          (advanced; use dim_days / dp_days if unsure)
# Rules the model enforces: a pregnant cow (dp_days > 0) must have been bred between day 50 and day 300 in milk, so
# dim_days - dp_days must lie in that window; dry-off is at 222 days pregnant; calving at about 277.
# If a state is impossible you get an error that lists valid choices.
SCENARIOS = [
    # >>> ACTIVE: every dict(...) line WITHOUT a leading # is run; put a # in front to switch one off. <<<
    # 0. Open cow, lactation 2, day 1 in milk, not pregnant, at DAY resolution (698,070 states: takes a few minutes).
    dict(label="Open cow, lactation 2, DIM 1 (DAY)", time_unit="day", lactation=2, dim_days=1, dp_days=0),

    # 1. A new heifer entering the herd (start of lactation 1). The classic "value of a replacement" case.
    # dict(label="New heifer, start of lactation 1 (WEEK)", time_unit="week", lactation=1, dim_days=0, dp_days=0),

    # 2. An open cow, lactation 2, 150 days in milk. She can still be bred (days 50-300).
    # dict(label="Open cow, lactation 2, DIM 150 (WEEK)", time_unit="week", lactation=2, dim_days=150, dp_days=0),

    # 3. The same cow, pregnant 60 days (conceived at DIM 90). Compare with scenario 2 = the value of the pregnancy.
    # dict(label="Same cow, pregnant 60 d (WEEK)", time_unit="week", lactation=2, dim_days=150, dp_days=60),

    # 4. A dry cow: lactation 3, 230 days pregnant (dried off at 222), 330 days in milk, calving in about 47 days.
    # dict(label="Dry cow, lactation 3, 230 d pregnant (WEEK)", time_unit="week", lactation=3, dim_days=330, dp_days=230),

    # 5. The same open cow at MONTH resolution, to see the effect of the time discretization (faster, a bit less exact).
    # dict(label="Open cow, lactation 2, DIM 150 (MONTH)", time_unit="month", lactation=2, dim_days=150, dp_days=0),

    # 6. DAY resolution (most exact, slow: 698,070 states).
    # dict(label="Open cow, lactation 2, DIM 150 (DAY)", time_unit="day", lactation=2, dim_days=150, dp_days=0),

    # 7. State given directly in model steps (WEEK: dim 20 = about day 147; dp 8 = about 56 days pregnant).
    # dict(label="Lactation 4, steps dim=20 dp=8 (WEEK)", time_unit="week", lactation=4, dim=20, dp=8),
]

# Reward components drawn (money, US$): the names are the model's own (cow/rewards/). Costs are negative.
# Tracked physical quantities (not money) are drawn as extra figures, per day, cumulative without discounting.
PHYSICAL_QUANTITIES = {"milk_produced_kg": "milk produced, kg", "dmi_kg": "dry matter intake, kg", "enteric_ch4_g": "enteric methane, g",
                       "respiratory_co2_kg": "respiratory CO2, kg", "nitrogen_intake_g": "nitrogen intake, g", "manure_n_g": "manure nitrogen, g"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# =====================================================================================================================
# 2. WORKER: one process per time resolution (the model reads TIME_UNIT at import, so units cannot share a process)
# =====================================================================================================================
def worker(unit, indices, out_path):
    os.environ["COW_TIME_UNIT"] = unit
    sys.path.insert(0, ROOT)
    import io, contextlib
    import numpy as np
    import cow.parameters as P
    from mrp import core as mc, simulator as sim
    from cow.states import build_cow_states, repro_status
    from cow.probabilities import compute_probabilities
    from cow.rewards import compute_rewards, BASE_REWARD_SPECS
    from cow.transitions import compute_transitions, get_replacement_state_id

    U = P.UNIT_DAYS
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = mc.assemble_matrix(states, rows, cols, vals)
    mc.validate_row_stochasticity(matrix, states)
    rewards = compute_rewards(states, probs)
    rid = get_replacement_state_id(states)
    n_steps = mc.steps_per_year(U) * YEARS
    gamma = mc.annual_to_per_step_gamma(ANNUAL_DISCOUNT_FACTOR, U)

    # reward matrix: signed monetary components, then total, environmental rewards if switched on, then physical quantities
    money = [(s["name"], s["sign"] * np.asarray(rewards[s["name"]], float)) for s in BASE_REWARD_SPECS]
    money.append(("total", np.asarray(rewards["total"], float)))
    for k in ("reward_nitrogen", "reward_enteric_methane", "reward_manure_methane", "reward_n2o"):
        if k in rewards and np.abs(np.asarray(rewards[k], float)).sum() > 0:
            money.append((k, np.asarray(rewards[k], float)))
    phys = [(k, np.asarray(rewards[k], float)) for k in PHYSICAL_QUANTITIES if k in rewards]
    names = [n for n, _ in money] + [n for n, _ in phys]
    R = np.column_stack([v for _, v in money] + [v for _, v in phys])
    n_money = len(money)

    status = np.array(repro_status(states["dp"].values))
    lac = states["lactation"].values
    dim_bin = np.minimum(((states["dim"].values + 1) * U) // 30, 35).astype(int)     # 30-day bins of days in milk (mean age during the step)
    results = {}
    for idx in indices:
        sc = SCENARIOS[idx]
        dim = sc["dim"] if "dim" in sc else P.dim_step_of_days(sc.get("dim_days", 0))
        dp = sc["dp"] if "dp" in sc else P.dp_step_of_days(sc.get("dp_days", 0))
        try:
            s0 = mc.find_state_id(states, lactation=sc["lactation"], dim=dim, dp=dp)
        except ValueError as e:
            near = states[(states["lactation"] == sc["lactation"]) & ((states["dp"] > 0) == (dp > 0))].copy()
            near["gap"] = (near["dim"] - dim).abs() + (near["dp"] - dp).abs()
            ex = near.sort_values("gap").head(5)
            sugg = "\n".join(f"   dict(lactation={int(r.lactation)}, dim_days={int((r.dim + 1) * U)}, dp_days={int(r.dp * U)})   # model steps dim={int(r.dim)}, dp={int(r.dp)}" for r in ex.itertuples())
            raise SystemExit(f"\nScenario '{sc['label']}' ({unit}): this state does not exist in the model.\n"
                             f"A pregnant cow must have been bred between day 50 and day 300 in milk (dim_days - dp_days must lie in that window); one {unit} step = {U} day(s).\n"
                             f"Closest valid states, in calendar days:\n{sugg}\n(technical detail: {str(e).splitlines()[0]})")
        pi_u = mc.make_initial_distribution(states, lactation=sc["lactation"], dim=dim, dp=dp)
        pi_t = np.zeros_like(pi_u)
        T = n_steps
        exp = np.zeros((T, R.shape[1]))
        shares_status = np.zeros((T, 3)); shares_lac = np.zeros((T, P.MAX_LACTATION)); dist_dim = np.zeros((T, 36))
        orig = np.zeros(T)
        for t in range(T):
            pi = pi_u + pi_t                                              # distribution over states at step t
            exp[t] = pi @ R
            for j, s in enumerate(("Open", "Pregnant", "Dry")):
                shares_status[t, j] = pi[status == s].sum()
            shares_lac[t] = np.bincount(lac - 1, weights=pi, minlength=P.MAX_LACTATION)
            dist_dim[t] = np.bincount(dim_bin, weights=pi, minlength=36)
            orig[t] = pi_u.sum()
            pi_u, pi_t = mc.distribution_recursive_origin_tracked(matrix, pi_u, pi_t, 1, rid)   # advance one step
        disc = gamma ** np.arange(T)
        wmat = np.ones((T, R.shape[1]))
        wmat[:, :n_money] = disc[:, None]                                   # money is discounted, physical quantities are not
        cum = np.cumsum(exp * wmat, axis=0)
        # consistency with the engine's own finite-horizon value iteration
        with contextlib.redirect_stdout(io.StringIO()):
            V = mc.evaluate_finite_horizon(matrix, rewards["total"], gamma, YEARS, U)
        total_col = names.index("total")
        check = abs(float(V[s0]) - float(cum[-1, total_col])) / max(1.0, abs(float(V[s0])))
        # same distribution as the engine's plain recursion
        end_ref = mc.distribution_recursive(matrix, mc.make_initial_distribution(states, lactation=sc["lactation"], dim=dim, dp=dp), T)
        check_dist = float(np.abs(end_ref - (pi_u + pi_t)).sum())
        r = {"label": sc["label"], "unit": unit, "U": U, "start": {"lactation": sc["lactation"], "dim_step": int(dim), "dp_step": int(dp)},
             "t_days": (np.arange(T) * U).tolist(), "names": names, "n_money": n_money, "exp_per_step": exp.tolist(), "cum": cum.tolist(),
             "status": shares_status.tolist(), "lactation_shares": shares_lac.tolist(), "dim_bins": dist_dim.tolist(), "original_still_here": orig.tolist(),
             "gamma": gamma, "finite_horizon_value": float(V[s0]), "check_vs_finite_horizon": check, "check_distribution": check_dist,
             "n_states": int(len(states))}
        if N_TRAJECTORIES > 0:
            hist = sim.run_monte_carlo(matrix, [s0], T, n_sims=N_TRAJECTORIES, seed=RANDOM_SEED)
            ids = np.array([h[:T, 0] for h in hist])                         # (n_sims, T)
            per = R[ids]                                                      # (n_sims, T, n_cols)
            cum_sims = np.cumsum(per * wmat[None, :, :], axis=1)
            r["mc"] = {"p10": np.percentile(cum_sims, 10, axis=0).tolist(), "p50": np.percentile(cum_sims, 50, axis=0).tolist(), "p90": np.percentile(cum_sims, 90, axis=0).tolist(),
                       "n": N_TRAJECTORIES}
        results[idx] = r
        print(f"[{unit}] '{sc['label']}': {T} steps, cumulative discounted total {cum[-1, total_col]:,.0f} US$ "
              f"(engine finite-horizon value {float(V[s0]):,.0f}; relative difference {check:.1e}; distribution check {check_dist:.1e})")
    json.dump(results, open(out_path, "w"))


# =====================================================================================================================
# 3. FIGURES (Plotly) and tables. No model code here either: it only draws what the workers returned.
# =====================================================================================================================
def day_ticks(total_days):
    """Tick spacing (days) for the time axis: about 8-12 ticks, a multiple of 30 days."""
    return 30 * max(1, -(-total_days // 300))


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60]


def make_outputs(results):
    import numpy as np, pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as po
    from plotly.colors import qualitative
    out = os.path.join(ROOT, OUTPUT_FOLDER)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "plotly.min.js"), "w", encoding="utf-8").write(po.get_plotlyjs())
    figs = []        # (file, title, caption, fig)
    colors = qualitative.D3
    ordered = [results[i] for i in sorted(results)]
    names = ordered[0]["names"]; n_money = ordered[0]["n_money"]
    nice = {"milk_income": "Milk income", "pregnancy_reward": "Pregnancy reward (calf value)", "cull_reward": "Cull reward (salvage - culling cost)", "mortality_cost": "Mortality cost",
            "replacement_cost": "Replacement cost (rearing - salvage)", "feed_cost": "Feed cost", "reproduction_cost": "Reproduction cost (AI, diagnosis, synchronization)", "total": "TOTAL reward",
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
        fig.update_xaxes(title_text="days from now", title_standoff=8, dtick=day_ticks(YEARS * 365), tick0=0, tickangle=0)
        fig.update_layout(template="plotly_white", title=dict(text=f"{nice.get(key, key)} over time", x=0.01), height=520,
                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), margin=dict(l=70, r=30, t=90, b=120))
        cap = (f"{nice.get(key, key)}. Left: expected amount per day at each future time, from the model's own transition and reward vectors "
               f"(a cow that is culled or dies is replaced by a heifer, whose cost and future income are included). "
               + ("Right: running total, discounted at " + f"{ANNUAL_DISCOUNT_FACTOR} per year" + (" (shaded: 10-90 % of simulated individual cows)." if N_TRAJECTORIES else ".") if money else "Right: running total, not discounted."))
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
        fig.update_xaxes(title_text="days from now", title_standoff=8, dtick=day_ticks(YEARS * 365), tick0=0, tickangle=0)
        fig.update_yaxes(title_text="probability", row=1, col=1); fig.update_yaxes(title_text="probability", row=1, col=2)
        fig.update_yaxes(title_text="days in milk", row=1, col=3)
        s0 = r["start"]
        fig.update_layout(template="plotly_white", height=560, title=dict(text=f"Where the cow will be: {r['label']}", x=0.01),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0), margin=dict(l=70, r=30, t=100, b=130))
        cap = (f"Starting state: lactation {s0['lactation']}, model step dim = {s0['dim_step']}, dp = {s0['dp_step']} ({r['unit']} resolution, step = {r['U']} d, {r['n_states']:,} states). "
               "All panels are probability distributions of ONE cow slot over the future, computed by repeatedly applying the model's transition matrix. Replaced animals are renewed by a heifer, "
               "so the status and lactation panels describe the slot; the last panel isolates the original cow.")
        figs.append((f"distribution_{slug(r['label'])}", f"Distribution over time: {r['label']}", cap, fig))
    # write
    index = ["<html><head><meta charset='utf-8'><title>CowMRP projection</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto}li{margin:8px 0}</style></head><body><h1>CowMRP projection</h1>",
             f"<p>Horizon {YEARS} years, discount factor {ANNUAL_DISCOUNT_FACTOR} per year. Scenarios: " + "; ".join(html.escape(r["label"]) for r in ordered) + ". <a href='projection_all.html'>All figures on one page</a>.</p><ul>"]
    allp = ["<html><head><meta charset='utf-8'><title>CowMRP projection (all)</title><script src='plotly.min.js'></script></head><body style='font-family:sans-serif;max-width:1400px;margin:auto'><h1>CowMRP projection</h1>"]
    for name, title, cap, fig in figs:
        fig.write_html(os.path.join(out, name + ".html"), include_plotlyjs="directory", full_html=True)
        index.append(f"<li><a href='{name}.html'><b>{html.escape(title)}</b></a><br><span style='color:#444'>{html.escape(cap)}</span></li>")
        allp.append(f"<h2>{html.escape(title)}</h2>" + fig.to_html(full_html=False, include_plotlyjs=False, div_id=name + "_div") + f"<p>{html.escape(cap)}</p>")
    if os.environ.get("PROJECTION_PNG_DIR"):          # optional static copies for a quick look (needs the kaleido package; not required)
        os.makedirs(os.environ["PROJECTION_PNG_DIR"], exist_ok=True)
        for name, title, cap, fig in figs:
            fig.write_image(os.path.join(os.environ["PROJECTION_PNG_DIR"], name + ".png"), width=1400, height=int(fig.layout.height or 520))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write("\n".join(index) + "</ul></body></html>")
    open(os.path.join(out, "projection_all.html"), "w", encoding="utf-8").write("\n".join(allp) + "</body></html>")
    # tables
    rows = []
    for r in ordered:
        cum = np.array(r["cum"]); t = np.array(r["t_days"]); U = r["U"]
        spy = round(365 / U)
        for j, key in enumerate(names):
            row = {"scenario": r["label"], "time_unit": r["unit"], "reward": key, "money": j < n_money}
            for y in range(1, YEARS + 1):
                row[f"cumulative_year_{y}"] = float(cum[min(y * spy, len(cum)) - 1, j])
            rows.append(row)
        ts = pd.DataFrame({"year": t / 365.0, "days": t})
        for j, key in enumerate(names):
            ts[f"{key}_per_day"] = np.array(r["exp_per_step"])[:, j] / U
            ts[f"{key}_cumulative"] = cum[:, j]
        for k, nm in enumerate(("Open", "Pregnant", "Dry")):
            ts[f"share_{nm}"] = np.array(r["status"])[:, k]
        ts["original_cow_still_in_herd"] = r["original_still_here"]
        ts.to_csv(os.path.join(out, f"timeseries_{slug(r['label'])}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)
    print(f"\nwrote {len(figs)} Plotly files + tables to {out}\nopen: {os.path.join(out, 'index.html')}")


# =====================================================================================================================
# 4. MAIN: group scenarios by time resolution, run one worker process per resolution, draw
# =====================================================================================================================
def main():
    if not SCENARIOS:
        raise SystemExit("SCENARIOS is empty: add at least one scenario at the top of this file.")
    for i, sc in enumerate(SCENARIOS):
        if sc.get("time_unit") not in ("month", "week", "day"):
            raise SystemExit(f"Scenario {i} ('{sc.get('label')}'): time_unit must be 'month', 'week' or 'day'.")
        if "label" not in sc or not (1 <= int(sc.get("lactation", 0)) <= 10):
            raise SystemExit(f"Scenario {i}: needs a label and lactation 1-10.")
    out = os.path.join(ROOT, OUTPUT_FOLDER, "_data")
    os.makedirs(out, exist_ok=True)
    results = {}
    for unit in ("month", "week", "day"):
        idx = [i for i, sc in enumerate(SCENARIOS) if sc["time_unit"] == unit]
        if not idx:
            continue
        path = os.path.join(out, f"{unit}.json")
        print(f"--- {unit.upper()}: {len(idx)} scenario(s)")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", unit, ",".join(map(str, idx)), path], cwd=ROOT)
        if p.returncode:
            raise SystemExit(p.returncode)
        for k, v in json.load(open(path)).items():
            results[int(k)] = v
    make_outputs(results)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        a = sys.argv.index("--worker")
        worker(sys.argv[a + 1], [int(x) for x in sys.argv[a + 2].split(",")], sys.argv[a + 3])
    else:
        main()

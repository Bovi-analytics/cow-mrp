"""
run_projection.py  --  "where will this cow be, and what will she earn, over the next N years?"

THE ONLY FILE YOU NEED TO EDIT AND RUN:      python runners/cow/projection.py

(The calculation lives in the package, cowmrp.domains.cow.projection.compare_cows, and the figures and tables in
cowmrp.viz.projection. This file keeps the settings below and writes the files.
From Python: compare_cows([dict(label=..., time_unit=..., lactation=..., dim_days=..., dp_days=...), ...], years=2).)

WHAT IT DOES
  1. You describe one or more starting states below (lactation, days in milk, days pregnant) and give each a LABEL.
  2. For each time resolution used (month / week / day) it builds the model ONCE with the existing code:
        cow.states / cow.probabilities / cow.transitions / cow.rewards   (the biology and the economics)
        cowmrp.mrp.core                                                         (distribution evolution, discounting, simulation)
     This file contains NO model equations: no probability, no reward, no discount formula is written here.
  3. It projects the probability distribution over states forward in time, and the expected reward of EVERY reward component
     at every future step (per day, and cumulative with discounting), and writes one interactive Plotly file per reward plus one per
     scenario for the state distribution.

OUTPUT (runners/cow/outputs/projection/)
  index.html                 list of everything          projection_all.html     every figure on one page
  reward_<name>.html         one figure per reward: expected reward per day (left), cumulative discounted reward (right)
  distribution_<label>.html  status / lactation / days-in-milk distribution over time, and "original cow still in herd"
  summary.csv, timeseries_<label>.csv     the numbers behind the figures (open in Excel)

HOW TIME AND DISCOUNTING WORK
  Reward at step t is collected in the state the cow occupies at step t; the cumulative curve is sum_t gamma^t * E[reward_t] with
  gamma = ANNUAL_DISCOUNT_FACTOR ** (step length / 365) (cowmrp.mrp.core.annual_to_per_step_gamma). The final cumulative total of
  'total' equals the finite-horizon value from cowmrp.mrp.core.evaluate_finite_horizon (checked on every run).
  A cow that dies or is culled is replaced by a new heifer (the model's replacement state), so the projection follows the cow SLOT:
  the heifer's future milk and costs are included. The replacement cost is NOT in the reward vector or in TOTAL; it is shown as the
  extra series "replacement_cost_memo" (expected spending on replacements, negative) so it is not lost. "original cow still in herd"
  shows when the first cow has left.
"""
import os, sys, html

# =====================================================================================================================
# 1. USER SETTINGS
# =====================================================================================================================
YEARS = 2                       # projection horizon
ANNUAL_DISCOUNT_FACTOR = 0.95   # value today of 1 US$ received in one year (the discount convention used everywhere in CowMRP)
N_TRAJECTORIES = 0              # 0 = expected values only. >0 = also simulate that many individual cows (Monte Carlo, slower) and draw
                                #   10 % - 90 % bands on the cumulative reward ("possible futures" of a single cow)
RANDOM_SEED = 1
OUTPUT_FOLDER = "runners/cow/outputs/projection"

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
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)                 # lets the script run from a checkout without installing the package
sys.path.insert(0, os.path.join(ROOT, "src"))


def write_outputs(results, out):
    """Write the Plotly files, index pages and CSV tables for `results` (from compare_cows) into folder `out`."""
    import plotly.offline as po
    from cowmrp.viz.projection import projection_figures, projection_tables
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "plotly.min.js"), "w", encoding="utf-8").write(po.get_plotlyjs())
    ordered = [results[i] for i in sorted(results)]
    figs = projection_figures(results, YEARS, ANNUAL_DISCOUNT_FACTOR, N_TRAJECTORIES)
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
    summary, series = projection_tables(results, YEARS)
    for slug, ts in series.items():
        ts.to_csv(os.path.join(out, f"timeseries_{slug}.csv"), index=False)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(f"\nwrote {len(figs)} Plotly files + tables to {out}\nopen: {os.path.join(out, 'index.html')}")


def main():
    from cowmrp.domains.cow.projection import compare_cows
    try:
        results = compare_cows(SCENARIOS, YEARS, ANNUAL_DISCOUNT_FACTOR, N_TRAJECTORIES, RANDOM_SEED)
    except ValueError as e:
        raise SystemExit(str(e))
    out = os.path.join(ROOT, OUTPUT_FOLDER)
    os.makedirs(os.path.join(out, "_data"), exist_ok=True)
    write_outputs(results, out)


if __name__ == "__main__":
    main()

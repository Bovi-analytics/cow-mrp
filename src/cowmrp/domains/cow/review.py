"""
domains/cow/review.py
=====================
What the review pages (runners/_shared/review.py) show for the cow: every probability, reward and physical quantity of the model by
days in milk (one line per lactation) and by days pregnant, open / pregnant / dry cows apart. Values come straight from the model's
own arrays; this file only arranges them. Item format: see runners/_shared/review.py.
"""
import numpy as np
import pandas as pd

from cowmrp.mrp import core
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.cow.states import repro_status
from cowmrp.domains.cow.notebook_hooks import INITIAL_STATE, replacement_state_id

NOTES = [
    "The cow's state is (lactation, dim, dp): lactation number, steps since calving, steps since conception (0 = open). "
    f"One step = one {TIME_UNIT} ({UNIT_DAYS} d); the axes show days. Open = not pregnant, Pregnant = pregnant and milking, Dry = pregnant and dried off.",
    "In the reward arrays costs are stored as positive numbers and subtracted in the total reward. Money is USD per cow per step; replacement cost is not in the reward (it is charged in the cow-versus-heifer comparison).",
    "The environmental rewards are switched off in the default model (all zero) and are not shown.",
]

STATUSES = ("Open", "Pregnant", "Dry")
REWARD_TITLES = {
    "milk_income": "Milk income", "pregnancy_reward": "Pregnancy reward", "cull_reward": "Cull reward (salvage)",
    "mortality_cost": "Mortality cost", "feed_cost": "Feed cost", "reproduction_cost": "Reproduction cost", "total": "Total reward",
}
REWARD_CHECK = {
    "milk_income": "Follows the lactation curve: rising to a peak in the first months, then falling, higher for later lactations, zero when dry? Is the size per step plausible?",
    "pregnancy_reward": "The calf value (1,150) paid in the step the cow actually calves: probability of surviving, not aborting and calving, times the calf value. Zero for open cows and for most of the pregnancy, then large in the calving step. Is that the shape and size you expect?",
    "cull_reward": "Expected salvage in the step: probability of being culled (or ending the last lactation) times the net meat value (live weight x dressing % x price - culling cost). Small where culling is rare, close to the full meat value where culling is certain. Do both ends make sense?",
    "mortality_cost": "Expected cost of death in the step: probability of dying times the loss. Small, and following the death probability? Is the size right?",
    "feed_cost": "Cost of the feed eaten in the step. Should follow intake: higher in early lactation, lower when dry. Plausible per step?",
    "reproduction_cost": "Cost of breeding (inseminations, heat detection) while open. Only open cows pay it, and only after the waiting period. Is that so?",
    "total": "Milk income and the credits minus the costs, per step. Positive for a healthy milking cow, lower or negative when dry or very late? Do the pieces add up the way you expect?",
}


def build_items(model):
    st, pr, rw, P = model.states, model.probs, model.rewards, model.P
    status = np.asarray(repro_status(st["dp"].values))
    mask = {s: status == s for s in STATUSES}
    days_milk = st["dim"].values * UNIT_DAYS
    days_preg = st["dp"].values * UNIT_DAYS
    X, XP = "Days in milk (start of step)", "Days pregnant (start of step)"
    per = f"per {TIME_UNIT}"

    def pivot(values, m, by="dim", cols="lactation", label="Lactation"):
        idx = (days_milk if by == "dim" else days_preg)
        d = pd.DataFrame({"x": idx, "c": st[cols].values, "v": np.asarray(values, dtype=float).reshape(-1)})[m]
        t = d.pivot_table(index="x", columns="c", values="v", aggfunc="mean")
        t.columns = [f"{label} {c}" for c in t.columns]
        return t

    def panel(title, df, xlabel, ylabel, kind="lines"):
        return dict(title=title, kind=kind, df=df, xlabel=xlabel, ylabel=ylabel)

    items = []
    items.append(dict(id="p_preg", group="Probabilities", title="Conception, open cows", unit=f"probability {per}",
        check="Zero during the voluntary waiting period, then a plausible monthly rate that falls with days in milk? Lower for later lactations?",
        panels=[panel("Probability of conceiving this step (open cows)", pivot(pr["p_preg"], mask["Open"]), X, f"probability {per}")]))
    items.append(dict(id="p_resume", group="Probabilities", title="Re-conception after a pregnancy loss", unit=f"probability {per}",
        check="Only matters in the step where a pregnancy is lost: the chance of conceiving again in the remaining part of that step. Should be well below the ordinary conception rate. Is it?",
        panels=[panel("Probability of conceiving again after a loss (states where it applies)", pivot(pr["p_resume"], np.asarray(pr["p_resume"]) > 0), X, f"probability {per}")]))
    items.append(dict(id="p_abort", group="Probabilities", title="Pregnancy loss", unit=f"probability {per}",
        check="By days pregnant. Highest in early pregnancy and falling later? Higher for later lactations? Is the early level in line with the early-loss figures the model cites?",
        panels=[panel("Probability of losing the pregnancy this step", pivot(pr["p_abort"], mask["Pregnant"] | mask["Dry"], by="dp"), XP, f"probability {per}")]))
    items.append(dict(id="p_calve", group="Probabilities", title="Calving", unit=f"probability {per}",
        check="Zero until close to term and then 1 by the end of gestation (about 280 days). Does the calving window sit where you expect?",
        panels=[panel("Probability of calving this step", pivot(pr["p_calve"], mask["Pregnant"] | mask["Dry"], by="dp"), XP, f"probability {per}")]))
    items.append(dict(id="p_dead", group="Probabilities", title="Death", unit=f"probability {per}",
        check="Small, higher in the first weeks after calving and in later lactations? Is the annual total (about 4-6 % of cows) in line with what you know?",
        panels=[panel(f"{s} cows", pivot(pr["p_dead"], mask[s]), X, f"probability {per}") for s in STATUSES]))
    items.append(dict(id="p_cull", group="Probabilities", title="Culling", unit=f"probability {per}",
        check="Voluntary culling plus the milk-floor and reproduction culls, combined. Where it reaches 1 the cow is always culled (milk below the floor, or past the last breeding month). Do those places make sense?",
        panels=[panel(f"{s} cows", pivot(pr["p_cull"], mask[s]), X, f"probability {per}") for s in STATUSES]))

    for c in ("milk_income", "pregnancy_reward", "cull_reward", "mortality_cost", "feed_cost", "reproduction_cost", "total"):
        items.append(dict(id=f"r_{c}", group="Rewards", title=REWARD_TITLES[c], unit=f"USD per cow {per}", check=REWARD_CHECK[c],
            panels=[panel(f"{s} cows", pivot(rw[c], mask[s]), X, f"USD {per}") for s in STATUSES]))
    for c, title, unit, chk in (("milk_produced_kg", "Milk produced", f"kg per cow {per}", "The lactation curve in kilograms: peak at 40-70 days, falling after, zero when dry? Higher for later lactations?"),
                                ("dmi_kg", "Dry matter intake", f"kg per cow {per}", "Follows milk and body weight: highest in early lactation, lower when dry. Per-day values (divide by days) in the usual 20-25 kg range for a milking cow?"),
                                ("bw_kg", "Body weight", "kg", "Rises with lactation number and flattens in later lactations? Level plausible for the breed?")):
        items.append(dict(id=f"b_{c}", group="Biology behind the rewards", title=title, unit=unit, check=chk,
            panels=[panel(f"{s} cows", pivot(rw[c], mask[s]), X, unit) for s in STATUSES]))

    gamma = core.annual_to_per_step_gamma(0.95, UNIT_DAYS)
    V = core.evaluate_infinite_horizon(P, rw["total"], gamma)
    items.append(dict(id="v_value", group="Values", title="Value of a cow (infinite horizon)", unit="USD per cow",
        check="Discounted value of being in this state forever after (5 % annual discount). Higher for productive mid-lactation cows, lower late or in old age; a pregnant cow should be worth more than an open one at the same day in milk. Is it so?",
        panels=[panel(f"{s} cows", pivot(V, mask[s]), X, "USD") for s in STATUSES]))

    count = pd.DataFrame({s: pd.Series(1, index=days_milk[mask[s]]).groupby(level=0).sum() for s in STATUSES}).fillna(0)
    items.append(dict(id="n_states", group="Population", title="States per day in milk", unit="states",
        check="How many (lactation, dim, dp) states exist at each day in milk. Pregnant states should only appear once the waiting period is over.",
        panels=[panel("Number of states", count, X, "states")]))

    start = core.make_initial_distribution(st, **INITIAL_STATE)
    rid = replacement_state_id(st)
    n_steps = core.steps_per_year(UNIT_DAYS)
    po, pt = start.copy(), np.zeros_like(start)
    rows = []
    for t in range(n_steps + 1):
        pi = po + pt
        rows.append(dict(days=t * UNIT_DAYS, Open=float(pi[mask["Open"]].sum()), Pregnant=float(pi[mask["Pregnant"]].sum()), Dry=float(pi[mask["Dry"]].sum()),
                         **{"Never replaced": float(po.sum()), "Replaced at least once": float(pt.sum())}))
        po, pt = core.distribution_recursive_origin_tracked(P, po, pt, 1, rid)
    h = pd.DataFrame(rows).set_index("days")
    items.append(dict(id="herd_year", group="Population", title="One new heifer followed for a year", unit="share of the animal's probability mass",
        check="Starting as a fresh first-lactation cow (dim 0, open). Does the open / pregnant / dry split develop as you expect, and does the share replaced (culled or dead) over the year look right?",
        panels=[panel("Open, pregnant or dry", h[list(STATUSES)], "Days since start", "share", "area"),
                panel("Replaced or not", h[["Never replaced", "Replaced at least once"]], "Days since start", "share", "area")]))
    return items

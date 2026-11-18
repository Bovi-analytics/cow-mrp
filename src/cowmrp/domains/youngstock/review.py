"""
domains/youngstock/review.py
============================
What the review pages (runners/_shared/review.py) show for the youngstock chain (calf -> heifer -> first calving): the per-step
probabilities by age, stage and days pregnant, where terminal calves go, the birth mix by breeding program, and how a birth cohort
moves through the chain. Values come straight from the model's own probabilities and matrix Q; this file only arranges them.
Item format: see runners/_shared/review.py.
"""
import numpy as np
import pandas as pd

from cowmrp.mrp import core
from cowmrp.domains.youngstock import parameters as YP
from cowmrp.domains.youngstock.probabilities import compute_probabilities

TU, UD = YP.TIME_UNIT, YP.UNIT_DAYS
PROGRAMS = ("conventional", "sexed_dairy", "beef_on_dairy")
OUTCOMES = ["GRADUATED", "SOLD", "DIED", "CULLED", "STILLBORN"]

NOTES = [
    "Youngstock has no rewards in this model (no prices or costs are attached to a calf or heifer here); the herd module prices the flows. So this page shows probabilities and populations only.",
    f"One step = one {TU} ({UD} d); ages are shown in days. Lines: DAIRY_F = dairy heifer calves, DAIRY_M = dairy bull calves, BEEF_F / BEEF_M = beef-cross calves.",
    f"Calves that are not kept leave the farm at CALF_SALE_AGE_DAYS = {YP.CALF_SALE_AGE_DAYS:g} d (a farmer input). At {TU} steps that is within the first step; use --time-unit week or day for a finer picture.",
]


def _panel(title, df, xlabel, ylabel, kind="lines"):
    return dict(title=title, kind=kind, df=df, xlabel=xlabel, ylabel=ylabel)


def _next_stage(P, st, sid):
    """Where one step from state `sid` leads, summed by (target line if absorbing, target stage), from the matrix itself."""
    row = P.getrow(sid)
    tgt_stage = st["stage"].values[row.indices]
    out = {}
    for s, p in zip(tgt_stage, row.data):
        out[s] = out.get(s, 0.0) + float(p)
    return out


def _birth_mix(states, program):
    """birth_mix per dam parity under a breeding program (the program is read live by the domain; restored afterwards)."""
    old = YP.BREEDING_PROGRAM
    YP.BREEDING_PROGRAM = program
    try:
        return {k: np.asarray(v, float) for k, v in compute_probabilities(states)["birth_mix"].items()}
    finally:
        YP.BREEDING_PROGRAM = old


def _cohort(P, st, births, n_steps):
    """Expected path of a birth cohort (one birth event; twins and stillbirths included) through Q."""
    pi = births / births.sum()
    alive = st["alive"].values
    stage = st["stage"].values
    line_stage = (st["line"] + " " + st["stage"]).values
    rows, live = [], []
    for t in range(n_steps + 1):
        row = {"days": t * UD, "alive": float(pi[alive].sum())}
        row.update({o: float(pi[stage == o].sum()) for o in OUTCOMES})
        rows.append(row)
        live.append(pd.Series(pi[alive]).groupby(line_stage[alive]).sum())
        pi = core.distribution_recursive(P, pi, 1)
    return pd.DataFrame(rows).set_index("days"), pd.DataFrame(live).fillna(0.0).set_axis([t * UD for t in range(n_steps + 1)])


def build_items(model):
    st, pr, P = model.states, model.probs, model.P
    items = []
    unit_p = f"probability per {TU}"

    d = st[(st["line"] == "DAIRY_F") & st["stage"].isin(["CALF", "HEIFER"])]
    items.append(dict(id="p_die", group="Probabilities", title="Calf and heifer death by age", unit=unit_p,
        check="Highest in the first weeks of life and low for weaned heifers? Is the preweaning level in line with the roughly 5 % preweaning mortality the model is built on?",
        panels=[_panel("Dairy heifer calf / heifer: probability of dying this step", pd.DataFrame({"Dairy heifer": pr["p_die"][d.index]}, index=d["age_days"].values), "Age (days)", unit_p)]))

    term = st[st["line"].isin(["DAIRY_M", "BEEF_F", "BEEF_M"]) & (st["stage"] == "CALF")]
    rows = {}
    for sid, line, age in zip(term["state_id"], term["line"], term["age"]):
        n = _next_stage(P, st, int(sid))
        rows[f"{line}, calf age step {int(age)}"] = {"sold": n.get("SOLD", 0.0), "died before sale": n.get("DIED", 0.0), "still on the farm": n.get("CALF", 0.0)}
    items.append(dict(id="terminal_fate", group="Probabilities", title="Bull and beef-cross calves: next step", unit=unit_p,
        check=f"What happens to a bull or beef-cross calf in its first steps: sold, died before sale, or still on the farm. Sale age is {YP.CALF_SALE_AGE_DAYS:g} d; the pre-sale death is about 2.4 % (a placeholder for beef-cross). Do the shares match that?",
        panels=[_panel("Fate of terminal calves in one step", pd.DataFrame(rows).T, "State", unit_p, "bars")]))

    b = st[(st["line"] == "DAIRY_F") & (st["stage"] == "BREEDING")]
    n = _next_stage(P, st, int(b["state_id"].iloc[0]))
    items.append(dict(id="breeding", group="Probabilities", title="Breeding heifer: next step", unit=unit_p,
        check="Per step, a breeding heifer conceives, leaves (culled or sold), or stays open. Is the conception level per step plausible for a heifer, and the leaving share small?",
        panels=[_panel("Fate of a breeding heifer in one step", pd.DataFrame({"conceives (pregnant)": [n.get("PREGNANT", 0.0)], "leaves (culled / sold)": [n.get("CULLED", 0.0) + n.get("SOLD", 0.0)],
                                                                          "stays open": [n.get("BREEDING", 0.0)]}, index=["Breeding heifer"]), "State", unit_p, "bars")]))

    g = st[(st["line"] == "DAIRY_F") & (st["stage"] == "PREGNANT")]
    items.append(dict(id="pregnant", group="Probabilities", title="Pregnant heifer by months pregnant", unit=unit_p,
        check="Calving should be zero until near term and then rise to 1; removal and pregnancy loss should be small and flat. Do they?",
        panels=[_panel("Pregnant heifer, probability per step", pd.DataFrame({"calves": pr["p_calve"][g.index], "removed (culled / sold)": pr["p_exit_pregnant"][g.index],
                                                                              "loses the pregnancy": pr["p_abort"][g.index]}, index=g["dp"].values), f"{TU.capitalize()}s pregnant", unit_p)]))

    states = st
    for k, prog in enumerate(PROGRAMS):
        mix = _birth_mix(states, prog)
        if k == 0:
            labels = (st["line"] + " " + st["stage"]).values
            tabs = {par: {} for par in mix}
        for par, vec in mix.items():
            tabs[par][prog] = pd.Series(vec).groupby(labels).sum()
    panels = []
    for par in ("1", "2", "3+"):
        df = pd.DataFrame(tabs[par]).fillna(0.0)
        df = df[(df > 0).any(axis=1)]
        panels.append(_panel(f"Dam parity {par}", df, "Outcome of a calving", "calves per calving event", "bars"))
    items.append(dict(id="birth_mix", group="Population", title="Birth mix by breeding program", unit="calves per calving event",
        check="What a calving produces: dairy heifer calves, dairy bull calves, beef-cross calves, stillbirths. Per event the total is a little above 1 (twins). With sexed semen the heifer share should rise; with beef semen the beef-cross share should. Is that what you see?",
        panels=panels))

    n_steps = int(round(3 * 365 / UD))
    fates, live = _cohort(P, st, np.asarray(pr["birth_mix"]["2"], float), n_steps)
    items.append(dict(id="cohort_fates", group="Population", title=f"Birth cohort over three years ({YP.BREEDING_PROGRAM} program)", unit="share of the birth cohort",
        check="One birth event of a parity-2 dam, followed for three years. Do the shares that end up sold, dead, culled and graduated to first calving look right?",
        panels=[_panel("Where the cohort is", fates[["alive"] + OUTCOMES], "Days since birth", "share", "area"),
                _panel("Live animals by line and stage", live, "Days since birth", "share", "area")]))

    mix = _birth_mix(states, "beef_on_dairy")
    _, live_b = _cohort(P, st, mix["2"], max(int(round(90 / UD)), 3))
    calves = live_b[[c for c in live_b.columns if c.endswith(" CALF") and not c.startswith("DAIRY_F")]]
    items.append(dict(id="calf_inventory", group="Population", title="Calves waiting to be sold (beef-on-dairy program)", unit="share of the birth cohort",
        check=f"How long bull and beef-cross calves stay on the farm after birth, for a farm using beef semen on cows. The sale age is {YP.CALF_SALE_AGE_DAYS:g} d; with {TU} steps the first point after birth is already {UD:g} d. Calves sold inside their own birth step never appear here. Does the curve fall to zero when the calves are sold?",
        panels=[_panel("Calves still on the farm", calves, "Days since birth", "share", "lines")]))
    return items

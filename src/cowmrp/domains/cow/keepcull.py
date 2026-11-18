"""cowmrp.domains.cow.keepcull -- keep / cull table for every cow of a herd.

Each cow is compared with a fresh replacement heifer bought now, over the SAME horizon and discount. This is
``cowmrp.domains.cow.value_comparisons.replacement_comparison`` applied to every state of a herd at once: the same two solves
(``replacement_net_values``: V_total and the present value of future replacement costs, per state) and the same convention; nothing is
re-derived here. The three quantities are kept apart:

    keep(s)     expected future value of the cow in state s, her future replacement costs included    = V_total(s) - PV_rc(s)
    replace(s)  value of replacing her now: a heifer in the replacement state h = (1, 0, 0), bought now = keep(h) - C
                (C = cost of one replacement, charged once at time 0)
    value_N     the keep-versus-cull comparison = keep(s) - replace = the BREAK-EVEN SALE VALUE over an N-year horizon
                [value = (V_total(s) - V_total(h)) - (PV_rc(s) - PV_rc(h)) + C, USD, expected, discounted]

Culling and replacing her now beats keeping her only if the sale brings more than value_N. The salvage of a sale is NOT credited in the
model's own comparison (a documented limitation: the salvage value of a sale is not part of the comparison) and no sale price is invented here; with ``offer`` (USD per cow) the table adds
a KEEP / CULL call per horizon: CULL when offer > value_N. Without an offer it only ranks the cows (rank 1 = lowest break-even = first to sell).
One solve per horizon values every state; the herd is a lookup.

To see the call built step by step, with the sale of the cow now as its own column (an indicative estimate, or the net offer you type for a cow), write the Excel
accounting workbook: ``cowmrp.domains.cow.keepcull_excel.write_keepcull_workbook`` or ``python runners/cow/farm/keepcull.py --excel``. It reuses the same solves, is a
reporting output only and changes no number of this table.

    from cowmrp.domains.cow.keepcull import keep_cull_table, keep_cull_summary
    table = keep_cull_table(herd, years=(1, 3, 5), offer=900)       # herd: DataFrame with a state_id column (model state of each cow)
    summary = keep_cull_summary(table, (1, 3, 5))

The herd's ``state_id`` values refer to the state table of the current time resolution (``COW_TIME_UNIT``).
``runners/cow/farm/keepcull.py`` calls this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def replacement_values(model, num_years=None, annual_discount=0.95):
    """The one solve behind every keep / replace comparison: ``replacement_net_values`` for one horizon (``num_years`` None = infinite),
    the replacement state ``rid`` and the cost of one replacement ``C``. keep(s) = ``net["V_keep"][s]``; the value of replacing now is
    ``replace = net["V_keep"][rid] - C`` (a number, the same for every cow); the comparison is keep(s) - replace.
    Used by ``keep_cull_table`` (finite horizons) and by the herd values of ``cowmrp.domains.cow.farm`` (infinite horizon)."""
    from cowmrp.mrp import core
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
    from cowmrp.domains.cow.value_comparisons import replacement_net_values
    from cowmrp.domains.cow.transitions import get_replacement_state_id
    gamma = core.annual_to_per_step_gamma(annual_discount, UNIT_DAYS)
    net = replacement_net_values(model.states, model.P, model.rewards, model.probs, gamma, num_years=num_years)
    rid = get_replacement_state_id(model.states)
    C = float(cost_per_replacement())
    return {"net": net, "rid": rid, "C": C, "replace": net["V_keep"][rid] - C}


def keep_cull_table(herd, model=None, years=(1, 3, 5), annual_discount=0.95, offer=None, detail=False):
    """One row per cow of ``herd``; ``value_<N>y`` (break-even sale value) for each horizon in ``years``, ``decision_<N>y`` when an
    ``offer`` (USD per cow) is given, ``rank_longest`` (1 = weakest cow, by the longest horizon) and, with an offer, ``stable``
    (the call is the same at every horizon).

    ``herd``: DataFrame with ``state_id``; ``cow_id``, ``lactation``, ``dim_days``, ``dp_days`` are copied when present.
    ``model``: a built ``cowmrp.model.Model`` of the cow (built here when None). ``detail=True`` adds ``keep_<N>y`` (expected future
    value of the cow) and ``replace_<N>y`` (value of the replacement heifer bought now) beside ``value_<N>y``; the default output is unchanged.
    """
    from cowmrp.model import build_model

    if "state_id" not in herd:
        raise ValueError("herd needs a 'state_id' column (the model state of each cow)")
    model = model or build_model("cow")
    states = model.states
    years = [int(y) for y in years]
    if not years or min(years) < 1:
        raise ValueError("years must be a non-empty list of whole years >= 1")
    sid = herd["state_id"].to_numpy()
    if len(sid) and (sid.min() < 0 or sid.max() >= len(states)):
        raise ValueError(f"herd state_id outside 0..{len(states) - 1}: the herd was built for another time resolution or state table")
    cols = [c for c in ("cow_id", "state_id", "lactation", "dim_days", "dp_days") if c in herd]
    out = herd[cols].copy()
    for y in years:
        rv = replacement_values(model, y, annual_discount)
        keep = rv["net"]["V_keep"]
        out[f"value_{y}y"] = keep[sid] - rv["replace"]
        if detail:
            out[f"keep_{y}y"] = keep[sid]
            out[f"replace_{y}y"] = rv["replace"]
        if offer is not None:
            out[f"decision_{y}y"] = np.where(offer > out[f"value_{y}y"], "CULL", "KEEP")
    last = f"value_{years[-1]}y"
    out["rank_longest"] = out[last].rank(method="first").astype(int)         # 1 = weakest cow
    if offer is not None:
        out["stable"] = out[[f"decision_{y}y" for y in years]].nunique(axis=1) == 1
    return out


def keep_cull_summary(table, years):
    """Value spread per horizon (all cows and by lactation) and, when the table has decisions, the number and share of cows to cull."""
    rows = []
    for y in years:
        for lac, g in [("all", table)] + [(str(k), t) for k, t in table.groupby("lactation")]:
            v = g[f"value_{y}y"]
            row = {"horizon_years": y, "lactation": lac, "cows": len(g), "min_value": float(v.min()),
                   "p10_value": float(v.quantile(0.1)), "mean_value": float(v.mean())}
            if f"decision_{y}y" in g:
                row["cull"] = int((g[f"decision_{y}y"] == "CULL").sum())
                row["cull_pct"] = 100 * float((g[f"decision_{y}y"] == "CULL").mean())
            rows.append(row)
    return pd.DataFrame(rows)

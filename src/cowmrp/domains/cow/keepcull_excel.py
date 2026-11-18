"""cowmrp.domains.cow.keepcull_excel -- Excel workbook that shows the keep-versus-cull accounting of every cow, step by step.

A reporting output only. It reads the same solves as ``cowmrp.domains.cow.keepcull`` (``replacement_values``) and changes nothing in the model:
the MRP reward vector is untouched and the future cull salvage stays inside it. The one addition is the sale of the CURRENT cow now, which the
model does not credit (a documented limitation); here it is a separate, visible column.

    KEEP          = V_total(s) - PV_rc(s)                  expected future value of the cow, her future replacement costs paid
    REPLACE       = S + V_keep(h) - C                      S = sale of the cow now, h = replacement heifer state (1, 0, 0), C = rearing cost (once, now)
    Delta Q       = KEEP - REPLACE
    break-even S  = KEEP - V_keep(h) + C                   the sale value at which keeping and replacing are equal (= ``value_<N>y`` of keep_cull_table)
    decision      = CULL when S > break-even, else KEEP

S is the cow's own estimate (body weight of the state x dressing percentage x beef price - culling fee, the same inputs as ``cull_reward``),
or, when a net offer is typed for that cow, the offer (net cash received; the culling fee is NOT subtracted again). The estimate is indicative:
it uses the model's state-based body weight, not the weight on the day of sale.

The workbook keeps the model's numbers (V_total, PV_rc, V_keep(h): from the matrix solves) as values and everything else as Excel formulas, so every
column can be inspected and the inputs (prices, fee, offers) changed.

    from cowmrp.domains.cow.keepcull_excel import write_keepcull_workbook
    write_keepcull_workbook(herd, "keepcull_accounting.xlsx", model, years=(1, 3, 5))      # herd: DataFrame with a state_id column
"""
from __future__ import annotations

import numpy as np

HEADERS = ["cow_id", "state_id", "lactation", "dim_days", "dp_days", "body_weight_kg (model, state-based)", "V_total(s)", "PV future replacement costs (s)",
           "KEEP = V_total - PV_rc", "estimated salvage S_est", "actual net offer (type here, optional)", "S used", "V_keep(heifer)",
           "rearing cost C", "REPLACE = S + V_keep(h) - C", "Delta Q = KEEP - REPLACE", "break-even offer", "decision", "S source",
           "value from keep_cull_table (check)", "difference to check"]

# One entry per column of the Cows_<N>y sheets, in the order of HEADERS: (column letter, kind, formula of row {r} or None, meaning).
# kind: "herd" = copied from the herd file; "model" = number pasted from the matrix solves; "input" = typed by the user; "formula" = Excel formula; "check" = reference value / difference.
# It is documentation data (the public workbook page and the README table are generated from it); tests/integration/test_cow_keepcull_excel.py compares it with the cells that are written.
WORKBOOK_COLUMNS = (
    ("A", "herd", None, "Cow identifier from the herd file (row number if the herd has none)."),
    ("B", "herd", None, "Model state of the cow (index into the state table of the current time resolution)."),
    ("C", "herd", None, "Lactation number (copied when the herd has it)."),
    ("D", "herd", None, "Days in milk (copied when the herd has it)."),
    ("E", "herd", None, "Days pregnant (copied when the herd has it)."),
    ("F", "model", None, "Body weight of the state in kg, computed as cull_reward computes it for the salvage value. A model estimate, not the weight on the day of sale."),
    ("G", "model", None, "V_total(s): expected discounted total reward of the cow over the horizon (milk, pregnancy, cull, mortality, feed, reproduction), future cull salvage included."),
    ("H", "model", None, "PV_rc(s): expected discounted cost of the future replacements of this cow and of her replacements over the horizon."),
    ("I", "formula", "=G{r}-H{r}", "KEEP: expected future value of keeping the cow, every future replacement cost paid."),
    ("J", "formula", "=F{r}*Inputs!$B$3*Inputs!$B$2-Inputs!$B$4", "Estimated salvage now: body weight x dressing percentage x beef price - culling fee (Inputs B3, B2, B4)."),
    ("K", "input", None, "Actual net offer for this cow, typed by the user (optional). Net cash received: the culling fee is not subtracted again."),
    ("L", "formula", '=IF(K{r}="",J{r},K{r})', "Salvage used: the typed offer if there is one, otherwise the estimate."),
    ("M", "formula", "=Inputs!$E${h}", "V_keep(h): value of a fresh replacement heifer (state (1, 0, 0)) over the same horizon (Inputs sheet)."),
    ("N", "formula", "=Inputs!$B${h}", "C: rearing cost of one replacement, charged once, now (Inputs sheet)."),
    ("O", "formula", "=L{r}+M{r}-N{r}", "REPLACE: salvage used + value of the replacement heifer - rearing cost."),
    ("P", "formula", "=I{r}-O{r}", "Delta Q = KEEP - REPLACE. Positive: keeping is better; negative: replacing is better."),
    ("Q", "formula", "=I{r}-M{r}+N{r}", "Break-even offer: the salvage at which KEEP and REPLACE are equal."),
    ("R", "formula", '=IF(L{r}>Q{r},"CULL","KEEP")', "Decision: CULL when the salvage used is above the break-even offer, otherwise KEEP."),
    ("S", "formula", '=IF(K{r}="","estimate","offer")', "Which salvage was used: the estimate or a typed offer."),
    ("T", "check", None, "value_<N>y from keep_cull_table for the same cow and horizon (pasted from the model)."),
    ("U", "check", "=Q{r}-T{r}", "Break-even offer minus the reference value; must be 0 (to rounding)."),
)

README = [
    "Keep versus cull: accounting workbook (reporting only; the MRP reward vector is unchanged)",
    "",
    "KEEP     = V_total(s) - PV_rc(s)           expected discounted future value of the cow; the cost of every future replacement is paid.",
    "REPLACE  = S + V_keep(h) - C               S: sale of the cow now. V_keep(h): value of a fresh replacement heifer (state (1,0,0)) over the same horizon. C: rearing cost, charged once, now.",
    "Delta Q  = KEEP - REPLACE                  positive: keeping is better; negative: replacing is better.",
    "Break-even offer = KEEP - V_keep(h) + C    the net sale value at which the two are equal.",
    "Decision: CULL when S used > break-even offer, otherwise KEEP.",
    "",
    "S used: the actual net offer typed for that cow (column K), otherwise the estimate S_est. A typed offer is the net cash received: the culling fee is not subtracted again.",
    "S_est = body weight x dressing percentage x beef price - culling fee. The body weight is the model's state-based weight (not the weight on the sale day): indicative only.",
    "",
    "What is a model value (pasted, from the matrix solves): V_total(s), PV_rc(s), V_total(h), PV_rc(h), body weight. Everything else is a formula; change the Inputs sheet or the offer column and it recalculates.",
    "Future cull salvage and future replacement costs of the cow and of her replacements stay inside V_total and PV_rc (the cow's lineage). The sale of the CURRENT cow is the only salvage added here, once, on the REPLACE side.",
    "The model's own comparison (keep_cull_table, replacement_comparison) does not credit the sale of the current cow (a documented limitation); this workbook does not change that or the model's defaults.",
    "Discounting and the horizon are those of the model (annual discount on the Inputs sheet; horizon in the sheet name). Values are expected values, USD.",
    "The Checks sheet compares the break-even column with the existing keep_cull_table values: the difference must be zero.",
]


def _salvage_inputs():
    from cowmrp.domains.cow.rewards.base import BEEF_PRICE_PER_KG, DRESSING_PCT, CULLING_COST
    return BEEF_PRICE_PER_KG, DRESSING_PCT, CULLING_COST


def body_weight_of_states(states, sid):
    """Body weight (kg) of the states ``sid``, exactly as ``cull_reward`` computes it for the salvage value."""
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.rewards.base import body_weight_kg, age_days_of, mid_step_dim_days
    s = states.iloc[sid]
    lac, dim, dp = s["lactation"].to_numpy(), s["dim"].to_numpy(), s["dp"].to_numpy()
    return np.asarray(body_weight_kg(age_days_of(lac, dim), mid_step_dim_days(dim), dp * UNIT_DAYS), dtype=float)


def write_keepcull_workbook(herd, path, model=None, years=(1, 3, 5), annual_discount=0.95):
    """Write the accounting workbook for every cow of ``herd`` (DataFrame with ``state_id``; ``cow_id``, ``lactation``, ``dim_days``, ``dp_days`` are copied
    when present) and the horizons ``years``. Returns ``path``. Needs openpyxl (``pip install cowmrp[excel]``)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from cowmrp.model import build_model
    from cowmrp.domains.cow.keepcull import keep_cull_table, replacement_values

    if "state_id" not in herd:
        raise ValueError("herd needs a 'state_id' column (the model state of each cow)")
    model = model or build_model("cow")
    years = [int(y) for y in years]
    sid = herd["state_id"].to_numpy()
    if len(sid) and (sid.min() < 0 or sid.max() >= len(model.states)):
        raise ValueError(f"herd state_id outside 0..{len(model.states) - 1}: the herd was built for another time resolution or state table")
    price, dressing, fee = _salvage_inputs()
    bw = body_weight_of_states(model.states, sid)
    ref = keep_cull_table(herd, model, years, annual_discount)

    bold, head_fill = Font(bold=True), PatternFill("solid", fgColor="DDEBF7")
    input_fill, model_fill = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor="EDEDED")
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for i, line in enumerate(README, 1):
        ws.cell(i, 1, line)
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 160

    inp = wb.create_sheet("Inputs")
    inp["A1"], inp["B1"], inp["C1"] = "Input", "Value", "Meaning / source in the model"
    rows = [("Beef price per kg dressed (USD)", price, "BEEF_PRICE_PER_KG (rewards/base.py), the price cull_reward uses"),
            ("Dressing percentage", dressing, "DRESSING_PCT (rewards/base.py)"),
            ("Culling fee (USD)", fee, "CULLING_COST (rewards/base.py); subtracted from the estimate only, never from a typed offer"),
            ("Annual discount", annual_discount, "as used for every horizon below (model solve; changing it here does not recompute V)")]
    for i, (a, b, c) in enumerate(rows, 2):
        inp.cell(i, 1, a), inp.cell(i, 2, b), inp.cell(i, 3, c)
        inp.cell(i, 2).fill = input_fill if i < 5 else model_fill
    inp["A7"], inp["B7"], inp["C7"], inp["D7"], inp["E7"] = "Horizon (years)", "C: rearing cost (once, now)", "V_total(h)", "PV_rc(h)", "V_keep(h) = V_total(h) - PV_rc(h)"
    horizon_row = {}
    for k, y in enumerate(years):
        r = 8 + k
        rv = replacement_values(model, y, annual_discount)
        h = rv["rid"]
        inp.cell(r, 1, y), inp.cell(r, 2, rv["C"])
        inp.cell(r, 3, float(rv["net"]["V_total"][h])), inp.cell(r, 4, float(rv["net"]["PV_replacement_cost"][h]))
        inp.cell(r, 5, f"=C{r}-D{r}")
        for c in (2, 3, 4):
            inp.cell(r, c).fill = model_fill
        horizon_row[y] = r
    inp.cell(9 + len(years), 1, "Grey = value from the model; yellow = input you may change; white = formula.")
    from cowmrp.domains.cow.economics import active_record
    rec = active_record()
    if rec and rec["overrides"]:                    # only inside an economic_overrides block: say so, in the workbook itself
        r0 = 11 + len(years)
        inp.cell(r0, 1, "NON-DEFAULT ECONOMIC SETTINGS (scenario run: results are not the package defaults)").font = bold
        for j, t in enumerate(("Setting", "Default", "Value used", "Unit", "Source recorded in the repository"), 1):
            inp.cell(r0 + 1, j, t).font = bold
        for i, (n, o) in enumerate(rec["overrides"].items(), r0 + 2):
            for j, v in enumerate((n, o["default"], o["value"], f'{o["unit"]} ({o["currency"]})', o["source"]), 1):
                inp.cell(i, j, v)
        for k, note in enumerate(rec["notes"], r0 + 2 + len(rec["overrides"])):
            inp.cell(k, 1, "Note: " + note)
    for cell in inp[1] + inp[7]:
        cell.font, cell.fill = bold, head_fill
    for col, w in zip("ABCDE", (38, 28, 70, 14, 36)):
        inp.column_dimensions[col].width = w

    n = len(herd)
    last = n + 1
    for y in years:
        rv = replacement_values(model, y, annual_discount)
        vt, pv = rv["net"]["V_total"][sid], rv["net"]["PV_replacement_cost"][sid]
        sh = wb.create_sheet(f"Cows_{y}y")
        for j, t in enumerate(HEADERS, 1):
            c = sh.cell(1, j, t)
            c.font, c.fill, c.alignment = bold, head_fill, Alignment(wrap_text=True, vertical="top")
        hr = horizon_row[y]
        for i in range(n):
            r = i + 2
            sh.cell(r, 1, herd["cow_id"].iloc[i].item() if "cow_id" in herd else i + 1)
            sh.cell(r, 2, int(sid[i]))
            for col, name in ((3, "lactation"), (4, "dim_days"), (5, "dp_days")):
                if name in herd:
                    sh.cell(r, col, herd[name].iloc[i].item())
            sh.cell(r, 6, float(bw[i]))
            sh.cell(r, 7, float(vt[i]))
            sh.cell(r, 8, float(pv[i]))
            sh.cell(r, 9, f"=G{r}-H{r}")
            sh.cell(r, 10, f"=F{r}*Inputs!$B$3*Inputs!$B$2-Inputs!$B$4")
            sh.cell(r, 12, f'=IF(K{r}="",J{r},K{r})')
            sh.cell(r, 13, f"=Inputs!$E${hr}")
            sh.cell(r, 14, f"=Inputs!$B${hr}")
            sh.cell(r, 15, f"=L{r}+M{r}-N{r}")
            sh.cell(r, 16, f"=I{r}-O{r}")
            sh.cell(r, 17, f"=I{r}-M{r}+N{r}")
            sh.cell(r, 18, f'=IF(L{r}>Q{r},"CULL","KEEP")')
            sh.cell(r, 19, f'=IF(K{r}="","estimate","offer")')
            sh.cell(r, 20, float(ref[f"value_{y}y"].iloc[i]))
            sh.cell(r, 21, f"=Q{r}-T{r}")
            sh.cell(r, 11).fill = input_fill
            for c in (6, 7, 8, 20):
                sh.cell(r, c).fill = model_fill
        sh.freeze_panes = "C2"
        sh.row_dimensions[1].height = 62
        for j in range(1, len(HEADERS) + 1):
            sh.column_dimensions[get_column_letter(j)].width = 16

    ck = wb.create_sheet("Checks")
    ck.append(["Horizon (years)", "cows", "sum |break-even - keep_cull_table value|", "cows to CULL (with the offers entered)", "must be"])
    for c in ck[1]:
        c.font, c.fill = bold, head_fill
    for k, y in enumerate(years):
        r = 2 + k
        s = f"Cows_{y}y"
        ck.append([y, f"=COUNTA({s}!B2:B{last})", f"=SUMPRODUCT(ABS({s}!U2:U{last}))", f'=COUNTIF({s}!R2:R{last},"CULL")', "difference = 0 (to rounding)"])
    ck.cell(3 + len(years), 1, "The break-even column is the existing value_<N>y of keep_cull_table; Delta Q = value - S used, so Delta Q < 0 exactly when the call is CULL.")
    for col, w in zip("ABCDE", (16, 10, 40, 36, 30)):
        ck.column_dimensions[col].width = w

    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wb.save(path)
    return path

"""Keep / cull accounting workbook (cowmrp.domains.cow.keepcull_excel): the formulas it writes, evaluated here, reproduce
value_comparisons.replacement_comparison and keep_cull_table; offers take precedence over the estimate; the model is not touched."""
import re

import numpy as np
import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")
from cowmrp.domains.cow.keepcull import keep_cull_table
from cowmrp.domains.cow.keepcull_excel import write_keepcull_workbook

GAMMA = 0.95


def _eval(wb, sheet, row):
    """Evaluate the (simple) formulas of one cow row: cell references, + - * and IF(...)."""
    ws = wb[sheet]
    cache = {}

    def val(ref, ws_=ws):
        if isinstance(ref, str) and "!" in ref:
            sh, cell = ref.replace("$", "").split("!")
            return get(wb[sh], cell)
        return get(ws_, ref.replace("$", ""))

    def get(w, cell):
        key = (w.title, cell)
        if key in cache:
            return cache[key]
        v = w[cell].value
        if isinstance(v, str) and v.startswith("="):
            v = ev(v[1:], w)
        cache[key] = v
        return v

    def ev(f, w):
        m = re.fullmatch(r'IF\((\w+)="",(\w+),(\w+)\)', f)
        if m:
            a = get(w, m.group(1))
            return get(w, m.group(2)) if a in (None, "") else get(w, m.group(3))
        m = re.fullmatch(r'IF\((\w+)>(\w+),"CULL","KEEP"\)', f)
        if m:
            return "CULL" if get(w, m.group(1)) > get(w, m.group(2)) else "KEEP"
        m = re.fullmatch(r'IF\((\w+)="","estimate","offer"\)', f)
        if m:
            return "estimate" if get(w, m.group(1)) in (None, "") else "offer"
        expr = re.sub(r"(Inputs!)?\$?([A-Z]+)\$?(\d+)", lambda mm: repr(get(wb["Inputs"] if mm.group(1) else w, f"{mm.group(2)}{mm.group(3)}")), f)
        return eval(expr, {"__builtins__": {}})

    return {c.column_letter: get(ws, c.coordinate) for c in ws[row]}


def test_workbook_reproduces_replacement_comparison(cow, tmp_path):
    from cowmrp.mrp import core
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.rewards.base import BEEF_PRICE_PER_KG, DRESSING_PCT, CULLING_COST
    from cowmrp.domains.cow.value_comparisons import replacement_comparison, replacement_net_values
    _, m = cow
    sid = [int(s) for s in np.linspace(5, len(m.states) - 5, 12)]
    herd = pd.DataFrame({"state_id": sid})
    p = write_keepcull_workbook(herd, tmp_path / "w.xlsx", m, years=(1, 3), annual_discount=GAMMA)
    wb = openpyxl.load_workbook(p)
    assert wb.sheetnames == ["README", "Inputs", "Cows_1y", "Cows_3y", "Checks"]
    gamma = core.annual_to_per_step_gamma(GAMMA, UNIT_DAYS)
    ref = keep_cull_table(herd, m, (1, 3), GAMMA)
    for y in (1, 3):
        net = replacement_net_values(m.states, m.P, m.rewards, m.probs, gamma, y)
        for i, s in enumerate(sid):
            st = m.states.iloc[s]
            rc = replacement_comparison(m.states, m.P, m.rewards, m.probs, gamma, int(st.lactation), int(st.dim), int(st.dp), num_years=y, net=net)
            r = _eval(wb, f"Cows_{y}y", i + 2)
            est = r["F"] * DRESSING_PCT * BEEF_PRICE_PER_KG - CULLING_COST
            assert r["I"] == pytest.approx(rc["V_state"], rel=1e-12, abs=1e-9)                       # KEEP
            assert r["M"] - r["N"] == pytest.approx(rc["V_comparison"], rel=1e-12, abs=1e-9)         # V_keep(h) - C
            assert r["J"] == pytest.approx(est) and r["L"] == pytest.approx(est) and r["S"] == "estimate"
            assert r["O"] == pytest.approx(est + rc["V_comparison"], rel=1e-12, abs=1e-9)            # REPLACE
            assert r["P"] == pytest.approx(rc["value"] - est, rel=1e-12, abs=1e-9)                   # Delta Q = value - S
            assert r["Q"] == pytest.approx(rc["value"], rel=1e-12, abs=1e-9)                         # break-even = value_N
            assert r["Q"] == pytest.approx(ref[f"value_{y}y"].iloc[i], rel=1e-12, abs=1e-9)
            assert abs(r["U"]) < 1e-9
            assert r["R"] == ("CULL" if r["P"] < 0 else "KEEP")


def test_typed_offer_replaces_the_estimate_without_the_fee(cow, tmp_path):
    _, m = cow
    p = write_keepcull_workbook(pd.DataFrame({"state_id": [100]}), tmp_path / "w.xlsx", m, years=(1,))
    wb = openpyxl.load_workbook(p)
    base = _eval(wb, "Cows_1y", 2)
    wb["Cows_1y"]["K2"] = base["Q"] + 50.0                       # a net offer 50 above the break-even
    r = _eval(wb, "Cows_1y", 2)
    assert r["L"] == base["Q"] + 50.0 and r["S"] == "offer" and r["R"] == "CULL"       # used as typed: no culling fee subtracted
    assert r["P"] == pytest.approx(-50.0)
    wb["Cows_1y"]["K2"] = base["Q"] - 50.0
    assert _eval(wb, "Cows_1y", 2)["R"] == "KEEP"


def test_writing_the_workbook_leaves_the_table_unchanged(cow, tmp_path):
    _, m = cow
    herd = pd.DataFrame({"state_id": [10, 200, 400]})
    before = keep_cull_table(herd, m, (1, 3), GAMMA, 900)
    write_keepcull_workbook(herd, tmp_path / "w.xlsx", m, years=(1, 3))
    pd.testing.assert_frame_equal(before, keep_cull_table(herd, m, (1, 3), GAMMA, 900))


def test_documented_columns_match_the_written_workbook(cow, tmp_path):
    """WORKBOOK_COLUMNS (the source of the public documentation) must describe the cells that are really written: same count and headers, same formulas."""
    openpyxl = pytest.importorskip("openpyxl")
    from cowmrp.domains.cow.keepcull_excel import HEADERS, WORKBOOK_COLUMNS
    _, m = cow
    herd = pd.DataFrame({"state_id": [100, 200], "cow_id": [7, 8], "lactation": [1, 2], "dim_days": [30, 60], "dp_days": [0, 0]})
    wb = openpyxl.load_workbook(write_keepcull_workbook(herd, tmp_path / "w.xlsx", m, years=(1, 3)))
    assert len(WORKBOOK_COLUMNS) == len(HEADERS) == 21
    inputs = wb["Inputs"]
    for y, hr in ((1, 8), (3, 9)):
        assert inputs.cell(hr, 1).value == y                                        # the {h} row of the Inputs sheet is the horizon's row
        ws = wb[f"Cows_{y}y"]
        for j, (letter, kind, formula, meaning) in enumerate(WORKBOOK_COLUMNS, 1):
            assert openpyxl.utils.get_column_letter(j) == letter and meaning
            assert ws.cell(1, j).value == HEADERS[j - 1]
            cell = ws.cell(2, j).value
            if kind == "formula" or formula:
                assert cell == formula.format(r=2, h=hr), (letter, cell, formula)
            else:
                assert not (isinstance(cell, str) and cell.startswith("=")), (letter, cell)
    assert [wb[s].title for s in wb.sheetnames] == ["README", "Inputs", "Cows_1y", "Cows_3y", "Checks"]

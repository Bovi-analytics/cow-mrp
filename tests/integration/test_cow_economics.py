"""Economic overrides (cowmrp.domains.cow.economics): defaults untouched, overrides temporary and restored, only the intended rewards move,
the transition matrix and probabilities never change, KEEP/CULL reads REARING_COST, unsupported names are refused."""
import importlib
import pkgutil
import subprocess
import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
import cowmrp
from cowmrp.mrp.utils import matrix_checksum
from cowmrp.domains.cow import economics
from cowmrp.domains.cow.economics import (ECONOMIC_SETTINGS, DEFAULTS, HOLDERS, EconomicSettingsWarning, active_record, economic_overrides,
                                          reprice_model)
from cowmrp.domains.cow.keepcull import keep_cull_table, replacement_values

BASE = "cowmrp.domains.cow.rewards.base"
SIGN = {"milk_income": 1, "pregnancy_reward": 1, "cull_reward": 1, "mortality_cost": -1, "feed_cost": -1, "reproduction_cost": -1}
COMPONENTS = tuple(SIGN)
# which reward components may change for each setting (everything else must be bit-identical); REARING_COST: none
EXPECTED = {"MILK_PRICE_PER_KG": {"milk_income"}, "FEED_PRICE_PER_KG_DM": {"feed_cost"}, "DRY_PERIOD_DAILY_COST": {"feed_cost"},
            "CALF_VALUE": {"pregnancy_reward"}, "BEEF_PRICE_PER_KG": {"cull_reward"}, "DRESSING_PCT": {"cull_reward"}, "CULLING_COST": {"cull_reward"},
            "DISPOSAL_COST": {"mortality_cost"}, "AI_COST": {"reproduction_cost"}, "DIAGNOSIS_COST": {"reproduction_cost"},
            "SYNCHRONIZATION_COST": {"reproduction_cost"}, "ABORTION_COST": {"reproduction_cost"},
            "DAYS_OPEN_PENALTY_PER_DAY": {"reproduction_cost"}, "REARING_COST": set()}


def _numeric_keys(rewards):
    return [k for k in rewards if not k.endswith("_method")]


def _same(a, b, keys):
    return all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float)) for k in keys)


def _held_values():
    """(module, name) -> current value, for every module that holds a copy of an economic constant."""
    return {(m, n): getattr(importlib.import_module(m), n) for n, mods in HOLDERS.items() for m in (BASE,) + mods}


# ---- coverage and refusal -------------------------------------------------------------------------------------------------------------------------

def test_every_economic_constant_is_covered_and_registry_describes_the_defaults():
    base = importlib.import_module(BASE)
    assert set(EXPECTED) == set(HOLDERS) == set(ECONOMIC_SETTINGS)
    for name, s in ECONOMIC_SETTINGS.items():
        assert s["default"] == float(getattr(base, name)) == DEFAULTS[name]        # the registry describes rewards/base.py, it is not its source
        assert s["currency"] == "USD" and s["unit"] and s["source"]


def test_no_module_holds_an_uncovered_copy():
    """Scan every module of the package: a module that has an attribute named like an economic constant must be listed in HOLDERS (or be rewards.base)."""
    for info in pkgutil.walk_packages(cowmrp.__path__, "cowmrp."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue                                                               # optional dependencies (plotting etc.)
        for name, holders in HOLDERS.items():
            if hasattr(mod, name) and info.name != BASE:
                assert info.name in holders, f"{info.name} holds a copy of {name} that economic_overrides would not replace"


def test_unsupported_and_invalid_values_are_refused_before_anything_changes():
    before = _held_values()
    for bad in ({"TIME_UNIT": "week"}, {"GESTATION_LENGTH_DAYS": 280}, {"CONCEPTION_RATE": 0.5}, {"MILK_PRICE": 0.5}, {"REPLACEMENT_STRATEGY": "buy"},
                {"PURCHASE_COST": 2000.0}, {"MILK_PRICE_PER_KG": float("nan")}, {"MILK_PRICE_PER_KG": "high"}, {"MILK_PRICE_PER_KG": True},
                {"REARING_COST": -1.0}, {"DRESSING_PCT": 0.0}, {"DRESSING_PCT": 1.2}):
        with pytest.raises(ValueError):
            with economic_overrides(**bad):
                pass
    with pytest.raises(ValueError, match="MILK_PRICE_PER_KG"):                      # a typo suggests the nearest name
        with economic_overrides(MILK_PRICE=0.5):
            pass
    assert _held_values() == before and active_record() is None


# ---- restoration ----------------------------------------------------------------------------------------------------------------------------------

def test_values_are_restored_after_the_block_and_after_an_exception():
    before = _held_values()
    over = {n: v * 1.5 for n, v in DEFAULTS.items() if v}
    over["DRESSING_PCT"] = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EconomicSettingsWarning)
        with economic_overrides(**over):
            assert _held_values() != before and active_record() is not None
        assert _held_values() == before and active_record() is None
        with pytest.raises(KeyError):
            with economic_overrides(**over):
                raise KeyError("boom")
    assert _held_values() == before and active_record() is None


def test_nested_blocks_are_refused_and_do_not_leak():
    before = _held_values()
    with economic_overrides(MILK_PRICE_PER_KG=0.6):
        with pytest.raises(RuntimeError, match="nested"):
            with economic_overrides(CALF_VALUE=1000.0):
                pass
        assert importlib.import_module(BASE).CALF_VALUE == DEFAULTS["CALF_VALUE"]
        assert importlib.import_module(BASE).MILK_PRICE_PER_KG == 0.6
    assert _held_values() == before


# ---- default parity ---------------------------------------------------------------------------------------------------------------------------------

def test_default_rewards_are_bit_identical_with_and_without_the_module(cow):
    _, m = cow
    keys = _numeric_keys(m.rewards)
    assert _same(reprice_model(m).rewards, m.rewards, keys)                        # no block: unchanged
    with economic_overrides():                                                     # an empty block: unchanged
        assert _same(reprice_model(m).rewards, m.rewards, keys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EconomicSettingsWarning)
        with economic_overrides(MILK_PRICE_PER_KG=0.9, REARING_COST=3000.0):
            pass
    assert _same(reprice_model(m).rewards, m.rewards, keys)                        # after a block: back to the defaults


# ---- only the intended components move; P and probabilities never change ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_override_changes_only_its_reward_components(cow, name):
    _, m = cow
    keys = _numeric_keys(m.rewards)
    x = DEFAULTS[name] or 4.0                                                      # DAYS_OPEN_PENALTY default is 0: use a test value
    if name == "DRESSING_PCT":
        pts = [0.30, 0.40, 0.50]
    else:
        pts = [0.0, x, 2 * x]
    out, chk0 = [], matrix_checksum(m.P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EconomicSettingsWarning)
        for v in pts:
            with economic_overrides(**{name: v}):
                s = reprice_model(m)
                assert s.states is m.states and s.probs is m.probs and s.P is m.P           # same objects: nothing biological can change
                out.append(s.rewards)
    assert matrix_checksum(m.P) == chk0
    changed = set(EXPECTED[name]) | ({"total"} if EXPECTED[name] else set())
    for k in keys:
        if k in changed:
            continue
        assert all(np.array_equal(np.asarray(r[k], float), np.asarray(out[0][k], float)) for r in out), f"{name} changed {k}"
    for k in EXPECTED[name]:                                                       # affine in the constant: f(b) - f(a) over equal steps
        a, b, c = (np.asarray(r[k], float) for r in out)
        assert np.allclose(c - b, b - a, rtol=1e-9, atol=1e-9) and not np.allclose(a, c)
    if EXPECTED[name]:                                                             # total = signed sum of the components (every state)
        for r in out:
            assert np.allclose(np.asarray(r["total"], float), sum(SIGN[k] * np.asarray(r[k], float) for k in COMPONENTS), rtol=1e-12, atol=1e-9)
    else:
        for r in out:
            assert _same(r, m.rewards, keys)                                       # REARING_COST: the reward vector is untouched


def test_milk_price_scales_milk_income_exactly(cow):
    _, m = cow
    with economic_overrides(MILK_PRICE_PER_KG=2 * DEFAULTS["MILK_PRICE_PER_KG"]):
        s = reprice_model(m)
    assert np.allclose(np.asarray(s.rewards["milk_income"], float), 2 * np.asarray(m.rewards["milk_income"], float), rtol=1e-12)


def test_cull_reward_matches_its_formula_under_overrides(cow):
    """cull_reward = w * (body weight * dressing * beef price - fee), checked against an independent weight and the weight w recovered from the vector."""
    from cowmrp.domains.cow.keepcull_excel import body_weight_of_states
    _, m = cow
    sid = np.arange(len(m.states))
    bw = body_weight_of_states(m.states, sid)
    with economic_overrides(BEEF_PRICE_PER_KG=0.0, CULLING_COST=1.0):
        w = -np.asarray(reprice_model(m).rewards["cull_reward"], float)           # = weight of a cull in every state
    with economic_overrides(BEEF_PRICE_PER_KG=3.1, DRESSING_PCT=0.5, CULLING_COST=75.0):
        got = np.asarray(reprice_model(m).rewards["cull_reward"], float)
    assert np.allclose(got, w * (bw * 0.5 * 3.1 - 75.0), rtol=1e-10, atol=1e-9)


# ---- KEEP/CULL --------------------------------------------------------------------------------------------------------------------------------------

def _herd(m, n=40):
    return pd.DataFrame({"state_id": np.linspace(3, len(m.states) - 3, n).astype(int)})


def test_keepcull_default_is_unchanged_and_rearing_cost_acts_only_in_the_keepcull_layer(cow):
    from cowmrp.mrp import core
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.value_comparisons import replacement_comparison, replacement_net_values
    _, m = cow
    herd, years = _herd(m), (1, 3)
    base = keep_cull_table(herd, m, years, 0.95, 900)
    vals = {}
    for c in (0.0, 2450.0, 4900.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EconomicSettingsWarning)
            with economic_overrides(REARING_COST=c):
                s = reprice_model(m)
                assert _same(s.rewards, m.rewards, _numeric_keys(m.rewards))              # not in the reward vector
                t = keep_cull_table(herd, s, years, 0.95, 900)
                vals[c] = t
                gamma = core.annual_to_per_step_gamma(0.95, UNIT_DAYS)
                net = replacement_net_values(s.states, s.P, s.rewards, s.probs, gamma, 3)
                for sid, row in zip(herd["state_id"].iloc[:10], t.itertuples()):
                    st = s.states.iloc[sid]
                    rc = replacement_comparison(s.states, s.P, s.rewards, s.probs, gamma, int(st.lactation), int(st.dim), int(st.dp), num_years=3, net=net)
                    assert row.value_3y == pytest.approx(rc["value"], rel=1e-12, abs=1e-9)
                    assert rc["replacement_cost_now"] == c
    pd.testing.assert_frame_equal(vals[2450.0], base)                                     # the default value passed explicitly = the default result
    for y in years:                                                                       # the break-even is affine in C
        a, b, c = (vals[k][f"value_{y}y"].to_numpy() for k in (0.0, 2450.0, 4900.0))
        assert np.allclose(c - b, b - a, rtol=1e-9, atol=1e-7) and not np.allclose(a, c)
    pd.testing.assert_frame_equal(keep_cull_table(herd, m, years, 0.95, 900), base)       # after the blocks: the default again
    assert replacement_values(m, 3, 0.95)["C"] == 2450.0


def test_matrix_and_probabilities_unchanged_after_overrides(cow):
    _, m = cow
    chk, probs = matrix_checksum(m.P), {k: np.array(v, float, copy=True) for k, v in m.probs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EconomicSettingsWarning)
        with economic_overrides(**{n: v * 1.3 for n, v in DEFAULTS.items() if v}):
            reprice_model(m)
    assert matrix_checksum(m.P) == chk and all(np.array_equal(probs[k], np.asarray(m.probs[k], float)) for k in probs)


# ---- notes, workbook, runner -----------------------------------------------------------------------------------------------------------------------------

def test_notes_are_factual_and_only_emitted_when_relevant():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with economic_overrides(MILK_PRICE_PER_KG=0.5):
            pass
        assert not [x for x in w if issubclass(x.category, EconomicSettingsWarning)]
        with economic_overrides(BEEF_PRICE_PER_KG=2.5) as rec:
            pass
    msgs = [str(x.message) for x in w if issubclass(x.category, EconomicSettingsWarning)]
    assert any("DRESSING_PCT is not" in s for s in msgs) and rec["notes"] == msgs
    assert not any("double" in s.lower() or "incorrect" in s.lower() or "wrong" in s.lower() for s in msgs)
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        with economic_overrides(DAYS_OPEN_PENALTY_PER_DAY=4.0):
            pass
    assert any("DAYS_OPEN_PENALTY_PER_DAY is above 0" in str(x.message) for x in w2)


def test_workbook_records_non_default_settings_only_inside_a_block(cow, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from cowmrp.domains.cow.keepcull_excel import write_keepcull_workbook
    _, m = cow
    herd = pd.DataFrame({"state_id": [100, 200]})
    p0 = write_keepcull_workbook(herd, tmp_path / "default.xlsx", m, years=(1,))
    text0 = " ".join(str(c.value) for row in openpyxl.load_workbook(p0)["Inputs"].iter_rows() for c in row if c.value is not None)
    assert "NON-DEFAULT" not in text0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EconomicSettingsWarning)
        with economic_overrides(REARING_COST=2800.0, BEEF_PRICE_PER_KG=2.6):
            p1 = write_keepcull_workbook(herd, tmp_path / "scenario.xlsx", reprice_model(m), years=(1,))
    inp = openpyxl.load_workbook(p1)["Inputs"]
    text1 = " ".join(str(c.value) for row in inp.iter_rows() for c in row if c.value is not None)
    assert "NON-DEFAULT" in text1 and "REARING_COST" in text1 and inp["B2"].value == 2.6 and inp["B8"].value == 2800.0


def _run_runner(args, out):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONIOENCODING="utf8")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "farm", "keepcull.py"), "--years", "1", "--out-dir", str(out)] + args,
                       capture_output=True, text=True, env=env, cwd=ROOT)
    return r


FARM_CSV = os.path.join(ROOT, "runners", "cow", "outputs", "farm", "farm_data.csv")


@pytest.mark.skipif(not os.path.exists(FARM_CSV), reason="needs the generated farm file")
def test_runner_default_writes_no_extra_file_and_scenario_writes_a_record(tmp_path):
    d0, d1 = tmp_path / "default", tmp_path / "scenario"
    r0 = _run_runner([], d0)
    assert r0.returncode == 0, r0.stderr[-500:]
    assert sorted(os.listdir(d0)) == ["keepcull.csv", "summary.csv"]
    r1 = _run_runner(["--set", "REARING_COST=2800", "--set", "MILK_PRICE_PER_KG=0.5"], d1)
    assert r1.returncode == 0, r1.stderr[-500:]
    assert "NON-DEFAULT" in r1.stdout and sorted(os.listdir(d1)) == ["keepcull.csv", "keepcull.settings.json", "summary.csv"]
    rec = json.load(open(d1 / "keepcull.settings.json", encoding="utf8"))
    assert rec["overrides"]["REARING_COST"]["value"] == 2800.0 and rec["overrides"]["REARING_COST"]["default"] == 2450.0
    a, b = pd.read_csv(d0 / "keepcull.csv"), pd.read_csv(d1 / "keepcull.csv")
    assert not np.allclose(a["value_1y"], b["value_1y"])
    r2 = _run_runner(["--set", "TIME_UNIT=1"], tmp_path / "bad")
    assert r2.returncode != 0 and "not an overridable" in (r2.stderr + r2.stdout)

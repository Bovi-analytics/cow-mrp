"""Whole-farm layer (cowmrp.domains.cow.farm, cowmrp.viz.farm): raw-record loader, state mapping, projection, outputs.

Reference = the original runners/cow/farm/generate.py + simulate.py (frozen copies, 67 output files byte-identical after the migration);
baselines/farm_golden.json holds their numbers (herds of 200 cows, seed 42, equilibrium; day: 100 cows).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import HERE, ROOT
from cowmrp.data import example_herd_path, example_raw_herd_path
from cowmrp.domains.cow.farm import (HerdDataError, generate_herd, herd_values, load_herd, project_herd, state_ids_for)
from cowmrp.domains.cow.parameters import dim_step_of_days, dp_step_of_days

GOLD = json.load(open(os.path.join(HERE, "baselines", "farm_golden.json"), encoding="utf8"))


def _herd_from_gold(unit):
    return pd.DataFrame({"cow_id": range(1, len(GOLD["cases"][unit]["state_id"]) + 1), "state_id": GOLD["cases"][unit]["state_id"]})


def _check_result(r, unit, mode):
    e = GOLD["cases"][unit]["modes"][mode]
    ds = r["distribution_summary"]
    ref = pd.DataFrame(e["distribution_summary"])
    assert list(ds["grouper"]) == list(ref["grouper"]) and list(ds["label"]) == list(ref["label"])
    for c in ("current", "predicted", "change"):
        assert np.allclose(ds[c], ref[c], rtol=1e-9, atol=1e-9), c
    assert len(r["regular"]) == e["regular_cols"]
    for k, key in (("regular", "regular_last"), ("replaced", "replaced_last")):
        last = pd.Series(r[k][-1])
        for lab, v in e[key].items():
            assert last.get(lab, 0) == pytest.approx(v if v is not None else 0, rel=1e-9, abs=1e-9)
    if "values" in e and r["values"] is not None:
        for col, ref_vals in e["values"].items():
            got = r["values"][col].to_numpy(float)
            want = np.array([np.nan if x is None else x for x in ref_vals], float)
            assert np.allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True), col
    if mode == "monte_carlo":
        mc = r["monte_carlo"]
        assert len(mc) == e["mc_rows"] and mc["mean"].sum() == pytest.approx(e["mc_mean_sum"], rel=1e-12)
        assert mc["std"].sum() == pytest.approx(e["mc_std_sum"], rel=1e-12)
    if mode == "stochastic":
        t = r["trajectory"]
        assert list(t.shape) == e["traj_shape"] and int(t.drop(columns="cow_id").to_numpy().sum()) == e["traj_sum"]


# ---------------------------------------------------------------- the loader
def test_example_herd_is_200_cows_and_equals_the_generator(cow):
    _, m = cow
    h = load_herd(example_herd_path(), m)
    assert len(h) == 200 and h["cow_id"].is_unique
    pd.testing.assert_frame_equal(h, generate_herd(m, "equilibrium", 200, 42))          # seed 42, same draws as the original generate.py
    assert h.equals(pd.read_csv(example_herd_path()))


def test_raw_records_map_to_the_same_states_as_the_generator(cow):
    """Raw days (what a farmer exports) -> state_id equals the generator's state_id: the mapping is generate.py's, not a new one."""
    _, m = cow
    ref = generate_herd(m, "equilibrium", 200, 42)
    raw = load_herd(example_raw_herd_path(), m)
    assert raw["state_id"].tolist() == ref["state_id"].tolist()
    assert (raw[["lactation", "uim", "up", "status"]].values == ref[["lactation", "uim", "up", "status"]].values).all()


def test_mapping_equals_the_scalar_functions_of_the_original_exact_ids(cow):
    _, m = cow
    s = m.states
    rng = np.random.default_rng(3)
    ref = generate_herd(m, "equilibrium", 400, 11)
    ids, dim, dp = state_ids_for(s, ref["lactation"], ref["dim_days"], ref["dp_days"])
    for i in rng.choice(len(ref), 60, replace=False):
        q = (s["lactation"] == ref["lactation"][i]) & (s["dim"] == dim_step_of_days(ref["dim_days"][i])) & (s["dp"] == dp_step_of_days(ref["dp_days"][i]))
        assert ids[i] == int(s.loc[q, "state_id"].iloc[0]) == ref["state_id"][i]


@pytest.mark.parametrize("mode", ["mix", "exact"])
def test_other_generator_modes_load_back_identically(cow, mode):
    _, m = cow
    h = generate_herd(m, mode, 150, 7)
    again = load_herd(h[["cow_id", "lactation", "dim_days", "dp_days"]], m)
    assert again["state_id"].tolist() == h["state_id"].tolist()


def test_mapped_records_and_agreement_check(cow):
    _, m = cow
    ref = generate_herd(m, "equilibrium", 50, 1)
    mapped = load_herd(ref[["cow_id", "state_id"]], m)
    assert mapped["state_id"].tolist() == ref["state_id"].tolist() and (mapped["dim_days"] == ref["dim_days"]).all()
    both = ref[["cow_id", "state_id", "lactation", "dim_days", "dp_days"]].copy()
    assert load_herd(both, m)["state_id"].tolist() == ref["state_id"].tolist()
    both.loc[3, "state_id"] = (both.loc[3, "state_id"] + 1) % len(m.states)
    with pytest.raises(HerdDataError, match="disagrees"):
        load_herd(both, m)


def test_cow_id_defaults_and_dp_days_default(cow):
    _, m = cow
    h = load_herd(pd.DataFrame({"lactation": [1, 2], "dim_days": [60, 120]}), m)
    assert h["cow_id"].tolist() == [1, 2] and (h["dp_days"] == 0).all() and (h["status"] == "open").all()


def test_invalid_records_are_reported_together_with_rows(cow):
    _, m = cow
    bad = pd.DataFrame({"cow_id": [1, 2, 2, 4, 5, 6, 7], "lactation": [1, 11, 2, 0, "x", 2, 2], "dim_days": [60, 60, 60, 60, 60, -5, 60],
                        "dp_days": [0, 0, 0, 0, 0, 0, 240]})
    with pytest.raises(HerdDataError) as e:
        load_herd(bad, m)
    msg = str(e.value)
    for needle in ("duplicate cow_id", "lactation must be 1..", "non-numeric", "must be >= 0", "row 3", "cow_id 2"):
        assert needle in msg, needle


def test_a_cow_without_a_model_state_is_named(cow):
    _, m = cow
    with pytest.raises(HerdDataError, match="no model state") as e:
        load_herd(pd.DataFrame({"cow_id": [9], "lactation": [2], "dim_days": [60], "dp_days": [240]}), m)
    assert "row 2" in str(e.value) and "cow_id 9" in str(e.value) and "closest" not in "" and "lactation=2" in str(e.value)


def test_missing_columns_state_ids_files_and_empty(cow, tmp_path):
    _, m = cow
    with pytest.raises(HerdDataError, match="need either"):
        load_herd(pd.DataFrame({"foo": [1]}), m)
    with pytest.raises(HerdDataError, match="outside"):
        load_herd(pd.DataFrame({"state_id": [10 ** 6]}), m)
    with pytest.raises(HerdDataError, match="non-numeric"):
        load_herd(pd.DataFrame({"state_id": [None, 3]}), m)
    with pytest.raises(HerdDataError, match="not found"):
        load_herd(str(tmp_path / "nothing.csv"), m)
    with pytest.raises(HerdDataError, match="no cows"):
        load_herd(pd.DataFrame({"state_id": []}), m)


def test_herd_of_another_time_resolution_is_refused(cow, tmp_path):
    _, m = cow
    p = tmp_path / "h.csv"
    pd.read_csv(example_herd_path()).to_csv(p, index=False)
    json.dump({"time_unit": "week", "unit_days": 7}, open(tmp_path / "h.meta.json", "w"))
    with pytest.raises(HerdDataError, match="different configuration"):
        load_herd(str(p), m)
    assert len(load_herd(str(p), m, check_config=False)) == 200


# ---------------------------------------------------------------- projection against the original runner
@pytest.mark.parametrize("mode", ["expected", "stochastic", "monte_carlo"])
def test_month_projection_matches_original_runner(cow, mode):
    _, m = cow
    e = GOLD["cases"]["month"]["modes"][mode]
    r = project_herd(_herd_from_gold("month"), m, mode, e["steps"], seed=42, sims=e["sims"] or 100)
    _check_result(r, "month", mode)


def test_expected_projection_conserves_cows_and_tracks_replacements(cow):
    _, m = cow
    r = project_herd(load_herd(example_herd_path(), m), m, "expected", 12, values=False)
    for reg, rep in zip(r["regular"], r["replaced"]):
        assert reg.add(rep, fill_value=0).sum() == pytest.approx(200, abs=1e-6)
    assert r["values"] is None and r["replaced"][0].sum() == 0 and r["replaced"][-1].sum() > 0


def test_values_have_the_documented_properties(cow):
    _, m = cow
    from cowmrp.domains.cow.transitions import get_replacement_state_id
    h = pd.DataFrame({"cow_id": [1], "state_id": [get_replacement_state_id(m.states)]})
    v = herd_values(m, h)
    from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
    assert v["replacement_value"].iloc[0] == pytest.approx(cost_per_replacement(), abs=1e-6)     # the heifer vs herself: value = cost of one replacement
    assert np.isnan(v["pregnancy_value"].iloc[0])                                  # not pregnant


def test_projection_input_errors(cow):
    _, m = cow
    h = _herd_from_gold("month")
    with pytest.raises(ValueError, match="mode"):
        project_herd(h, m, "bogus")
    with pytest.raises(ValueError, match="grouper"):
        project_herd(h, m, groupers=["nope"])
    with pytest.raises(HerdDataError):
        project_herd(pd.DataFrame({"cow_id": [1], "state_id": [10 ** 6]}), m)


def test_outputs_are_written(cow, tmp_path):
    pytest.importorskip("matplotlib")
    from cowmrp.viz.farm import write_farm_outputs
    _, m = cow
    r = project_herd(load_herd(example_herd_path(), m), m, "monte_carlo", 6, sims=5)
    names = {os.path.basename(f) for f in write_farm_outputs(r, str(tmp_path))}
    assert {"distribution_summary.csv", "cow_values.csv", "monte_carlo_summary.csv", "replacement_tracking_regular.csv",
            "replacement_tracking_replaced.csv", "dashboard.html"} <= names


def test_example_script_runs(tmp_path):
    pytest.importorskip("matplotlib")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "whole_farm_example.py")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (tmp_path / "farm_example_output" / "distribution_summary.csv").exists() and (tmp_path / "farm_example_output" / "keepcull.csv").exists()


# ---------------------------------------------------------------- other resolutions and the runners
_CHILD = (
    "import json, sys\n"
    "import pandas as pd\n"
    "from cowmrp.domains.cow.farm import project_herd\n"
    "from cowmrp.model import build_model\n"
    "g = json.load(open(sys.argv[1]))['cases'][sys.argv[2]]\n"
    "m = build_model('cow'); mode = sys.argv[3]; e = g['modes'][mode]\n"
    "h = pd.DataFrame({'cow_id': range(1, len(g['state_id']) + 1), 'state_id': g['state_id']})\n"
    "r = project_herd(h, m, mode, e['steps'], 42, e['sims'] or 100, values=False)\n"
    "print('RESULT' + r['distribution_summary'].to_json(orient='records'))\n"
)


def _in_resolution(unit, mode):
    env = dict(os.environ, COW_TIME_UNIT=unit)
    r = subprocess.run([sys.executable, "-c", _CHILD, os.path.join(HERE, "baselines", "farm_golden.json"), unit, mode], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    got = pd.DataFrame(json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))[6:]))
    ref = pd.DataFrame(GOLD["cases"][unit]["modes"][mode]["distribution_summary"])
    assert list(got["label"]) == list(ref["label"])
    for c in ("current", "predicted"):
        assert np.allclose(got[c], ref[c], rtol=1e-9, atol=1e-9)


def test_week_projection_matches_original_runner():
    _in_resolution("week", "expected")                      # expected distribution is the default and the quick check


@pytest.mark.slow
def test_week_monte_carlo_matches_original_runner():
    _in_resolution("week", "monte_carlo")


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["expected", "stochastic"])
def test_day_projection_matches_original_runner(mode):
    _in_resolution("day", mode)


def test_runners_call_the_package_and_write_the_original_files(tmp_path):
    env = dict(os.environ, COW_TIME_UNIT="month")
    herd = tmp_path / "farm_data.csv"
    subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "farm", "generate.py"), "--n", "200", "--seed", "42", "--out", str(herd)],
                   check=True, capture_output=True, env=env, cwd=ROOT)
    assert pd.read_csv(herd).equals(pd.read_csv(example_herd_path()))                # the runner's file IS the shipped example herd
    out = tmp_path / "res"
    subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "farm", "simulate.py"), "--steps", "24", "--farm-csv", str(herd),
                    "--out-dir", str(out)], check=True, capture_output=True, env=env, cwd=ROOT)
    got = pd.read_csv(out / "distribution_summary.csv")
    ref = pd.DataFrame(GOLD["cases"]["month"]["modes"]["expected"]["distribution_summary"])
    assert np.allclose(got["predicted"], ref["predicted"], rtol=1e-9, atol=1e-9)
    assert {"cow_values.csv", "dashboard.html", "replacement_tracking_regular.csv"} <= set(os.listdir(out))
    for script in ("simulate.py", "generate.py"):
        text = open(os.path.join(ROOT, "runners", "cow", "farm", script), encoding="utf8").read()
        assert "cowmrp.domains.cow.farm" in text and "assemble_matrix(" not in text

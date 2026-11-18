"""Keep / cull table (cowmrp.domains.cow.keepcull) against the original runners/cow/farm/keepcull.py (baselines/keepcull_golden.json),
and against value_comparisons.replacement_comparison (the single-cow comparison it generalises)."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from conftest import HERE, ROOT
from cowmrp.domains.cow.keepcull import keep_cull_summary, keep_cull_table

GOLD = json.load(open(os.path.join(HERE, "baselines", "keepcull_golden.json"), encoding="utf8"))
OFFER, GAMMA = GOLD["offer"], GOLD["annual_discount"]


def _herd(unit):
    return pd.DataFrame({"state_id": GOLD["cases"][unit]["state_id"]})


def _check(table, unit):
    c = GOLD["cases"][unit]
    wanted = [col for col in c["columns"] if col not in ("cow_id", "lactation", "dim_days", "dp_days")]     # herd descriptors are copied only when given
    assert list(table.columns) == wanted
    for y in c["years"]:
        assert np.allclose(table[f"value_{y}y"], c["values"][str(y)], rtol=1e-12, atol=1e-9), y
        assert table[f"decision_{y}y"].tolist() == c["decisions"][str(y)]
    assert table["rank_longest"].tolist() == c["rank_longest"] and table["stable"].tolist() == c["stable"]


def test_month_table_matches_original_runner(cow):
    _, m = cow
    c = GOLD["cases"]["month"]
    t = keep_cull_table(_herd("month"), m, c["years"], GAMMA, OFFER)
    _check(t, "month")


def test_month_summary_matches_original(cow):
    _, m = cow
    c = GOLD["cases"]["month"]
    herd = _herd("month")
    herd["lactation"] = m.states.loc[herd["state_id"], "lactation"].to_numpy()          # the runner's herd file carries lactation
    s = keep_cull_summary(keep_cull_table(herd, m, c["years"], GAMMA, OFFER), c["years"])
    ref = pd.DataFrame(c["summary"])
    assert s.shape == ref.shape and list(s.columns) == list(ref.columns)
    for col in ("horizon_years", "cows", "cull"):
        assert (s[col].to_numpy() == ref[col].to_numpy()).all()
    for col in ("min_value", "p10_value", "mean_value", "cull_pct"):
        assert np.allclose(s[col], ref[col], rtol=1e-12, atol=1e-9)


def test_without_an_offer_the_table_only_ranks(cow):
    _, m = cow
    t = keep_cull_table(_herd("month"), m, (1, 3), GAMMA, None)
    assert not any(c.startswith("decision_") for c in t.columns) and "stable" not in t.columns
    assert sorted(t["rank_longest"]) == list(range(1, len(t) + 1))


def test_equals_replacement_comparison_for_single_cows(cow):
    """The table is replacement_comparison applied to every cow: same keep, same heifer, same difference."""
    from cowmrp.mrp import core
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.value_comparisons import replacement_comparison, replacement_net_values
    _, m = cow
    years, herd = (1, 3), _herd("month").iloc[:25]
    t = keep_cull_table(herd, m, years, GAMMA, None, detail=True)
    gamma = core.annual_to_per_step_gamma(GAMMA, UNIT_DAYS)
    for y in years:
        net = replacement_net_values(m.states, m.P, m.rewards, m.probs, gamma, y)
        for sid, row in zip(herd["state_id"], t.itertuples()):
            s = m.states.iloc[sid]
            rc = replacement_comparison(m.states, m.P, m.rewards, m.probs, gamma, int(s.lactation), int(s.dim), int(s.dp), num_years=y, net=net)
            assert getattr(row, f"value_{y}y") == pytest.approx(rc["value"], rel=1e-12, abs=1e-9)
            assert getattr(row, f"keep_{y}y") == pytest.approx(rc["V_state"], rel=1e-12, abs=1e-9)
            assert getattr(row, f"replace_{y}y") == pytest.approx(rc["V_comparison"], rel=1e-12, abs=1e-9)


def test_three_quantities_stay_separate(cow):
    _, m = cow
    t = keep_cull_table(_herd("month"), m, (3,), GAMMA, OFFER, detail=True)
    assert np.allclose(t["value_3y"], t["keep_3y"] - t["replace_3y"], rtol=1e-12, atol=1e-9)     # comparison = keep - replace
    assert t["replace_3y"].nunique() == 1                                                          # the heifer is the same for every cow
    plain = keep_cull_table(_herd("month"), m, (3,), GAMMA, OFFER)
    assert [c for c in t.columns if c not in ("keep_3y", "replace_3y")] == list(plain.columns)      # detail only adds columns
    assert (t["decision_3y"] == np.where(OFFER > t["value_3y"], "CULL", "KEEP")).all()             # CULL when offer > break-even value


def test_invalid_herds_are_errors(cow):
    _, m = cow
    with pytest.raises(ValueError, match="state_id"):
        keep_cull_table(pd.DataFrame({"cow": [1]}), m)
    with pytest.raises(ValueError, match="outside"):
        keep_cull_table(pd.DataFrame({"state_id": [10 ** 7]}), m, (1,))
    with pytest.raises(ValueError, match="years"):
        keep_cull_table(_herd("month"), m, ())


_CHILD = textwrap.dedent("""
    import json, sys
    import pandas as pd
    from cowmrp.domains.cow.keepcull import keep_cull_table
    c = json.load(open(sys.argv[1]))["cases"][sys.argv[2]]
    t = keep_cull_table(pd.DataFrame({"state_id": c["state_id"]}), None, c["years"], 0.95, 900.0)
    print("RESULT" + t.to_json(orient="columns"))
""")


def _in_resolution(unit):
    env = dict(os.environ, COW_TIME_UNIT=unit)
    r = subprocess.run([sys.executable, "-c", _CHILD, os.path.join(HERE, "baselines", "keepcull_golden.json"), unit], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    return pd.DataFrame(json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))[6:]))


def test_week_matches_original_runner():
    t = _in_resolution("week")
    _check(t, "week")


@pytest.mark.slow
def test_day_matches_original_runner():
    t = _in_resolution("day")
    _check(t, "day")


def test_runner_reproduces_the_original_output(tmp_path):
    """generate.py (seed 42) -> runners/cow/farm/keepcull.py -> keepcull.csv equals the original runner's values."""
    herd = tmp_path / "farm_data.csv"
    env = dict(os.environ, COW_TIME_UNIT="month")
    subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "farm", "generate.py"), "--n", "200", "--seed", "42", "--mode", "equilibrium",
                    "--out", str(herd)], check=True, capture_output=True, env=env, cwd=ROOT)
    out = tmp_path / "kc"
    subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "farm", "keepcull.py"), "--years", "1,3,5", "--offer", "900",
                    "--farm-csv", str(herd), "--out-dir", str(out)], check=True, capture_output=True, env=env, cwd=ROOT)
    t = pd.read_csv(out / "keepcull.csv")
    assert t["state_id"].tolist() == GOLD["cases"]["month"]["state_id"]            # the generator is deterministic
    assert list(t.columns) == GOLD["cases"]["month"]["columns"]                    # the runner's file has exactly the original columns
    for y in (1, 3, 5):
        assert np.allclose(t[f"value_{y}y"], GOLD["cases"]["month"]["values"][str(y)], rtol=1e-12, atol=1e-9)
        assert t[f"decision_{y}y"].tolist() == GOLD["cases"]["month"]["decisions"][str(y)]
    assert (out / "summary.csv").exists()

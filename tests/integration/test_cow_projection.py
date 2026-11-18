"""Cow A versus cow B (cowmrp.domains.cow.projection / cowmrp.viz.projection).

The saved reference (baselines/projection_golden.json) was produced by the pre-migration runner/projection.py worker (the original
implementation, independent of the package) for fixed scenarios: month and week in the default tests, day in the slow test.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import HERE, ROOT
from cowmrp.domains.cow.projection import compare_cows, project_scenarios, validate_scenarios

GOLDEN = json.load(open(os.path.join(HERE, "baselines", "projection_golden.json"), encoding="utf8"))
SCEN = {s["label"]: s for s in GOLDEN["scenarios"]}


def _summary(r):
    """The numbers that identify a result: end-of-horizon cumulative vector, per-column totals, shares and checks."""
    return {"cum_end": r["cum"][-1], "exp_sum": np.asarray(r["exp_per_step"]).sum(axis=0).tolist(), "status_end": r["status"][-1],
            "lac_sum": np.asarray(r["lactation_shares"]).sum(axis=0).tolist(), "dim_sum": np.asarray(r["dim_bins"]).sum(axis=0).tolist(),
            "orig_end": r["original_still_here"][-1], "finite_horizon_value": r["finite_horizon_value"], "names": r["names"],
            "n_steps": len(r["t_days"]), "n_states": r["n_states"], "gamma": r["gamma"]}


def _assert_matches(result, label):
    ref, got = GOLDEN["results"][label], _summary(result)
    assert got["names"] == ref["names"] and got["n_steps"] == ref["n_steps"] and got["n_states"] == ref["n_states"]
    for key in ("cum_end", "exp_sum", "status_end", "lac_sum", "dim_sum"):
        assert np.allclose(got[key], ref[key], rtol=1e-12, atol=1e-9), key
    for key in ("orig_end", "finite_horizon_value", "gamma"):
        assert got[key] == pytest.approx(ref[key], rel=1e-12)
    assert result["check_vs_finite_horizon"] < 1e-9 and result["check_distribution"] < 1e-9


def _scen(units):
    return [SCEN[l] for l in SCEN if SCEN[l]["time_unit"] in units]


def test_month_scenarios_match_the_original_implementation():
    sc = _scen(("month",))
    res = compare_cows(sc, years=GOLDEN["years"], verbose=False)
    for i, s in enumerate(sc):
        _assert_matches(res[i], s["label"])


def test_week_scenarios_run_in_a_worker_process_and_match():
    sc = _scen(("week",))
    res = compare_cows(sc, years=GOLDEN["years"], verbose=False)         # this process is at month resolution: week goes through the worker
    for i, s in enumerate(sc):
        _assert_matches(res[i], s["label"])


def test_mixed_resolutions_keep_the_order_of_the_scenarios():
    sc = [SCEN["Heifer week"], SCEN["Open cow L2 DIM150 month"]]
    res = compare_cows(sc, years=GOLDEN["years"], verbose=False)
    assert [res[0]["unit"], res[1]["unit"]] == ["week", "month"] and res[0]["label"] == "Heifer week"


@pytest.mark.slow
def test_day_scenario_matches():
    sc = _scen(("day",))
    res = compare_cows(sc, years=GOLDEN["years"], verbose=False)
    _assert_matches(res[0], sc[0]["label"])


def test_monte_carlo_bands_are_reproducible_and_bracket_the_expectation(cow):
    sc = [SCEN["Open cow L2 DIM150 month"]]
    a = compare_cows(sc, years=1, n_trajectories=20, seed=5, verbose=False)[0]
    b = compare_cows(sc, years=1, n_trajectories=20, seed=5, verbose=False)[0]
    assert a["mc"] == b["mc"] and a["mc"]["n"] == 20
    j = a["names"].index("total")
    assert a["mc"]["p10"][-1][j] <= a["mc"]["p50"][-1][j] <= a["mc"]["p90"][-1][j]


def test_cow_a_versus_cow_b_pregnancy_has_value(cow):
    """The documented use case: the same cow open versus pregnant 60 d (scenario 2 vs 3 of the runner)."""
    a = dict(label="open", time_unit="month", lactation=2, dim_days=150, dp_days=0)
    b = dict(label="pregnant", time_unit="month", lactation=2, dim_days=150, dp_days=60)
    res = compare_cows([a, b], years=2, verbose=False)
    assert res[0]["finite_horizon_value"] != res[1]["finite_horizon_value"]
    assert res[0]["start"] != res[1]["start"]


def test_invalid_inputs_are_errors_not_silent():
    with pytest.raises(ValueError, match="empty"):
        compare_cows([], verbose=False)
    with pytest.raises(ValueError, match="time_unit"):
        validate_scenarios([dict(label="x", time_unit="year", lactation=1)])
    with pytest.raises(ValueError, match="lactation 1-10"):
        validate_scenarios([dict(label="x", time_unit="month", lactation=11)])
    with pytest.raises(ValueError, match="does not exist in the model"):
        compare_cows([dict(label="bad", time_unit="month", lactation=2, dim_days=60, dp_days=240)], verbose=False)
    with pytest.raises(ValueError, match="another"):
        project_scenarios([dict(label="w", time_unit="week", lactation=1, dim_days=0, dp_days=0)], verbose=False)     # wrong resolution for this process


def test_tables_and_figures(cow):
    pytest.importorskip("plotly")
    from cowmrp.viz.projection import projection_figures, projection_tables
    sc = [SCEN["Open cow L2 DIM150 month"], SCEN["Dry cow L3 230d month"]]
    res = compare_cows(sc, years=GOLDEN["years"], verbose=False)
    summary, series = projection_tables(res, GOLDEN["years"])
    names = res[0]["names"]
    assert len(summary) == 2 * len(names) and f"cumulative_year_{GOLDEN['years']}" in summary.columns
    assert set(series) == {"Open_cow_L2_DIM150_month", "Dry_cow_L3_230d_month"}
    ts = series["Open_cow_L2_DIM150_month"]
    assert {"total_per_day", "total_cumulative", "share_Open", "original_cow_still_in_herd"} <= set(ts.columns)
    figs = projection_figures(res, GOLDEN["years"], 0.95, 0)
    assert len(figs) == len(names) + len(sc) and all(len(f) == 4 for f in figs)


def test_runner_writes_the_same_tables_as_the_package(tmp_path, monkeypatch):
    """The runner only writes files: its CSV tables equal the package tables."""
    import importlib, sys
    sys.path.insert(0, ROOT)
    try:
        runner = importlib.import_module("runners.cow.projection")
    finally:
        sys.path.remove(ROOT)
    monkeypatch.setattr(runner, "SCENARIOS", [SCEN["Open cow L2 DIM150 month"]])
    monkeypatch.setattr(runner, "YEARS", GOLDEN["years"])
    monkeypatch.setattr(runner, "N_TRAJECTORIES", 0)
    monkeypatch.setattr(runner, "OUTPUT_FOLDER", str(tmp_path))
    runner.main()
    from cowmrp.viz.projection import projection_tables
    res = compare_cows([SCEN["Open cow L2 DIM150 month"]], years=GOLDEN["years"], verbose=False)
    summary, series = projection_tables(res, GOLDEN["years"])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "summary.csv"), summary, check_exact=False, rtol=1e-12)
    assert (tmp_path / "index.html").exists() and (tmp_path / "projection_all.html").exists()

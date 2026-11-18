"""build -> evaluate -> distribute -> simulate -> export through the package API, for the real domains."""
import os

import numpy as np
import pandas as pd
import pytest

from cowmrp.io import export_model
from cowmrp.model import (build_model, evaluate, evolve_distribution, group_summary, initial_distribution, monte_carlo,
                          per_step_gamma, replacement_split, simulate, start_state_id)


def test_build_is_deterministic(cow):
    dom, m = cow
    again = build_model(dom)
    assert (again.P != m.P).nnz == 0 and np.array_equal(again.rewards["total"], m.rewards["total"])


def test_full_versus_reachable_scope(cow):
    dom, m = cow
    full, reach = evaluate(m, dom), evaluate(m, dom, start="replacement")
    assert full["scope"] == "full" and full["n_states"] == 2970
    assert reach["scope"] == "reachable" and reach["n_states"] == 1210
    assert np.allclose(full["V_infinite"][reach["state_ids"]], reach["V_infinite"], rtol=1e-9, atol=1e-7)
    assert np.allclose(full["V_finite"][reach["state_ids"]], reach["V_finite"], rtol=1e-9, atol=1e-7)
    assert full["V_infinite"].mean() != pytest.approx(reach["V_infinite"].mean())       # the two means are different quantities


def test_evaluate_which_skips_work(cow):
    dom, m = cow
    out = evaluate(m, dom, start="replacement", which=("finite",))
    assert "V_finite" in out and "V_infinite" not in out


def test_evaluate_without_rewards_is_an_error(youngstock):
    dom, m = youngstock
    with pytest.raises(ValueError, match="no rewards"):
        evaluate(m, dom)


def test_start_state_resolution(cow):
    dom, m = cow
    rid = start_state_id(m, dom, "replacement")
    assert m.states.iloc[rid][["lactation", "dim", "dp"]].tolist() == [1, 0, 0]
    assert start_state_id(m, dom, 5) == 5
    assert m.states.iloc[start_state_id(m, dom, "initial")][["lactation", "dim", "dp"]].tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        start_state_id(m, dom, "nonsense")


def test_discount_matches_annual(cow):
    dom, _ = cow
    assert per_step_gamma(dom, 0.95) ** (365 / dom.unit_days) == pytest.approx(0.95, rel=1e-9)   # discount per 365 real days


def test_distribution_methods_and_mass(cow):
    dom, m = cow
    pi0 = initial_distribution(m, dom)
    assert pi0.sum() == 1.0
    pi = evolve_distribution(m, pi0, dom)
    assert pi.sum() == pytest.approx(1.0) and (pi >= 0).all()
    assert np.allclose(pi, evolve_distribution(m, pi0, dom, method="matrix_power"), atol=1e-9)
    with pytest.raises(ValueError):
        evolve_distribution(m, pi0, dom, method="bogus")


def test_replacement_split_conserves_mass_and_memory_mode_agrees(cow):
    dom, m = cow
    pi0 = initial_distribution(m, dom)
    never, replaced = replacement_split(m, pi0, dom, 12)
    assert len(never) == 13 and all((a + b).sum() == pytest.approx(1.0) for a, b in zip(never, replaced))
    assert replaced[0].sum() == 0 and replaced[-1].sum() > replaced[1].sum() > 0
    last_n, last_r = replacement_split(m, pi0, dom, 12, keep_trajectory=False)
    assert np.allclose(last_n, never[-1]) and np.allclose(last_r, replaced[-1])


def test_replacement_split_needs_the_hook(youngstock):
    dom, m = youngstock
    with pytest.raises(ValueError, match="replacement_state_id"):
        replacement_split(m, initial_distribution(m, dom), dom)


def test_simulation_seed_and_group_summary(cow):
    dom, m = cow
    ids = np.zeros(40, dtype=int)
    assert (simulate(m, ids, 12, seed=3) == simulate(m, ids, 12, seed=3)).all()
    sims = monte_carlo(m, ids, 12, n_sims=8, seed=3)
    g = group_summary(m, dom, sims)
    assert list(g.columns) == ["label", "mean", "std", "p05", "p95"] and g["mean"].sum() == pytest.approx(40.0)


def test_export_model_files_and_cow_column_names(cow, tmp_path):
    dom, m = cow
    files = export_model(m, dom, str(tmp_path), parquet=False)
    assert sorted(os.path.basename(p) for p in files) == ["reward_vector_cow.csv", "states.csv", "transition_matrix_cow.npz",
                                                          "transition_matrix_cow_long.csv"]
    long = pd.read_csv(tmp_path / "transition_matrix_cow_long.csv")
    assert list(long.columns) == ["src_lactation", "src_uim", "src_up", "tgt_lactation", "tgt_uim", "tgt_up", "probability"] and len(long) == 9040
    assert list(pd.read_csv(tmp_path / "reward_vector_cow.csv").columns)[:3] == ["lactation", "uim", "up"]


def test_export_youngstock_has_no_reward_file(youngstock, tmp_path):
    dom, m = youngstock
    files = export_model(m, dom, str(tmp_path), parquet=False)
    assert not any("reward_vector" in f for f in files)
    assert "src_up" in pd.read_csv(tmp_path / "transition_matrix_youngstock_long.csv").columns

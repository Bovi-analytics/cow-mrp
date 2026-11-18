"""The current implementation is the scientific reference (baselines/golden_month.json, month resolution).
Failure here means a model number moved: find the cause before touching the baseline."""
import numpy as np
import pytest

from cowmrp.model import evaluate, matrix_checksum


@pytest.mark.parametrize("name", ["cow", "youngstock"])
def test_structure_and_matrix(name, request, golden):
    dom, m = request.getfixturevalue(name)
    g = golden[name]
    assert len(m.states) == g["num_states"] and m.P.nnz == g["nnz"]
    assert matrix_checksum(m.P) == g["matrix_checksum"]


def test_cow_values_full_and_reachable(cow, golden):
    dom, m = cow
    g = golden["cow"]
    full, reach = evaluate(m, dom), evaluate(m, dom, start="replacement")
    assert full["V_finite"].mean() == pytest.approx(g["full_mean_V_finite"], rel=1e-9)
    assert full["V_infinite"].mean() == pytest.approx(g["full_mean_V_infinite"], rel=1e-9)
    assert reach["n_states"] == g["reachable_states"]
    assert reach["V_finite"].mean() == pytest.approx(g["reachable_mean_V_finite"], rel=1e-9)
    assert reach["V_infinite"].mean() == pytest.approx(g["reachable_mean_V_infinite"], rel=1e-9)
    assert float(m.rewards["total"].sum()) == pytest.approx(g["reward_total_sum"], rel=1e-12)


def test_reachable_means_equal_the_historical_model_a_baseline(cow):
    """Model_A (saved before the full state table) reported 12,382.709599 / 54,815.982755 over the 1,210 reachable states."""
    dom, m = cow
    reach = evaluate(m, dom, start="replacement")
    assert reach["V_finite"].mean() == pytest.approx(12382.709599, abs=2e-6)
    assert reach["V_infinite"].mean() == pytest.approx(54815.982755, abs=1e-5)


def test_cow_state_variables_and_reward_vector(cow):
    _, m = cow
    assert [c for c in m.states.columns if c != "state_id"] == ["lactation", "dim", "dp"]
    assert "replacement_cost" not in m.rewards          # charged in the cow-versus-heifer comparison, not in the state reward


def test_rows_sum_to_one(cow, youngstock):
    for _, m in (cow, youngstock):
        assert np.allclose(np.asarray(m.P.sum(axis=1)).ravel(), 1.0, atol=1e-8)

"""Unit tests of the generic engine on a hand-solvable 3-state MRP (no domain involved)."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cowmrp.mrp import core
from cowmrp.mrp.utils import matrix_checksum, reachable_states, row_sum_report

STATES = pd.DataFrame({"a": [0, 1, 2], "state_id": [0, 1, 2]})
ROWS, COLS, VALS = [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [0.8, 0.2, 0.5, 0.5, 1.0]
R = np.array([10.0, 4.0, -20.0])


@pytest.fixture
def P():
    return core.assemble_matrix(STATES, ROWS, COLS, VALS)


def test_assemble_shape_and_nnz(P):
    assert P.shape == (3, 3) and P.nnz == 5 and abs(P.sum() - 3) < 1e-12


def test_validate_row_stochasticity_accepts_and_rejects(P):
    assert core.validate_row_stochasticity(P, STATES)
    bad = core.assemble_matrix(STATES, ROWS, COLS, [0.8, 0.1, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="don't sum to 1"):
        core.validate_row_stochasticity(bad, STATES)


def test_infinite_horizon_matches_dense_solve(P):
    g = 0.95
    V = core.evaluate_infinite_horizon(P, R, g)
    assert np.allclose(V, np.linalg.solve(np.eye(3) - g * P.toarray(), R), rtol=1e-12)


def test_finite_horizon_matches_closed_form_and_converges(P):
    g, steps = 0.99, 12 * 3
    V = core.evaluate_finite_horizon(P, R, g, num_years=3, unit_days=30)       # steps_per_year(30) = 12
    Pd = P.toarray()
    expect = sum(np.linalg.matrix_power(g * Pd, t) @ R for t in range(steps))
    assert np.allclose(V, expect, rtol=1e-10)
    far = core.evaluate_finite_horizon(P, R, g, num_years=400, unit_days=30)
    assert np.allclose(far, core.evaluate_infinite_horizon(P, R, g), rtol=1e-6)


def test_annual_to_per_step_gamma_compounds_to_annual():
    for unit_days in (1, 7, 30):
        g = core.annual_to_per_step_gamma(0.95, unit_days)
        assert g ** (365 / unit_days) == pytest.approx(0.95, rel=1e-9)


def test_distribution_methods_agree(P):
    pi0 = np.array([1.0, 0, 0])
    a = core.distribution_recursive(P, pi0, 10)
    assert np.allclose(a, core.distribution_matrix_power(P, pi0, 10))
    assert np.allclose(a, core.distribution_eigen(P, pi0, 10))
    assert a.sum() == pytest.approx(1.0)


def test_origin_tracked_conserves_mass_and_tags_reentry(P):
    pu, pt = np.array([1.0, 0, 0]), np.zeros(3)
    for _ in range(30):
        pu, pt = core.distribution_recursive_origin_tracked(P, pu, pt, 1, 0)
    assert (pu + pt).sum() == pytest.approx(1.0)
    assert pt.sum() > 0


def test_export_names_come_from_the_caller(P):
    t = core.export_transitions(P, STATES, column_names={"a": "renamed"})
    assert "src_renamed" in t.columns and "tgt_renamed" in t.columns and len(t) == 5
    assert "src_a" in core.export_transitions(P, STATES).columns
    r = core.export_rewards(STATES, {"total": R}, column_names={"a": "renamed"})
    assert "renamed" in r.columns and list(r["total"]) == list(R)
    with pytest.raises(TypeError):
        core.export_transitions(P, STATES, unit_prefix="u")       # the cow-specific argument is gone


def test_simulation_is_reproducible_and_stays_on_support(P):
    h1 = core.simulate_stochastic_trajectory(P, np.zeros(50, dtype=int), 20, seed=7)
    h2 = core.simulate_stochastic_trajectory(P, np.zeros(50, dtype=int), 20, seed=7)
    assert h1.shape == (21, 50) and (h1 == h2).all()
    allowed = {(int(i), int(j)) for i, j in zip(ROWS, COLS)}
    assert all((int(a), int(b)) in allowed for a, b in zip(h1[:-1].ravel(), h1[1:].ravel()))


def test_utils_checksum_row_report_reachable(P):
    assert matrix_checksum(P) == matrix_checksum(P.copy()) and len(matrix_checksum(P)) == 16
    changed = core.assemble_matrix(STATES, ROWS, COLS, [0.7, 0.3, 0.5, 0.5, 1.0])
    assert matrix_checksum(P) != matrix_checksum(changed)
    rep = row_sum_report(csr_matrix(np.array([[0.5, 0.2], [0.0, 1.0]])))
    assert rep["rows_not_summing_to_1"] == 1 and rep["row_sum_min"] == pytest.approx(0.7)
    chain = csr_matrix(np.array([[0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]]))
    assert list(reachable_states(chain, 1)) == [1, 2]
    assert list(reachable_states(chain, 0)) == [0, 1, 2]

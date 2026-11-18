"""Performance guards: generous budgets that catch an accidental order-of-magnitude slowdown (not micro-benchmarks), plus memory
and equivalence checks for the optimised paths. Budgets assume month resolution on an ordinary laptop; they are not scientific results."""
import time
import tracemalloc

import numpy as np
import pytest

from cowmrp.domain import load_domain
from cowmrp.model import build_model, evaluate, group_summary, initial_distribution, monte_carlo, replacement_split

pytestmark = pytest.mark.performance


def _time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def test_cow_month_build_budget():
    dom = load_domain("cow")
    assert _time(lambda: build_model(dom), repeat=2) < 30.0


def test_reachable_solve_is_not_slower_than_full_and_equal(cow):
    dom, m = cow
    t_full = _time(lambda: evaluate(m, dom, which=("infinite",)))
    t_reach = _time(lambda: evaluate(m, dom, start="replacement", which=("infinite",)))
    assert t_reach < max(2.0 * t_full, 1.0)                       # the closed subsystem must never cost much more than the full solve
    full, reach = evaluate(m, dom, which=("infinite",)), evaluate(m, dom, start="replacement", which=("infinite",))
    assert np.allclose(full["V_infinite"][reach["state_ids"]], reach["V_infinite"], rtol=1e-9, atol=1e-7)


def test_value_iteration_budget(cow):
    dom, m = cow
    assert _time(lambda: evaluate(m, dom, which=("finite",))) < 5.0


def test_replacement_split_last_step_only_uses_less_memory(cow):
    dom, m = cow
    pi0 = initial_distribution(m, dom)

    def peak(**kw):
        tracemalloc.start()
        replacement_split(m, pi0, dom, 240, **kw)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p

    full, last = peak(keep_trajectory=True), peak(keep_trajectory=False)
    assert last < full / 5          # O(n) instead of O(steps x n): 240 steps x 2 vectors x 2,970 states x 8 bytes is about 11 MB


def test_group_summary_is_vectorised(cow):
    dom, m = cow
    sims = monte_carlo(m, np.zeros(200, dtype=int), 12, n_sims=100, seed=1)
    assert _time(lambda: group_summary(m, dom, sims)) < 1.0


def test_monte_carlo_budget(cow):
    dom, m = cow
    assert _time(lambda: monte_carlo(m, np.zeros(1000, dtype=int), 12, n_sims=20, seed=1), repeat=1) < 60.0

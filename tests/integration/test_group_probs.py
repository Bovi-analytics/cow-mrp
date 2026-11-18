"""group_probs: the integer-coded version must equal the original string-label groupby bit for bit (np.array_equal, same index).

REFERENCE is the original one-line implementation, kept here only for the comparison. Month runs in-process; week (fast) and day (slow) run in a
child process because the time resolution is fixed when the package is imported."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cowmrp.domains.cow.farm import DEFAULT_GROUPERS, GROUPERS, group_probs


def reference(states, pi, grouper):
    return pd.Series(pi).groupby(GROUPERS[grouper](states)).sum().sort_index()


def _inputs(n, rng):
    dirichlet = rng.dirichlet(np.ones(n))
    sparse = np.where(rng.random(n) < 0.8, 0.0, rng.random(n))
    sparse[0] += 1.0                                         # never all zero, so no 0/0
    sparse /= sparse.sum()
    point = np.zeros(n)
    point[n // 3] = 1.0
    tiny = rng.random(n) * 1e-300
    wide = rng.random(n) * 10.0 ** rng.integers(-12, 6, n)
    return {"dirichlet": dirichlet, "sparse": sparse, "point": point, "zeros": np.zeros(n), "uniform": np.full(n, 1.0 / n),
            "tiny": tiny, "wide": wide, "list": list(dirichlet), "negative": rng.normal(size=n), "nan": np.where(rng.random(n) < 0.1, np.nan, dirichlet)}


def _same(a, b):
    assert a.index.equals(b.index) and a.index.dtype == b.index.dtype and a.dtype == b.dtype and a.name == b.name
    assert np.array_equal(a.values, b.values, equal_nan=True)      # NaN only occurs in the explicit NaN input, where both give the same NaNs


def check_all(states, seed=0):
    rng = np.random.default_rng(seed)
    n = len(states)
    for grouper in GROUPERS:
        for label, pi in _inputs(n, rng).items():
            _same(group_probs(states, pi, grouper), reference(states, pi, grouper))
            _same(group_probs(states, pi, grouper), reference(states, pi, grouper))    # second call is served from the cache


def test_every_grouper_and_input_matches_the_original_at_month_resolution(cow):
    check_all(cow[1].states)


def test_a_subset_of_states_and_a_reordered_table_match(cow):
    states = cow[1].states
    check_all(states.iloc[:137].reset_index(drop=True), 1)                                  # only some labels present
    check_all(states.sample(frac=1.0, random_state=3).reset_index(drop=True), 2)           # different row order
    check_all(states.iloc[[0]].reset_index(drop=True), 3)                                   # single state


def test_a_table_changed_in_place_is_not_served_from_a_stale_cache(cow):
    states = cow[1].states.copy()
    pi = np.random.default_rng(4).dirichlet(np.ones(len(states)))
    _same(group_probs(states, pi, "lactation"), reference(states, pi, "lactation"))
    states.loc[states.index[:50], "lactation"] = 9
    _same(group_probs(states, pi, "lactation"), reference(states, pi, "lactation"))


def test_wrong_length_and_unknown_grouper_fail_like_the_original(cow):
    states = cow[1].states
    with pytest.raises(Exception):
        reference(states, np.ones(len(states) - 1), "status")
    with pytest.raises(Exception):
        group_probs(states, np.ones(len(states) - 1), "status")
    with pytest.raises(KeyError):
        group_probs(states, np.ones(len(states)), "nope")


def test_default_projection_columns_are_unchanged(cow):
    assert all(g in GROUPERS for g in DEFAULT_GROUPERS)
    states = cow[1].states
    pi = np.random.default_rng(5).dirichlet(np.ones(len(states)))
    assert list(group_probs(states, pi, "status").index) == ["Dry", "Open", "Pregnant"]


_CHILD = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from test_group_probs import check_all\n"
    "from cowmrp.model import build_model\n"
    "m = build_model('cow')\n"
    "check_all(m.states)\n"
    "print('OK', len(m.states))\n"
)


def _child(unit):
    env = dict(os.environ, COW_TIME_UNIT=unit)
    r = subprocess.run([sys.executable, "-c", _CHILD, os.path.dirname(os.path.abspath(__file__))], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-1500:]


def test_week_resolution_matches_the_original():
    _child("week")


@pytest.mark.slow
def test_day_resolution_matches_the_original():
    _child("day")

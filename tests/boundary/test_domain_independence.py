"""The MRP engine must stay domain-free, and a domain that is not an animal must run through the same pipeline."""
import ast
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from cowmrp.io import export_model
from cowmrp.model import evaluate, evolve_distribution, group_summary, initial_distribution, monte_carlo, replacement_split

MRP_DIR = os.path.join(ROOT, "src", "cowmrp", "mrp")
MRP_FILES = sorted(glob.glob(os.path.join(MRP_DIR, "*.py")))
FORBIDDEN_PREFIX = ("cowmrp.domain", "cowmrp.domains", "cowmrp.model", "cowmrp.io", "cowmrp.viz", "cowmrp.runner",
                    "runners", "domains", "couplings")
DOMAIN_WORDS = {"dim", "dp", "lactation", "uim", "up", "cow", "youngstock", "heifer", "calf"}


@pytest.mark.parametrize("path", MRP_FILES, ids=os.path.basename)
def test_mrp_imports_no_domain_or_upper_layer(path):
    tree = ast.parse(open(path, encoding="utf8").read())
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        for m in mods:
            assert not m.startswith(FORBIDDEN_PREFIX), f"{os.path.basename(path)}:{node.lineno} imports {m}"


@pytest.mark.parametrize("path", MRP_FILES, ids=os.path.basename)
def test_mrp_code_names_no_domain_columns(path):
    tree = ast.parse(open(path, encoding="utf8").read())
    hits = [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in DOMAIN_WORDS]
    assert not hits, hits


def test_boundary_check_script_passes():
    script = os.path.join(ROOT, "runners", "_shared", "checks", "verify_mrp_boundary.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_mrp_imports_cleanly_without_any_domain():
    """Import cowmrp.mrp in a fresh interpreter: no domain, model or runner module may be loaded as a side effect."""
    code = ("import sys, cowmrp.mrp.core, cowmrp.mrp.engine, cowmrp.mrp.simulator, cowmrp.mrp.utils;"
            "bad=[m for m in sys.modules if m.startswith(('cowmrp.domains','cowmrp.domain','cowmrp.model','runners'))];"
            "print('LOADED', bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---- a non-animal domain through the whole pipeline -----------------------------------------------------------------
def test_machine_domain_values_match_hand_solution(machine):
    dom, m = machine
    assert m.P.toarray().tolist() == [[0.8, 0.2, 0], [0, 0.5, 0.5], [1, 0, 0]]
    g = 0.95 ** (30 / 365)
    expect = np.linalg.solve(np.eye(3) - g * m.P.toarray(), [10.0, 4.0, -20.0])
    out = evaluate(m, dom, 0.95)
    assert np.allclose(out["V_infinite"], expect, rtol=1e-12) and out["gamma"] == pytest.approx(g)


def test_machine_domain_reachable_split_summary_and_export(machine, tmp_path):
    dom, m = machine
    assert evaluate(m, dom, start="replacement")["n_states"] == 3
    pi0 = initial_distribution(m, dom)
    never, replaced = replacement_split(m, pi0, dom, 24)
    assert all((a + b).sum() == pytest.approx(1.0) for a, b in zip(never, replaced)) and replaced[-1].sum() > 0
    assert evolve_distribution(m, pi0, dom, 24).sum() == pytest.approx(1.0)
    g = group_summary(m, dom, monte_carlo(m, np.zeros(30, dtype=int), 6, n_sims=5, seed=1))
    assert set(g["label"]) <= {"new", "worn", "broken"} and g["mean"].sum() == pytest.approx(30.0)
    export_model(m, dom, str(tmp_path), parquet=False)
    cols = pd.read_csv(tmp_path / "transition_matrix_machine_domain_long.csv").columns
    assert "src_age_bucket" in cols and "src_uim" not in cols          # the domain's own names, not the cow's

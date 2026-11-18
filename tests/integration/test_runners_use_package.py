"""The runner scripts migrated to the package build must reproduce the package model exactly (no second pipeline)."""
import importlib
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from cowmrp.mrp.utils import matrix_checksum


def test_farm_build_engine_equals_package_model(cow):
    sys.path.insert(0, ROOT)
    try:
        build = importlib.import_module("runners.cow.farm.build")
        states, probs, matrix = build.build_engine()
    finally:
        sys.path.remove(ROOT)
    _, m = cow
    assert matrix_checksum(matrix) == matrix_checksum(m.P) and len(states) == len(m.states)


@pytest.mark.parametrize("script,entry", [("runners/cow/projection.py", "compare_cows"), ("runners/cow/plots/plot_validation.py", "build_model"),
                                          ("runners/cow/plots/_plot_data.py", "build_model"), ("runners/cow/farm/build.py", "build_model"),
                                          ("runners/cow/farm/keepcull.py", "keep_cull_table")])
def test_migrated_scripts_use_the_package_not_a_private_pipeline(script, entry):
    text = open(os.path.join(ROOT, script), encoding="utf8").read()
    assert entry in text
    assert "assemble_matrix(" not in text and "compute_transitions(" not in text


def test_run_all_core_modules_succeed():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "runners", "run_all.py"), "--only", "cow,youngstock"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]

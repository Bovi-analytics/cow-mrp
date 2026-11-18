"""Clean-install test: build the wheel, install it into a brand-new virtual environment and use it from an unrelated directory.

Nothing from the development repository may be needed: the child interpreter runs in isolated mode (-I: no PYTHONPATH, no cwd on
sys.path, no user site), its working directory is an empty temp folder, and the package must be imported from site-packages of the new
venv. Dependencies are installed from the package index, so this needs network access (run with `pytest -m install`).
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT

pytestmark = [pytest.mark.install, pytest.mark.slow]

CHILD = textwrap.dedent("""
    import json, os, sys
    import cowmrp
    from cowmrp.domain import load_domain, validate_domain
    from cowmrp.model import build_model, evaluate, matrix_checksum
    from cowmrp.mrp import core
    from cowmrp.io import export_model

    out = {"file": cowmrp.__file__, "version": cowmrp.__version__, "cwd": os.getcwd(),
           "repo_on_path": any(os.path.exists(os.path.join(p or ".", "runners")) for p in sys.path)}
    for name in ("cow", "youngstock"):
        dom = load_domain(name)
        m = build_model(dom)
        out[name] = {"problems": validate_domain(dom), "states": len(m.states), "nnz": int(m.P.nnz), "checksum": matrix_checksum(m.P)}
    cow = load_domain("cow"); m = build_model(cow)
    r = evaluate(m, cow, start="replacement", which=("finite",))
    out["reachable_states"] = r["n_states"]; out["reachable_mean_V"] = float(r["V_finite"].mean())
    files = export_model(m, cow, sys.argv[1], parquet=False)
    out["exported"] = sorted(os.path.basename(f) for f in files)
    print("RESULT" + json.dumps(out))
""")


def _venv_python(venv):
    return os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    work = tmp_path_factory.mktemp("clean_install")
    dist = work / "dist"
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), ROOT], capture_output=True, text=True)
    if r.returncode:
        pytest.skip("cannot build the wheel here (pip install build): " + r.stderr[-300:])
    wheel = next(dist.glob("cowmrp-*.whl"))
    venv = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = _venv_python(str(venv))
    r = subprocess.run([py, "-m", "pip", "install", "--quiet", str(wheel)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip("cannot install the wheel's dependencies (no network?): " + r.stderr[-300:])
    return py, work, wheel


def test_wheel_contains_package_data_and_license(installed):
    import zipfile
    _, _, wheel = installed
    names = zipfile.ZipFile(wheel).namelist()
    assert any(n.endswith("cowmrp/domains/cow/data/README.md") for n in names)
    assert any("licenses/LICENSE" in n or n.endswith("LICENSE") for n in names)
    assert not any(n.startswith(("runners/", "tests/", "docs/", "older_outputs/")) for n in names)       # nothing from the dev repo
    assert not any(".pyc" in n for n in names)


def test_package_works_outside_the_repository(installed, golden):
    py, work, _ = installed
    cwd = work / "elsewhere"
    cwd.mkdir()
    out_dir = work / "exports"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "COW_TIME_UNIT")}
    r = subprocess.run([py, "-I", "-c", CHILD, str(out_dir)], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))[6:])
    assert "site-packages" in res["file"] and os.path.realpath(ROOT) not in os.path.realpath(res["file"])
    assert not res["repo_on_path"]
    assert os.path.realpath(res["cwd"]) == os.path.realpath(str(cwd))
    for name in ("cow", "youngstock"):
        assert res[name]["problems"] == []
        assert (res[name]["states"], res[name]["nnz"], res[name]["checksum"]) == \
               (golden[name]["num_states"], golden[name]["nnz"], golden[name]["matrix_checksum"])
    assert res["reachable_states"] == golden["cow"]["reachable_states"]
    assert res["reachable_mean_V"] == pytest.approx(golden["cow"]["reachable_mean_V_finite"], rel=1e-9)
    assert "transition_matrix_cow_long.csv" in res["exported"]


def test_installed_package_has_no_import_of_repo_layout(installed):
    py, work, _ = installed
    code = "import cowmrp.mrp.core as c, sys; print([m for m in sys.modules if m.startswith(('runners', 'domains', 'mrp.'))])"
    r = subprocess.run([py, "-I", "-c", code], cwd=str(work), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "[]"


FARM_CHILD = textwrap.dedent("""
    import json
    from cowmrp.data import example_herd_path, example_raw_herd_path
    from cowmrp.model import build_model
    from cowmrp.domains.cow.farm import load_herd, project_herd
    from cowmrp.domains.cow.keepcull import keep_cull_table
    m = build_model("cow")
    a, b = load_herd(example_herd_path(), m), load_herd(example_raw_herd_path(), m)
    r = project_herd(a, m, "expected", 12)
    t = keep_cull_table(a, m, (1, 3), offer=900)
    print("RESULT" + json.dumps({"n": len(a), "same_states": a["state_id"].tolist() == b["state_id"].tolist(),
          "cows_kept": float(sum(r["predicted"]["status"])), "n_keepcull": len(t), "has_values": r["values"] is not None}))
""")


def test_installed_package_runs_the_whole_farm_workflow(installed):
    """The tutorial dataset ships in the wheel and the loader, projection and keep/cull work from the installed package (no matplotlib needed)."""
    py, work, _ = installed
    cwd = work / "elsewhere_farm"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "COW_TIME_UNIT")}
    r = subprocess.run([py, "-I", "-c", FARM_CHILD], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))[6:])
    assert res == {"n": 200, "same_states": True, "cows_kept": 200.0, "n_keepcull": 200, "has_values": True}

"""Local installation test: build the wheel and sdist, install the WHEEL into a clean virtual environment and run every example and notebook
against the installed package (never against the development source).

    python check_installed.py [work_dir]        (default: a new folder in the system temp directory; the repository is not modified
                                                  except for the build/ folder setuptools makes, which is removed again)

Steps: 1 build wheel + sdist   2 new venv, pip install the wheel with the plot and excel extras (+ jupyter tools to run notebooks)
       3 check that cowmrp is imported from site-packages, that package data and extras are present   4 run the .py examples
       5 execute the notebooks with nbconvert   6 check that the Plotly files were written and are self-contained.
Nothing is uploaded or published.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="cowmrp_installtest_"))
WORK.mkdir(parents=True, exist_ok=True)
VENV, DIST, RUN = WORK / "venv", WORK / "dist", WORK / "run"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
results = []


def run(cmd, cwd=None, check=True):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)[:200]}")
    return p


def record(name, ok, note=""):
    results.append((name, ok, note))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({note})" if note else ""), flush=True)


print("work folder:", WORK)
# 1 build -----------------------------------------------------------------------------------------------------------------------------
builder = WORK / "builder"
run([sys.executable, "-m", "venv", builder])
bpy = builder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
run([bpy, "-m", "pip", "install", "-q", "build"])
run([bpy, "-m", "build", "--outdir", DIST, REPO])
shutil.rmtree(REPO / "build", ignore_errors=True)
wheel, sdist = next(DIST.glob("*.whl")), next(DIST.glob("*.tar.gz"))
record("build wheel and sdist", True, f"{wheel.name}, {sdist.name}")

# 2 clean venv + install the wheel ------------------------------------------------------------------------------------------------------
run([sys.executable, "-m", "venv", VENV])
run([PY, "-m", "pip", "install", "-q", f"{wheel}[plot,excel]", "nbconvert", "ipykernel"])
record("install wheel into a clean venv", True)

# 3 installed package, package data, extras --------------------------------------------------------------------------------------------
probe = (
    "import json, importlib.util as u, cowmrp, os\n"
    "from cowmrp.data import example_herd_path, example_raw_herd_path\n"
    "out = {'file': cowmrp.__file__, 'herd_exists': os.path.exists(example_herd_path()), 'raw_exists': os.path.exists(example_raw_herd_path()),\n"
    "       'plotly': u.find_spec('plotly') is not None, 'openpyxl': u.find_spec('openpyxl') is not None,\n"
    "       'domains': [d for d in ('cow', 'youngstock') if u.find_spec('cowmrp.domains.' + d)]}\n"
    "print(json.dumps(out))\n"
)
info = json.loads(run([PY, "-I", "-c", probe], cwd=WORK).stdout.strip().splitlines()[-1])
record("cowmrp imported from site-packages, not the repository", "site-packages" in info["file"] and str(REPO / "src") not in info["file"], info["file"])
record("package data present (example herds)", info["herd_exists"] and info["raw_exists"])
record("optional dependencies present (plot, excel)", info["plotly"] and info["openpyxl"])
record("shipped domains found", info["domains"] == ["cow", "youngstock"], str(info["domains"]))

# 4 scripts -------------------------------------------------------------------------------------------------------------------------------
if RUN.exists():
    shutil.rmtree(RUN)
shutil.copytree(HERE, RUN, ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", "check_installed.py"))
for script in sorted(RUN.glob("[0-9][0-9]_*.py")):
    p = run([PY, "-I", script.name], cwd=RUN, check=False)
    record(f"script {script.name}", p.returncode == 0, "" if p.returncode == 0 else p.stderr.strip().splitlines()[-1][:160])
for name in ("keepcull_result.csv", "cow_rewards.csv", "youngstock_distribution.html"):
    record(f"output {name}", (RUN / name).exists())
dash = RUN / "farm_output" / "dashboard_plotly.html"
text = dash.read_text(encoding="utf-8", errors="ignore") if dash.exists() else ""
record("Plotly dashboard written and self-contained (no CDN script)", bool(text) and "<script src=\"http" not in text and "Plotly" in text, f"{len(text) / 1e6:.1f} MB")

# 5 notebooks -----------------------------------------------------------------------------------------------------------------------------
NB = WORK / "notebooks"
if NB.exists():
    shutil.rmtree(NB)
shutil.copytree(HERE, NB, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
shutil.copytree(HERE / "custom_domain", NB / "custom_domain", dirs_exist_ok=True)
prep = """
import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
nb.cells = [c for c in nb.cells if 'pip-install' not in c.metadata.get('tags', [])]
guard = "import cowmrp\\nassert 'site-packages' in cowmrp.__file__, cowmrp.__file__"
nb.cells.insert(0, nbformat.v4.new_code_cell(guard))
nbformat.write(nb, sys.argv[1])
"""   # in each copy: drop the '%pip install' cell (the wheel is already installed) and add a guard cell
for nb in sorted(NB.glob("[0-9][0-9]_*.ipynb")):
    run([PY, "-c", prep, nb], cwd=NB)
    p = run([PY, "-m", "nbconvert", "--to", "notebook", "--execute", "--output", nb.stem + "_executed.ipynb",
             "--ExecutePreprocessor.timeout=600", nb.name], cwd=NB, check=False)
    record(f"notebook {nb.name}", p.returncode == 0, "" if p.returncode == 0 else p.stderr.strip().splitlines()[-1][:160])

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)} of {len(results)} checks passed" + (f"; FAILED: {[r[0] for r in failed]}" if failed else ""))
sys.exit(1 if failed else 0)

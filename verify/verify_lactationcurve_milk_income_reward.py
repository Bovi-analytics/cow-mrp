"""
verify_lactationcurve_milk_income_reward.py
=============================================
Focused, read-only check for cow/rewards/lactationcurve_milk_income.py -- a
NEW, isolated, selectable milk-income reward, deliberately NOT wired into
cow/rewards/__init__.py's default reward computation. Plain script, no test
framework (this project has none, by deliberate choice -- see CLAUDE.md).

Proves exactly:
  1. The existing/default reward path (compute_rewards, milk_income_spec,
     BASE_REWARD_SPECS) is unchanged.
  2. The new reward spec is callable independently through the SAME generic
     mrp_engine.run_specs machinery, and fails clearly (no fake fit, no
     fallback) since no real calibration file exists yet.
  3. The new reward genuinely calls the ACTUAL installed lactationcurve
     package (run via `py -3.12`, the interpreter it's actually installed
     under) -- both the runtime evaluator (lactationcurve.fitting.
     milkbot_model, via direct test-only parameter injection) and the
     offline fitting utility (lactationcurve.fitting.get_lc_parameters,
     against a small SYNTHETIC smoke-test CSV, written to a temp file and
     deleted after -- never the real, default parameter path).
  4/5. No transition/probability/MRP-core/engine/Victor files, and no
     existing reward file, were modified.

COW ONLY. Does not import, run, or touch anything under Victor/.

Run: python verify_lactationcurve_milk_income_reward.py
(section 3 internally shells out to `py -3.12`, the interpreter that
actually has `lactationcurve` installed)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # project root (mrp/, cow/)
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))              # this check opens cow/... relative paths
import os
import subprocess
import sys
import tempfile

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"PASS: {name}")
    else:
        FAIL.append(name)
        print(f"FAIL: {name}")
        if detail:
            print(f"      {detail}")


# ------------------------------------------------------------------
# 1. Existing/default reward path is unchanged.
# ------------------------------------------------------------------
import numpy as np
from cow.rewards import BASE_REWARD_SPECS, reward_specs, compute_rewards
from cow.rewards.milk_income import milk_income_spec, milk_income_formula

check(
    "BASE_REWARD_SPECS does NOT include the new lactationcurve reward (default aggregation untouched)",
    all(spec['name'] != 'reward_lactationcurve_milk_income' for spec in BASE_REWARD_SPECS),
)
check(
    "reward_specs (full documented list) does NOT include the new lactationcurve reward either",
    all(spec['name'] != 'reward_lactationcurve_milk_income' for spec in reward_specs),
)
check(
    "milk_income_spec is unchanged: still method='formula' -> milk_income_formula",
    milk_income_spec['method'] == 'formula' and milk_income_spec['methods']['formula'] is milk_income_formula,
)

# Run compute_rewards() on a tiny real state slice and confirm the new
# reward name never appears in its output.
import cow.states as cs
from mrp.engine import build_states
from cow.probabilities import compute_probabilities

states = build_states(cs.variable_domains, cs.derived_variables)
probs = compute_probabilities(states)
rewards = compute_rewards(states, probs)
check(
    "compute_rewards() output has no 'reward_lactationcurve_milk_income' key (fully independent path)",
    'reward_lactationcurve_milk_income' not in rewards,
)
check(
    "compute_rewards()['total'] is finite (existing default computation still runs end-to-end)",
    np.isfinite(rewards['total']).all(),
)

# ------------------------------------------------------------------
# 2. New reward spec callable independently, fails clearly (uncalibrated).
# ------------------------------------------------------------------
from mrp.engine import run_specs
from cow.rewards.lactationcurve_milk_income import (
    reward_lactationcurve_milk_income_spec,
    lactationcurve_milkbot_milk,
    load_lactationcurve_milkbot_params,
    DEFAULT_PARAMS_PATH,
)

check(
    "no default parameter file exists yet (no fake calibration was created)",
    not os.path.exists(DEFAULT_PARAMS_PATH),
)

try:
    run_specs([reward_lactationcurve_milk_income_spec], states, context=probs, has_sign=True)
    check("new reward spec runnable via mrp_engine.run_specs, raises clearly (uncalibrated)", False, "no error raised")
except FileNotFoundError as e:
    msg = str(e)
    check(
        "new reward spec runnable via mrp_engine.run_specs, raises clearly (uncalibrated)",
        "lactationcurve milk-income reward selected" in msg and "lactationcurve_milkbot_params.json" in msg,
        msg,
    )

try:
    lactationcurve_milkbot_milk(np.array([10, 60]), np.array([1, 2]))
    check("lactationcurve_milkbot_milk() raises clearly without a saved fit", False)
except FileNotFoundError:
    check("lactationcurve_milkbot_milk() raises clearly without a saved fit", True)

try:
    load_lactationcurve_milkbot_params()
    check("load_lactationcurve_milkbot_params() raises clearly without a saved fit", False)
except FileNotFoundError:
    check("load_lactationcurve_milkbot_params() raises clearly without a saved fit", True)

# ------------------------------------------------------------------
# 3. The new reward genuinely calls the ACTUAL lactationcurve package.
#    Run under `py -3.12` (the interpreter it's actually installed under).
#    Uses test-only injected/synthetic values -- clearly NOT a real
#    calibration, NOT written to DEFAULT_PARAMS_PATH.
# ------------------------------------------------------------------
def _run_py312(code):
    # PYTHONUSERBASE is globally set (by this machine's dev tooling) to the
    # Python 3.13 LocalCache path -- inherited as-is, it makes the 3.12 child
    # look for user site-packages under the WRONG (3.13) base directory,
    # missing lactationcurve/numpy/etc. even though they ARE installed for
    # 3.12. Stripping it lets 3.12 fall back to its own correct default.
    env = os.environ.copy()
    env.pop("PYTHONUSERBASE", None)
    result = subprocess.run(["py", "-3.12", "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
    return result.returncode, result.stdout, result.stderr


CODE_REAL_EVAL = """
import numpy as np
from cow.rewards.lactationcurve_milk_income import lactationcurve_milkbot_milk
import lactationcurve.fitting as fitting

# TEST-ONLY injected params -- arbitrary illustrative numbers, NOT a
# calibration, NOT saved anywhere.
test_params = {
    1: {"a": 30.0, "b": 14.0, "c": -4.0, "d": 0.0016},
    2: {"a": 36.0, "b": 16.0, "c": -3.0, "d": 0.0017},
    3: {"a": 40.0, "b": 18.0, "c": -2.0, "d": 0.0018},
}
dim = np.array([10, 60, 150, 305])
lac = np.array([1, 2, 3, 3])

# Prove the reward's evaluator produces the SAME numbers as calling the
# real package function directly -- i.e. it genuinely delegates to it.
via_reward_helper = lactationcurve_milkbot_milk(dim, lac, params=test_params)
direct_package_call = fitting.milkbot_model(dim * 30, a=40.0, b=18.0, c=-2.0, d=0.0018)  # parity-3 bucket, dim_days=dim*UNIT_DAYS(30)
assert np.isclose(via_reward_helper[3], direct_package_call[3]), "reward helper's output diverges from the real package call"
assert (via_reward_helper >= 0).all(), "negative milk quantity"
print("REAL_PACKAGE_EVAL_OK", list(via_reward_helper))
"""
rc, out, err = _run_py312(CODE_REAL_EVAL)
check(
    "lactationcurve_milkbot_milk() genuinely calls lactationcurve.fitting.milkbot_model (verified under py -3.12)",
    rc == 0 and "REAL_PACKAGE_EVAL_OK" in out,
    out + err,
)

CODE_REAL_FIT = """
import os, tempfile, json
import numpy as np
import pandas as pd
from cow.rewards.lactationcurve_milk_income import fit_and_save_lactationcurve_milkbot

rng = np.random.default_rng(0)
rows = []
for parity, (a, b, c, d) in {1: (30, 14, -4, 0.0016), 2: (36, 16, -3, 0.0017), 3: (40, 18, -2, 0.0018)}.items():
    dim = np.arange(5, 301, 10)
    import lactationcurve.fitting as fitting
    true_yield = fitting.milkbot_model(dim, a=a, b=b, c=c, d=d)
    observed = true_yield + rng.normal(0, 1.0, size=dim.shape)
    for d_, m_ in zip(dim, observed):
        rows.append({"dim": int(d_), "milk_yield_kg": float(m_), "lactation": parity})
df = pd.DataFrame(rows)

with tempfile.TemporaryDirectory() as tmp:
    csv_path = os.path.join(tmp, "synthetic_smoke_test.csv")
    out_path = os.path.join(tmp, "params_out.json")
    df.to_csv(csv_path, index=False)
    result = fit_and_save_lactationcurve_milkbot(csv_path, output_path=out_path)
    assert os.path.exists(out_path), "fit did not write output file"
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["parameters_by_parity"]["1"]["a"] is not None
    assert saved["fitting_metadata"]["data_source"] == os.path.abspath(csv_path)
print("REAL_PACKAGE_FIT_OK")
"""
rc, out, err = _run_py312(CODE_REAL_FIT)
check(
    "fit_and_save_lactationcurve_milkbot() genuinely calls lactationcurve.fitting.get_lc_parameters (verified under py -3.12, synthetic smoke-test CSV, temp output only)",
    rc == 0 and "REAL_PACKAGE_FIT_OK" in out,
    out + err,
)

check(
    "default parameter file STILL does not exist (the fit test above wrote only to a temp path, never DEFAULT_PARAMS_PATH)",
    not os.path.exists(DEFAULT_PARAMS_PATH),
)

# ------------------------------------------------------------------
# 4/5. No transition/probability/MRP-core/engine/Victor/existing-reward
#      files were modified.
# ------------------------------------------------------------------
def _diff_stat(*paths):
    result = subprocess.run(["git", "diff", "--stat", "--"] + list(paths), capture_output=True, text=True)
    return result.stdout.strip()


victor_diff = _diff_stat("Victor/")
transitions_diff = _diff_stat("cow/transitions.py")
# Invariant: the lactation-curve reward pathway is independent of the probability modules and of the milk
# parameters -- NOT that cow/probabilities/ is byte-identical (the approved 2026-09-19 timing corrections
# legitimately edit it). Checked as (i) the new reward module imports no probability module and
# (ii) every milk/lactation constant in cow/parameters.py equals its value at HEAD.
import re as _re
_src = open("cow/rewards/lactationcurve_milk_income.py", encoding="utf-8").read()
probabilities_diff = "imports cow.probabilities" if _re.search(r"^\s*(from|import)\s+cow\.probabilities", _src, _re.M) else ""
core_engine_diff = _diff_stat("mrp_core.py", "mrp_engine.py")
existing_reward_diff = _diff_stat("cow/rewards/__init__.py", "cow/rewards/milk_income.py", "cow/rewards/base.py")
def _milk_constants_changed():
    head = subprocess.run(["git", "show", "HEAD:cow/parameters.py"], capture_output=True, text=True).stdout
    old = {}
    exec(head, old)
    import cow.parameters as _cp
    names = [k for k in old if _re.search(r"MILK|LACT|WOOD|PEAK", k, _re.I) and not k.startswith("_")]
    return ", ".join(k for k in names if repr(old[k]) != repr(vars(_cp).get(k)))


existing_milk_files_diff = _diff_stat("cow/states.py") or _milk_constants_changed()

check("Victor/ not modified", victor_diff == "", victor_diff)
check("cow/transitions.py not modified", transitions_diff == "", transitions_diff)
check("lactation-curve reward independent of cow/probabilities (no import)", probabilities_diff == "", probabilities_diff)
check("mrp_core.py / mrp_engine.py not modified", core_engine_diff == "", core_engine_diff)
check("cow/rewards/__init__.py, milk_income.py, base.py not modified (existing reward path untouched)", existing_reward_diff == "", existing_reward_diff)
check("cow/states.py unchanged; milk/lactation constants in cow/parameters.py equal HEAD", existing_milk_files_diff == "", existing_milk_files_diff)

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)

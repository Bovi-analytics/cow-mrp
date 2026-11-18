"""
verify_state_sufficiency.py
===========================
Formal check that the state (lactation, dim, dp) is sufficient for every transition and reward: nothing in the
model reads hidden per-cow information (stored health, DNB, milk category, previous days open, previous pregnancy loss).

Four independent checks:
  1. DECLARED: every probability spec and reward spec lists only lactation / dim / dp as state dependencies.
  2. TABLE: the state table has exactly the columns state_id, lactation, dim, dp.
  3. STATIC: no code in cow/ reads a `states[...]` column other than lactation / dim / dp (AST scan), and no
     identifier that would store history (health, dnb, milk_category, previous_*, prior_*) is a state column or parameter name.
  4. BEHAVIOURAL: every probability and every base reward is a pure function of the row's (lactation, dim, dp):
     shuffling the rows and re-labelling state_id gives identical values per (lactation, dim, dp), and duplicating the table
     gives identical values in both copies (no cross-row or order dependence).

    python runners/cow/checks/verify_state_sufficiency.py        (COW_TIME_UNIT=month|week|day)
"""
import os, sys, ast
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.environ.setdefault("COW_TIME_UNIT", "month")

import numpy as np
import pandas as pd

STATE_VARS = {"lactation", "dim", "dp"}
failed = []


def ok(cond, msg):
    print(("[OK] " if cond else "[FAIL] ") + msg)
    if not cond:
        failed.append(msg)


from cowmrp.domains.cow.states import build_cow_states
from cowmrp.domains.cow.probabilities import compute_probabilities, probability_specs
from cowmrp.domains.cow.rewards import compute_rewards, BASE_REWARD_SPECS

# 1. declared dependencies
for s in probability_specs:
    extra = set(s["depends_on"]) - STATE_VARS
    ok(not extra, f"probability '{s['name']}' depends only on state variables {sorted(s['depends_on'])}")
for s in BASE_REWARD_SPECS:
    st = set(s["depends_on"]["state"]) | set(s["depends_on"]["derived_state"])
    ok(st <= STATE_VARS, f"reward '{s['name']}' state dependencies {sorted(st)} are within (lactation, dim, dp)")

# 2. table columns
states = build_cow_states()
ok(set(states.columns) == STATE_VARS | {"state_id"}, f"state table columns are exactly {sorted(states.columns)}")

# 3. static scan of cow/ for states[...] column reads and history-like names
used, bad_names = set(), []
FORBIDDEN = ("health", "dnb", "milk_category", "previous_days_open", "prev_days_open", "previous_loss", "prior_loss", "n_services", "days_since_loss")
for dirpath, _, files in os.walk(os.path.join(ROOT, "src", "cowmrp", "domains", "cow")):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "states":
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    used.add(sl.value)
            ident = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.attr if isinstance(node, ast.Attribute) else None
            if ident and any(w in ident.lower() for w in FORBIDDEN):
                bad_names.append((os.path.relpath(path, ROOT), ident))
ok(used <= STATE_VARS | {"state_id"}, f"code reads only these state columns: {sorted(used)}")
ok(not bad_names, f"no history-storing identifiers in cow/ code (checked: {', '.join(FORBIDDEN)})" + (f"  FOUND: {bad_names[:5]}" if bad_names else ""))

# 4. behavioural purity
rng = np.random.default_rng(0)
base_p = compute_probabilities(states)
base_r = compute_rewards(states, base_p)
key = pd.MultiIndex.from_frame(states[["lactation", "dim", "dp"]])
perm = rng.permutation(len(states))
shuf = states.iloc[perm].reset_index(drop=True).copy()
shuf["state_id"] = rng.permutation(len(shuf)) + 1000
p2 = compute_probabilities(shuf)
r2 = compute_rewards(shuf, p2)
key2 = pd.MultiIndex.from_frame(shuf[["lactation", "dim", "dp"]])
worst = 0.0
for name in base_p:
    a = pd.Series(np.asarray(base_p[name], float), index=key)
    b = pd.Series(np.asarray(p2[name], float), index=key2).reindex(key)
    worst = max(worst, float(np.nanmax(np.abs(a.values - b.values))))
for name in ("total",) + tuple(s["name"] for s in BASE_REWARD_SPECS):
    a = pd.Series(np.asarray(base_r[name], float), index=key)
    b = pd.Series(np.asarray(r2[name], float), index=key2).reindex(key)
    worst = max(worst, float(np.nanmax(np.abs(a.values - b.values))))
ok(worst <= 1e-12, f"probabilities and rewards identical after shuffling rows and relabelling state_id (max abs diff {worst:.1e})")

dbl = pd.concat([states, states], ignore_index=True)
dbl["state_id"] = np.arange(len(dbl))
p3 = compute_probabilities(dbl)
r3 = compute_rewards(dbl, p3)
n = len(states)
worst = 0.0
for name in base_p:
    v = np.asarray(p3[name], float)
    worst = max(worst, float(np.abs(v[:n] - v[n:]).max()))
for name in ("total",) + tuple(s["name"] for s in BASE_REWARD_SPECS):
    v = np.asarray(r3[name], float)
    worst = max(worst, float(np.abs(v[:n] - v[n:]).max()))
ok(worst <= 1e-12, f"duplicated table gives identical values in both copies (max abs diff {worst:.1e}): no cross-row dependence")

print("\nSTATE_SUFFICIENCY_OK" if not failed else f"\nSTATE_SUFFICIENCY_FAILED: {len(failed)} check(s)")
sys.exit(1 if failed else 0)

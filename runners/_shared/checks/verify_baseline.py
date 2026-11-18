"""
runners/_shared/checks/verify_baseline.py
=========================================
Parity check for ANY module: after a change to mrp/ (or to a domain) compare the module's model manifest
(runners/<module>/outputs/model/<unit>/manifest.json, written by engine_run.py) with its saved baseline
(runners/<module>/outputs/baseline/model_<unit>.json).

    python runners/_shared/checks/verify_baseline.py [module ...] [--time-unit month|week|day]     # default: every module with a baseline
    python runners/_shared/checks/verify_baseline.py cow --save                                      # (re)write the baseline from the current manifest

Compared: num_states, nnz_transitions and matrix_checksum exactly; every other number (value means, ...) to relative 1e-9.
A baseline is only ever written with --save, after you have decided the change is the intended one: never silently.
Run the module's run.py first so the manifest is current. Exit code 1 on any difference.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from runners._shared import common

EXACT = ("num_states", "nnz_transitions", "matrix_checksum")
SKIP = ("created_utc", "module")


def paths(module, unit):
    base = os.path.join(ROOT, "runners", module, "outputs")
    return os.path.join(base, "model", unit, "manifest.json"), os.path.join(base, "baseline", f"model_{unit}.json")


def compare(cur, ref, tol=1e-9):
    diffs = []
    for k, rv in ref.items():
        if k in SKIP:
            continue
        cv = cur.get(k)
        if k in EXACT or not isinstance(rv, (int, float)) or isinstance(rv, bool):
            if cv != rv:
                diffs.append(f"{k}: {cv} != baseline {rv}")
        elif cv is None or abs(cv - rv) > tol * max(1.0, abs(rv)):
            diffs.append(f"{k}: {cv} != baseline {rv}")
    return diffs


def main(argv):
    unit = argv[argv.index("--time-unit") + 1] if "--time-unit" in argv else "month"
    save = "--save" in argv
    mods = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--time-unit")]
    if not mods:
        base = os.path.join(ROOT, "runners")
        mods = sorted(m for m in os.listdir(base) if os.path.isfile(os.path.join(base, m, "outputs", "baseline", f"model_{unit}.json")))
    if not mods:
        print("no module has a baseline yet; create one with: verify_baseline.py <module> --save")
        return 1
    failed = 0
    for m in mods:
        man_p, base_p = paths(m, unit)
        if not os.path.isfile(man_p):
            print(f"[{m}] FAIL: no manifest at {os.path.relpath(man_p, ROOT)}; run runners/{m}/run.py --time-unit {unit}")
            failed += 1
            continue
        cur = json.load(open(man_p, encoding="utf-8"))
        if save:
            os.makedirs(os.path.dirname(base_p), exist_ok=True)
            json.dump({k: v for k, v in cur.items() if k not in SKIP}, open(base_p, "w", encoding="utf-8"), indent=1)
            print(f"[{m}] baseline written: {os.path.relpath(base_p, ROOT)}")
            continue
        if not os.path.isfile(base_p):
            print(f"[{m}] FAIL: no baseline; after checking the model, create it with --save")
            failed += 1
            continue
        diffs = compare(cur, json.load(open(base_p, encoding="utf-8")))
        print(f"[{m}] " + ("OK" if not diffs else "DIFFERENT:\n   " + "\n   ".join(diffs)))
        failed += bool(diffs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

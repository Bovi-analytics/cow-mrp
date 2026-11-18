"""
runners/run_all.py
==================
Run every module runner (runners/<module>/run.py) that exists, so a domain added with runners/_shared/new_domain.py is
picked up automatically. Each module writes its own runners/<module>/outputs/.

    python runners/run_all.py [--time-unit month|week|day] [--only cow,youngstock]
    python runners/run_all.py --views [--only cow,youngstock] [--skip validation,farm] [--steps 24] [--n-cows 10000]    -> views/index.html

A module whose domain is still a stub (NotImplementedError) is reported as "not implemented yet" and does not stop the others.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def modules():
    base = os.path.join(ROOT, "runners")
    return sorted(m for m in os.listdir(base) if not m.startswith("_") and os.path.isfile(os.path.join(base, m, "run.py")))


def main(argv):
    if "--views" in argv:        # the local report folder views/ and its central page (runners/_shared/views.py); other options pass through
        sys.path.insert(0, ROOT)
        from runners._shared import views
        return views.main([x for x in argv if x != "--views"])
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    unit = ["--time-unit", argv[argv.index("--time-unit") + 1]] if "--time-unit" in argv else []
    status = {}
    for m in modules():
        if only and m not in only:
            continue
        print(f"\n=== {m} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "runners", m, "run.py"), *unit], cwd=ROOT, capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode == 0:
            status[m] = "ok"
        elif "NotImplementedError" in r.stderr:
            status[m] = "not implemented yet"
        else:
            status[m] = "FAILED"
            print(r.stderr[-1500:])
    print("\nsummary:\n" + "\n".join(f"  {m:12s} {s}" for m, s in status.items()))
    return 1 if "FAILED" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

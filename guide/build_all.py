"""guide/build_all.py -- one command that brings every document in step with the code.

    python guide/build_all.py            write generated blocks, build both sites, then check that nothing is stale
    python guide/build_all.py --examples also re-run the examples first (slow; refreshes the example outputs of the web site)

Steps: guide/sync.py write -> guide/web/build_web.py -> guide/build.py -> guide/sync.py check (+ the web methods check). Stops at the first failure.
Needs the documentation requirements (pip install -r guide/requirements-docs.txt) for the API pages.
"""
import os
import subprocess
import sys

GUIDE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GUIDE)
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONIOENCODING="utf8")


def run(*args):
    print("\n$ python " + " ".join(os.path.relpath(a, ROOT) if os.path.isabs(a) else a for a in args))
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    run(os.path.join(GUIDE, "sync.py"), "write")
    if "--examples" in sys.argv:
        run(os.path.join(GUIDE, "web", "build_web.py"), "--examples")
    run(os.path.join(GUIDE, "web", "build_web.py"))
    run(os.path.join(GUIDE, "build.py"))
    for chk in ("web/check_methods.py",):
        if os.path.exists(os.path.join(GUIDE, chk)):
            run(os.path.join(GUIDE, chk))
    run(os.path.join(GUIDE, "sync.py"), "check")


if __name__ == "__main__":
    main()

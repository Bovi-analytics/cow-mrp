"""
runners/_shared/run_notebook.py
===============================
Run the one shared notebook (runners/_shared/notebook/mrp_engine.ipynb) for any domain. The notebook holds no domain logic: it
reads the domain name from the environment variable MRP_DOMAIN and loads everything from domains/<name>/ (and, optionally,
domains/<name>/notebook_hooks.py).

    python runners/_shared/run_notebook.py <domain> [--time-unit month|week|day]
    python runners/<module>/run.py --notebook            (the module runners call run() below)

OUTPUT  runners/<domain>/outputs/notebook/
    mrp_engine_<domain>.executed.ipynb      the notebook with its outputs
    transition_matrix_<domain>*, reward_vector_<domain>*   the notebook's own exports (Step 8)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners._shared import common

NOTEBOOK = os.path.join(common.project_root(), "runners", "_shared", "notebook", "mrp_engine.ipynb")


def run(domain, time_unit=None):
    """Execute the shared notebook for `domain`; returns the path of the executed copy."""
    out = common.module_outputs(domain, "notebook")             # also checks that runners/<domain>/ exists
    env = dict(os.environ, MRP_DOMAIN=domain, MRP_OUT_DIR=out)      # the tutorial notebook writes its exports to MRP_OUT_DIR
    # the notebook kernel may be a different interpreter than this one: let it find the cowmrp source tree (no install needed)
    src = os.path.join(common.project_root(), "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if time_unit:
        env["COW_TIME_UNIT"] = time_unit
    name = f"mrp_engine_{domain}.executed"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", NOTEBOOK,
                    "--output-dir", out, "--output", name, "--ExecutePreprocessor.timeout=-1"],
                   cwd=common.project_root(), env=env, check=True)
    return os.path.join(out, name + ".ipynb")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    unit = args[args.index("--time-unit") + 1] if "--time-unit" in args else None
    print(f"executed notebook: {run(args[0], unit)}")

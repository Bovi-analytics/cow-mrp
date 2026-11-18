"""
runners/_shared/new_domain.py
=============================
Add a new domain in one command. It creates the domain package and its runner module, so the new domain has its own
`run.py`, `checks/`, `plots/` and `outputs/` from the start and is picked up by runners/run_all.py.

    python runners/_shared/new_domain.py <name>        # e.g. python runners/_shared/new_domain.py pigs

Creates (nothing existing is overwritten):
    src/cowmrp/domains/<name>/{__init__,states,probabilities,transitions}.py     stubs: the three functions the engine needs
    runners/<name>/{__init__.py, run.py, README.md, checks/, plots/, outputs/<kind>/}

The stubs raise NotImplementedError until the domain fills them in. The model logic lives in domains/<name>; run.py only
calls it through runners/_shared/engine_run.py (generic engine mrp/) and writes the standard outputs.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DOMAIN_FILES = {
    "__init__.py": '"""The @NAME@ domain: states, probabilities, rewards and transitions for the generic engine (mrp/)."""\n',
    "states.py": (
        '"""States of the @NAME@ domain."""\n'
        "import pandas as pd\n\n\n"
        "def build_@NAME@_states() -> pd.DataFrame:\n"
        '    """One row per state, with a `state_id` column (0..n-1) and the state variables as columns."""\n'
        '    raise NotImplementedError("define the @NAME@ state variables and build the state table")\n'
    ),
    "probabilities.py": (
        '"""Probabilities of the @NAME@ domain (specs resolved by the engine)."""\n\n\n'
        "def compute_probabilities(states):\n"
        '    """Return the dict of per-state probability arrays used by compute_transitions."""\n'
        '    raise NotImplementedError("define the @NAME@ probability specs")\n'
    ),
    "transitions.py": (
        '"""Transition matrix of the @NAME@ domain."""\n\n\n'
        "def compute_transitions(states, probs):\n"
        '    """Return (rows, cols, vals) for cowmrp.mrp.core.assemble_matrix; each row must sum to 1."""\n'
        '    raise NotImplementedError("define the @NAME@ transitions")\n'
    ),
}

RUN_PY = '''"""
runners/@NAME@/run.py
The @NAME@ module in one command: build the @NAME@ MRP with the generic engine (mrp/), check it (rows sum to 1) and write the
standard outputs to runners/@NAME@/outputs/model/<unit>/ (states.csv, transition matrix csv/parquet/npz, reward_vector_@NAME@.csv
if the domain defines rewards, manifest.json with counts and a matrix checksum).

    python runners/@NAME@/run.py [--time-unit month|week|day] [--notebook]

No model logic here: it all comes from domains/@NAME@; the build/export code is shared (runners/_shared/engine_run.py).
After checking the model once: python runners/_shared/checks/verify_baseline.py @NAME@ --save  (then the same command without
--save tells you whether a later change to mrp/ or to the domain moved the model).
"""
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root
from runners._shared import common
common.set_time_unit_from_argv()

from runners._shared import engine_run, run_notebook

if __name__ == "__main__":
    if "--notebook" in _sys.argv:    # the shared notebook (runners/_shared/notebook) for this domain instead of this script
        print(f"executed notebook: {run_notebook.run('@NAME@')}")
        _sys.exit(0)
    model, out = engine_run.run_domain("@NAME@")
    print(f"@NAME@ module: {len(model.states):,} states, {model.P.nnz:,} transitions -> {out}")
'''

README = (
    "# @NAME@ runner\n\n"
    "`run.py` builds the @NAME@ model and writes `outputs/model/<unit>/`.\n"
    "`checks/` = verification scripts, `plots/` = figures, `outputs/<kind>/` = generated files (never hand-edited).\n"
    "Model logic: `domains/@NAME@/`. Output kinds: see `KINDS` in `runners/_shared/common.py`.\n"
)


def create(name):
    if not name.isidentifier() or name.startswith("_"):
        raise SystemExit(f"domain name must be a plain identifier (letters, digits, underscore): {name!r}")
    made = []

    def write(path, text):
        path = os.path.join(ROOT, path)
        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace("@NAME@", name))
        made.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))

    for fn, text in DOMAIN_FILES.items():
        write(f"src/cowmrp/domains/{name}/{fn}", text)
    write(f"runners/{name}/__init__.py", "")
    write(f"runners/{name}/run.py", RUN_PY)
    write(f"runners/{name}/README.md", README)
    for sub in ("checks", "plots"):
        write(f"runners/{name}/{sub}/__init__.py", "")
    for kind in ("model", "validation", "audit"):
        write(f"runners/{name}/outputs/{kind}/.gitkeep", "")
    print("created:\n  " + "\n  ".join(made) if made else "nothing to create (already exists)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    create(sys.argv[1])

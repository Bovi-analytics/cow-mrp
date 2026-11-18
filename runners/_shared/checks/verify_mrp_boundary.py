"""
runners/_shared/checks/verify_mrp_boundary.py
=============================================
The MRP engine (src/cowmrp/mrp) must stay domain-free:
  1. it imports nothing from cowmrp outside cowmrp.mrp (no domain, model, io, viz, runner) and no runners/;
  2. its code (not its prose docstrings) names no domain column or state variable.
    python runners/_shared/checks/verify_mrp_boundary.py
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MRP = os.path.join(ROOT, "src", "cowmrp", "mrp")
FORBIDDEN_IMPORT = re.compile(r"^(cowmrp\.(domain|domains|model|io|viz|runner)|runners|domains|couplings)(\.|$)")
FORBIDDEN_WORDS = {"dim", "dp", "lactation", "uim", "up", "cow", "youngstock", "heifer", "calf"}


def main():
    problems = []
    for path in sorted(glob.glob(os.path.join(MRP, "*.py"))):
        rel = os.path.relpath(path, ROOT)
        tree = ast.parse(open(path, encoding="utf8").read())
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            problems += [f"{rel}:{node.lineno} imports {m}" for m in mods if FORBIDDEN_IMPORT.match(m)]
            # string literals used as code (dict keys, column names), not docstrings
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_WORDS:
                problems.append(f"{rel}:{node.lineno} hardcodes the domain name {node.value!r}")
    for p in problems:
        print("FAIL", p)
    print(f"mrp boundary: {'OK' if not problems else str(len(problems)) + ' problem(s)'} ({len(glob.glob(os.path.join(MRP, '*.py')))} files checked)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""
audit_tables.py
===============
Evidence audit of every registered full lookup table (audit P4): for each table and each lactation 1..MAX_LACTATION,
how many rows are `source` / `derived` / `repeated` / `placeholder`, with the value range, so the evidence behind each
cell is visible and a copied value can never pass for an observation. Also reports the reward tables.

    python verify/audit_tables.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cow.probabilities, cow.rewards                      # register every table
from cow.lookup_tables import FULL_TABLE_REGISTRY, get_full_table, BASIS_CODES
import cow.parameters as P

for name in sorted(FULL_TABLE_REGISTRY):
    spec = FULL_TABLE_REGISTRY[name]
    df = get_full_table(name)
    col = spec["value_col"]
    print(f"\n=== {name}   (statuses: {spec['statuses'] or 'none'}, native months: {spec['n_months']}, value column: {col})")
    print(f"{'lac':>3} | " + " ".join(f"{b:>11s}" for b in BASIS_CODES) + " |   min .. max")
    for L in range(1, P.MAX_LACTATION + 1):
        d = df[df["lactation"] == L]
        counts = {b: int((d["basis"] == b).sum()) for b in BASIS_CODES}
        print(f"{L:>3} | " + " ".join(f"{counts[b]:>11d}" for b in BASIS_CODES) + f" | {d[col].min():.5g} .. {d[col].max():.5g}")
    notes = df.groupby("basis")["note"].first()
    for b, n in notes.items():
        print(f"    [{b}] {n[:230]}")

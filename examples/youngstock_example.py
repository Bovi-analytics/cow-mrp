"""Youngstock -- where a newborn heifer calf is after 6, 12, 24 and 36 months.

    python examples/youngstock_example.py

The youngstock chain (calf -> heifer -> first calving) is a second domain on the same engine. Starting from the newborn calf state,
the probability of being in each stage is carried forward with the chain's own transition matrix.
"""
import numpy as np
import pandas as pd

from cowmrp.domain import load_domain
from cowmrp.model import build_model, evolve_distribution, initial_distribution

dom = load_domain("youngstock")
model = build_model(dom)
print(f"{len(model.states)} states, {model.P.nnz} transitions\n")

pi0 = initial_distribution(model, dom)               # all probability on the domain's initial state (newborn calf)
rows = {}
for months in (6, 12, 24, 36):
    pi = evolve_distribution(model, pi0, dom, steps=months)
    rows[f"{months} months"] = pd.Series(pi).groupby(model.states["stage"].values).sum()
table = pd.DataFrame(rows).fillna(0.0)
print("Probability of each stage (rows) after the given time (columns):")
print(table.round(3).to_string())

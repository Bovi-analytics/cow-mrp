# %% [markdown]
# # 4. Young stock: a farm of 500 young animals
#
# The young stock domain (`"youngstock"`) follows a female calf through calf, heifer, breeding and pregnant stages until first calving
# (`GRADUATED`, ready to enter the milking herd) or until she leaves (`DIED`, `CULLED`, `SOLD`, `STILLBORN`). It has states, probabilities and a
# transition matrix, but **no rewards**, so the questions it answers are about numbers and timing, not money.
#
# **Your own data:** a CSV with `stage` (CALF, HEIFER, BREEDING or PREGNANT), `age_months` (CALF 0-2, HEIFER 3-15, else 0) and `dp_months`
# (months pregnant, 1-10, only for PREGNANT, else 0). The file in `data/youngstock_500.csv` is **synthetic** (random, fixed seed), not real animals.

# %% [markdown]
# ## Install
# Run this once (in a terminal: `pip install "cowmrp[plot]"`). Then restart the kernel if the package was not installed before.

# %%
# %pip install -q "cowmrp[plot]"

# %%
import os
os.environ.setdefault("COW_TIME_UNIT", "month")      # the CSV of this notebook is in months: keep month resolution (the chain also exists at week and day, with other state ages)

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cowmrp.domain import load_domain
from cowmrp.model import build_model, evolve_distribution

HERE = Path("data")                    # the data/ folder next to this notebook
dom = load_domain("youngstock")
model = build_model(dom)
states = model.states
assert dom.time_unit == "month", "the youngstock CSV of this notebook has ages in months: set COW_TIME_UNIT=month (the default) before importing cowmrp"
print(f"{len(states)} states, {model.P.nnz} transitions")
states.groupby("stage").agg(states=("age", "size"), first_age=("age", "min"), last_age=("age", "max"), max_dp=("dp", "max"))

# %% [markdown]
# ## Map the animals to states

# %%
animals = pd.read_csv(HERE / "youngstock_500.csv")                  # <- or your own file
index = {(s, a, d): i for s, a, d, i in zip(states["stage"], states["age"], states["dp"], states["state_id"])}
missing = [(r.stage, r.age_months, r.dp_months) for r in animals.itertuples() if (r.stage, r.age_months, r.dp_months) not in index]
assert not missing, f"not a state of the model: {sorted(set(missing))[:5]}"
animals["state_id"] = [index[(r.stage, r.age_months, r.dp_months)] for r in animals.itertuples()]
animals["stage"].value_counts()

# %% [markdown]
# ## Where will they be? Expected animals per stage
# The vector of animals per state is moved forward with the chain's own transition matrix.

# %%
pi0 = np.bincount(animals["state_id"], minlength=len(states)).astype(float)          # animals per state (a count, not a probability)
months = list(range(0, 37))
by_stage = pd.DataFrame(
    [pd.Series(evolve_distribution(model, pi0, dom, steps=m)).groupby(states["stage"].values).sum() for m in months], index=months)
by_stage.loc[[0, 6, 12, 24, 36]].round(1)

# %% [markdown]
# ## Figure (Plotly)

# %%
fig = go.Figure()
for stage in by_stage.columns:
    fig.add_trace(go.Scatter(x=by_stage.index, y=by_stage[stage], stackgroup="one", name=stage))
fig.update_layout(title="500 young animals: expected number in each stage", xaxis_title="months from now", yaxis_title="animals")
fig

# %%
fig.write_html("youngstock_distribution.html", include_plotlyjs=True)
final = by_stage.loc[36]
print(f"After 36 months: {final['GRADUATED']:.0f} reached first calving; "
      f"{final[['DIED', 'CULLED', 'SOLD', 'STILLBORN']].sum():.0f} left (died, culled, sold, stillborn); {final[['CALF', 'HEIFER', 'BREEDING', 'PREGNANT']].sum():.0f} still growing.")

# %% [markdown]
# ## One animal
# The chance that a newborn calf (state 0) is alive and has reached first calving after 3 years:

# %%
newborn = np.zeros(len(states)); newborn[0] = 1.0
after = pd.Series(evolve_distribution(model, newborn, dom, steps=36)).groupby(states["stage"].values).sum()
print(after.round(3).to_string())

# %% [markdown]
# ## Limits
# * No rewards: `evaluate()` needs them, so there is no value or keep/cull for young stock in this package.
# * Farm settings such as the share of conventional, sexed and beef semen (`BREEDING_MIX`) are constants of the domain; changing them means
#   editing `cowmrp/domains/youngstock/parameters.py`, they have no run-time switch yet.

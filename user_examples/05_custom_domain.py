# %% [markdown]
# # 5. Your own domain on the generic engine: a toy insemination model
#
# CowMRP's engine (`cowmrp.mrp`) knows nothing about cows: it only needs a **domain**, a folder of Python files that defines states,
# probabilities, transitions and (optionally) rewards. `custom_domain/insemination/` is such a folder: a cow is open for up to 6 months (each
# month she may conceive), then pregnant for 9 months, then calves and starts again; if still open after 6 months she is culled and replaced.
#
# **Package versus example.** The package provides the generic engine and its public interface: `load_domain`, `validate_domain`, `build_model` (assembles the matrix, checks that rows sum to 1),
# `evaluate` (state values), `evolve_distribution` and `initial_distribution`. The example folder `custom_domain/insemination/` is the user's side: it defines the states, the probabilities,
# the transitions and the rewards. Nothing of the engine is re-implemented here, and no CowMRP source file is edited.
#
# **The numbers in this toy domain are illustrative, not biological or economic estimates.** Read `custom_domain/README.md` for the rules a
# domain folder must follow.

# %% [markdown]
# ## Install
# Run this once (in a terminal: `pip install "cowmrp[plot]"`). Then restart the kernel if the package was not installed before.

# %%
# %pip install -q "cowmrp[plot]"

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = Path(".")                       # the folder that contains custom_domain/
sys.path.insert(0, str(HERE / "custom_domain"))           # makes the folder `insemination` importable; any location on sys.path works

from cowmrp.domain import load_domain, validate_domain
from cowmrp.model import build_model, evaluate, evolve_distribution, initial_distribution

dom = load_domain("insemination")                          # the folder name is the domain name
print("interface problems:", validate_domain(dom) or "none")

# %% [markdown]
# ## 1. States -> probabilities -> transition matrix -> reward vector
# `build_model` calls the domain's four functions in order and checks that every row of the matrix sums to 1.

# %%
model = build_model(dom)
states = model.states
print(f"{len(states)} states, {model.P.nnz} non-zero transitions")
states.assign(**{k: v.round(3) for k, v in model.probs.items()}).assign(reward_total=model.rewards["total"].round(1))

# %%
P_dense = pd.DataFrame(model.P.toarray(), index=[f"{p}{k}" for p, k in zip(states.phase, states.k)], columns=[f"{p}{k}" for p, k in zip(states.phase, states.k)])
print("row sums:", sorted(set(P_dense.sum(axis=1).round(10))))
P_dense.round(2).loc[["open0", "open5", "pregnant9"]].loc[:, lambda d: d.sum() > 0]

# %% [markdown]
# ## 2. State values
# The expected discounted reward from each state, for ever (`V_infinite`) and over 3 years (`V_finite`), at 5 % annual discount.

# %%
result = evaluate(model, dom, gamma_annual=0.95, years=3)
values = states[["phase", "k"]].assign(V_infinite=result["V_infinite"].round(1), V_3y=result["V_finite"].round(1))
values

# %%
fig = go.Figure(go.Bar(x=[f"{p} {k}" for p, k in zip(states.phase, states.k)], y=result["V_infinite"]))
fig.update_layout(title="Value of each state (toy insemination domain)", yaxis_title="USD, discounted")
fig

# %% [markdown]
# ## 3. Where is a cow after n months?
# Starting at open month 0, the distribution over states is moved forward with the same matrix.

# %%
pi0 = initial_distribution(model, dom)                      # from the domain's INITIAL_STATE hook
after = pd.DataFrame({m: evolve_distribution(model, pi0, dom, steps=m) for m in (3, 9, 12, 24)}, index=[f"{p}{k}" for p, k in zip(states.phase, states.k)])
after.round(3)

# %% [markdown]
# ## 4. Change an input and rebuild
# The parameters are plain module constants. Raise the breeding cost, rebuild, compare. (Restart the session to go back to the defaults.)

# %%
import insemination.parameters as params

before = evaluate(build_model(dom), dom, years=3)["V_infinite"][0]
params.BREEDING_COST_PER_OPEN_MONTH = 80.0
after_cost = evaluate(build_model(dom), dom, years=3)["V_infinite"][0]
params.BREEDING_COST_PER_OPEN_MONTH = 40.0
print(f"value of a cow at open month 0: {before:,.0f} USD with a breeding cost of 40, {after_cost:,.0f} USD with 80")

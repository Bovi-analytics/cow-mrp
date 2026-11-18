# %% [markdown]
# # 3. Individual rewards: what each state earns and costs per step
#
# The cow model's reward for a state is a **vector of components**: `milk_income`, `pregnancy_reward`, `cull_reward`, `mortality_cost`,
# `feed_cost`, `reproduction_cost`, and `total` = milk_income + pregnancy_reward + cull_reward - mortality_cost - feed_cost - reproduction_cost
# (the costs are stored as positive numbers and subtracted). Values are USD per model step (one month at month resolution).
# Replacement cost is not in the vector: it is charged in the cow-versus-heifer comparison (see example 1).
#
# The values come from the installed package (`build_model("cow").rewards`); this notebook only selects rows and adds them up as a check. The economic defaults behind them
# (milk price, feed price, calf value, ...) are model inputs, not validated farm economics; example 1 lists them with their recorded sources and shows how to change them.

# %% [markdown]
# ## Install
# Run this once (in a terminal: `pip install "cowmrp[plot,excel]"`). Then restart the kernel if the package was not installed before.

# %%
# %pip install -q "cowmrp[plot,excel]"

# %%
import os
os.environ.setdefault("COW_TIME_UNIT", "month")

from pathlib import Path

import pandas as pd

from cowmrp.domains.cow.farm import load_herd
from cowmrp.model import build_model

HERE = Path("data")                    # the data/ folder next to this notebook
model = build_model("cow")
herd = load_herd(HERE / "cows_1000.csv", model)       # <- or your own CSV
COMPONENTS = ["milk_income", "pregnancy_reward", "cull_reward", "mortality_cost", "feed_cost", "reproduction_cost"]
COSTS = ["mortality_cost", "feed_cost", "reproduction_cost"]       # stored positive, subtracted in `total`

# %% [markdown]
# ## One table: every state with its reward components

# %%
rewards = pd.DataFrame({c: model.rewards[c] for c in COMPONENTS + ["total"]})
state_rewards = pd.concat([model.states[["state_id", "lactation", "dim", "dp"]], rewards], axis=1)
state_rewards.sample(5, random_state=1).round(2)

# %% [markdown]
# ## The reward of each cow in your file
# A cow's reward is the reward of the state she maps to. The components add up to `total` once the three costs are subtracted.

# %%
cow_rewards = herd[["cow_id", "lactation", "dim_days", "dp_days", "status", "state_id"]].copy()
for c in COMPONENTS + ["total"]:
    cow_rewards[c] = model.rewards[c][cow_rewards["state_id"].values]
signed = cow_rewards[COMPONENTS].sum(axis=1) - 2 * cow_rewards[COSTS].sum(axis=1)
print("largest difference between the signed sum of components and total:", float((signed - cow_rewards["total"]).abs().max()))
cow_rewards.head(10).round(2)

# %% [markdown]
# ## One component only
# Average milk income and feed cost per cow and step, by lactation and status.

# %%
cow_rewards.groupby(["lactation", "status"])[["milk_income", "feed_cost", "total"]].mean().round(1).head(12)

# %%
cow_rewards.to_csv("cow_rewards.csv", index=False)
print("written: cow_rewards.csv")

# %% [markdown]
# ## Other rewards that exist but are off by default
# Environmental rewards (`reward_nitrogen`, `reward_enteric_methane`, `reward_environment_total`) are in `model.rewards` too; they are
# not part of `total` unless switched on. Physical quantities such as `milk_yield_kg` and `dmi_kg` are there as well:
# `sorted(model.rewards)` lists everything.

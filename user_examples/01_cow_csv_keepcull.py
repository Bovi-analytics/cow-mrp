# %% [markdown]
# # 1. Your cows from a CSV: monthly states, rewards and keep/cull
#
# **What this does.** Reads a CSV of cows, maps every cow to a state of the cow model (month resolution), looks up the reward of each
# cow's state, and tells you for every cow the sale price above which culling and replacing her beats keeping her.
#
# **Your own data.** Put your file path in `HERD_CSV` below. Columns (one row per cow):
#
# | column | meaning |
# |---|---|
# | `cow_id` | any label (optional) |
# | `lactation` | 1 to 10 (10 = tenth or later) |
# | `dim_days` | days in milk |
# | `dp_days` | days pregnant, 0 if the cow is open |
#
# If you do not give a file, the **synthetic** 1,000-cow example herd in `data/cows_1000.csv` is used. It is not real farm data.
#
# **What is the package and what is this notebook?** Everything that matters comes from the installed package: `load_herd` (mapping to states),
# `build_model` (states, probabilities, rewards, transition matrix), `keep_cull_table` and `keep_cull_summary` (the break-even values and decisions),
# `economic_overrides` and `reprice_model` (changing economic settings). The notebook only chooses inputs and prints tables.

# %% [markdown]
# ## Install
# Run this once (in a terminal: `pip install "cowmrp[plot,excel]"`). Then restart the kernel if the package was not installed before.

# %%
# %pip install -q "cowmrp[plot,excel]"

# %%
import os
os.environ.setdefault("COW_TIME_UNIT", "month")     # BEFORE importing cowmrp: month (default), week or day. One resolution per Python session.

from pathlib import Path
import pandas as pd

from cowmrp.domains.cow.economics import ECONOMIC_SETTINGS, economic_overrides, reprice_model
from cowmrp.domains.cow.farm import HerdDataError, load_herd
from cowmrp.domains.cow.keepcull import keep_cull_summary, keep_cull_table
from cowmrp.domains.cow.keepcull_excel import write_keepcull_workbook
from cowmrp.model import build_model

HERE = Path("data")                    # the data/ folder next to this notebook   # a notebook runs from its own folder
HERD_CSV = HERE / "cows_1000.csv"        # <- replace with your own file, e.g. "my_cows.csv"
OFFER = 900                               # <- the price (USD per cow) you are offered for a cull cow; keep/cull compares it with each cow's break-even value
MY_SETTINGS = {}                          # <- economic settings you want to change; empty = all defaults. Example: {"REARING_COST": 2800, "MILK_PRICE_PER_KG": 0.5}

# %% [markdown]
# ## Build the model
# The domain is `"cow"` (states = lactation 1-10, days in milk, days pregnant). `build_model("cow")` builds all 2,970 monthly states,
# their probabilities, rewards and the transition matrix.

# %%
import cowmrp
print("cowmrp", cowmrp.__version__)

model = build_model("cow")
print(f"{len(model.states):,} states, {model.P.nnz:,} transitions")

# %% [markdown]
# ## Map your cows to states
# The loader uses lactation, days in milk and days pregnant (not state ids). A cow it cannot map is reported, never silently dropped.
#
# * `dp_days` may be left out of the file: then **every cow is treated as open**. Include it if any of your cows are pregnant.
# * The model works in steps (30 days at month resolution), so the table shows the day value of the state a cow was mapped to, which can differ
#   from your number (100 days in milk becomes the step that contains day 100).
# * The example file is month-binned synthetic data. At week or day resolution 28 of its 1,000 cows have a combination that does not fit the model's rules
#   (for example conception before day 50 in milk), so they have no state: the loader stops, lists them and suggests the closest valid states.

# %%
herd = load_herd(HERD_CSV, model)
print(f"{len(herd)} cows; by status:\n{herd['status'].value_counts().to_string()}")
herd.head()

# %% [markdown]
# ## Economic settings: defaults, or your own
# Anything you do not list in `MY_SETTINGS` keeps its default. The table shows the settings you can change, their default and the recorded source.
# The defaults are model inputs, not validated farm economics: where the source column says that no source is recorded, check the value for your farm.

# %%
pd.DataFrame({k: {"default": v["default"], "unit": v.get("unit", ""), "source": v.get("source", "")} for k, v in ECONOMIC_SETTINGS.items()}).T

# %%
with economic_overrides(**MY_SETTINGS):               # an empty dict changes nothing
    scenario = reprice_model(model) if MY_SETTINGS else model
    table = keep_cull_table(herd, scenario, years=(1, 3), offer=OFFER)
    write_keepcull_workbook(herd, "keepcull_accounting.xlsx", scenario, years=(1, 3))      # the same numbers as an Excel accounting workbook (needs the excel extra)
table.head()

# %% [markdown]
# ## Keep/cull insights
# `value_3y` is the break-even sale value over a 3-year horizon: if your offer is higher, culling and replacing the cow is better.
# The model's own comparison does not credit the salvage value of the sale; here the offer is that credit, and it is your input.

# %%
print(f"Cull at an offer of {OFFER} USD (3-year horizon): {int((table['decision_3y'] == 'CULL').sum())} of {len(table)} cows")
table.nsmallest(10, "value_3y")[["cow_id", "lactation", "dim_days", "dp_days", "value_3y", "decision_3y"]].round(0)

# %%
keep_cull_summary(table, (1, 3)).round(0)

# %%
table.to_csv("keepcull_result.csv", index=False)
print("written: keepcull_result.csv")

# %% [markdown]
# ## What happens with wrong input
# The loader checks the file and says which rows are wrong. Here a lactation that does not exist (11) and a cow that is too far pregnant for her days in milk.

# %%
bad = pd.DataFrame({"cow_id": [1, 2, 3], "lactation": [3, 11, 2], "dim_days": [120, 150, 100], "dp_days": [0, 0, 280]})
try:
    load_herd(bad, model)
except HerdDataError as err:
    print(err)

# %% [markdown]
# ## Limits
# * Keep/cull compares each cow with a replacement heifer over the horizon; the salvage value of a sale is not part of the model's own comparison. The offer you type is that credit.
# * The break-even value is an expected value of the model, not a prediction for an individual cow: the model knows only her state (lactation, days in milk, days pregnant), not her health or milk history.
# * The example herd is synthetic and the defaults are not validated for your farm.

# %% [markdown]
# ## Other options (not run here)
# * **Time resolution:** set `COW_TIME_UNIT` to `week` or `day` before the first import (a command line: `set COW_TIME_UNIT=day` on Windows,
#   `export COW_TIME_UNIT=day` elsewhere). Daily resolution is supported by the package (about 2 million cow states) and you can build your own daily workflows with compatible cow data;
#   this notebook and its bundled CSV (monthly) have not been verified at daily resolution, and daily runtime and memory have not been benchmarked.
# * **Domain:** `build_model("cow")` is the cow domain. Cows must be lactation 1-10, with a days-in-milk and days-pregnant combination
#   the model allows. For young animals use `"youngstock"` (example 4).
# * **Excel workbook:** written above (`keepcull_accounting.xlsx`); it needs `pip install "cowmrp[excel]"`.

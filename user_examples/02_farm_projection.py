# %% [markdown]
# # 2. Where will your farm be in 24 months? Future distribution of the herd
#
# Starting from the cows in your CSV, the model's transition matrix is applied month after month. The result is the **expected number of cows**
# open, pregnant and dry, by lactation, and so on, plus how many cows have been replaced along the way (the replacement cohort).
#
# **What is the package and what is this notebook?** `load_herd`, `project_herd` (the herd is moved forward with the model's transition matrix) and `write_farm_outputs`
# (tables and dashboards) are package functions; the figure in the notebook is drawn from `result["evolution"]`. The numbers are **expected** cows (averages), not
# a forecast of individual animals, and the projection has no outside events such as buying cows or a change of management.
#
# **Your own data:** the same CSV as example 1 (`lactation`, `dim_days`, `dp_days`, optional `cow_id`). Without it the **synthetic** 1,000-cow
# example herd is used.

# %% [markdown]
# ## Install
# Run this once (in a terminal: `pip install "cowmrp[plot,excel]"`). Then restart the kernel if the package was not installed before.

# %%
# %pip install -q "cowmrp[plot,excel]"

# %%
import os
os.environ.setdefault("COW_TIME_UNIT", "month")      # before importing cowmrp

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from cowmrp.domains.cow.farm import load_herd, project_herd
from cowmrp.model import build_model
from cowmrp.viz.farm import write_farm_outputs

HERE = Path("data")                    # the data/ folder next to this notebook
HERD_CSV = HERE / "cows_1000.csv"       # <- your own file
STEPS = 24                               # <- months ahead (model steps; at week resolution 24 steps are 24 weeks)
OUT = Path("farm_output")                # <- where the files are written

# %%
model = build_model("cow")
herd = load_herd(HERD_CSV, model)
result = project_herd(herd, model, mode="expected", steps=STEPS, groupers=["status", "lactation_exact"])
result["distribution_summary"].query("grouper == 'status'").round(1)

# %% [markdown]
# ## A figure in the notebook
# `result["evolution"]["status"]` is a list with one entry per month (step 0 = today); each entry gives the expected cows per status. The
# original herd and the replacement cohort are kept apart in `result["regular"]` and `result["replaced"]` (lists of the same kind).

# %%
evo = pd.DataFrame(result["evolution"]["status"]).fillna(0.0)      # one row per step, one column per status
fig = go.Figure()
for column in evo.columns:
    fig.add_trace(go.Scatter(x=evo.index, y=evo[column], stackgroup="one", name=str(column)))
fig.update_layout(title="Expected cows by status", xaxis_title="model step (month)", yaxis_title="cows")
fig

# %% [markdown]
# ## The interactive dashboards
# `write_farm_outputs` writes CSV tables and the two dashboards. `dashboard_plotly.html` is the interactive one: open it in a browser
# (double-click it). It works offline.

# %%
files = write_farm_outputs(result, OUT)
print("\n".join(sorted(Path(f).name for f in files)))
print("open this file in a browser:", (OUT / "dashboard_plotly.html").resolve())

# %% [markdown]
# ## Other options (not run here)
# * `mode="stochastic"` follows individual cows by chance (slower); `mode="monte_carlo", sims=100` repeats that and gives bands.
# * Resolution: set `COW_TIME_UNIT=week` or `day` before the first import. `STEPS` is then in weeks or days.
# * Domain: this layer is for `"cow"` only; the herd must be lactations 1-10 with allowed days in milk and days pregnant.

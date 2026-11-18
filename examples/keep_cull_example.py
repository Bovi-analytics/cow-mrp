"""Keep or cull -- the break-even sale value of every cow in the example herd.

    python examples/keep_cull_example.py

keep = expected future value of the cow, her future replacement costs included; replace = value of a heifer bought now, after one replacement
cost; value = keep - replace = the sale value above which culling and replacing her beats keeping her. No sale price is built into the model:
the offer below is an input you choose. The salvage value of a sale is not credited in the model's own comparison (documented limitation).
The last part repeats the table under a different rearing cost to show how a default is overridden for one block of code and then restored.
"""
from cowmrp.data import example_herd_path
from cowmrp.domains.cow.economics import ECONOMIC_SETTINGS, economic_overrides, reprice_model
from cowmrp.domains.cow.farm import load_herd
from cowmrp.domains.cow.keepcull import keep_cull_summary, keep_cull_table
from cowmrp.model import build_model

model = build_model("cow")
herd = load_herd(example_herd_path(), model)
table = keep_cull_table(herd, model, years=(1, 3, 5), offer=900)       # offer = USD per cow

print("Weakest five cows at the 3-year horizon (value_3y = break-even sale value, USD):")
print(table.nsmallest(5, "value_3y")[["cow_id", "lactation", "dim_days", "value_3y", "decision_3y", "stable"]].round(0).to_string(index=False))
print("\nSummary at an offer of 900 USD:")
summary = keep_cull_summary(table, (1, 3, 5))
print(summary[summary["lactation"] == "all"].round(0).to_string(index=False))

# Scenario: the defaults are untouched outside the block. Anything not named keeps its default. 2,800 USD is an illustrative value, not a recommendation.
default_cost = ECONOMIC_SETTINGS["REARING_COST"]["default"]
with economic_overrides(REARING_COST=2800):
    scenario = reprice_model(model)                                   # same states and transition matrix; rewards recomputed
    table_s = keep_cull_table(herd, scenario, years=(1, 3, 5), offer=900)    # the keep/cull layer reads REARING_COST inside the block
n_default, n_scenario = int((table["decision_3y"] == "CULL").sum()), int((table_s["decision_3y"] == "CULL").sum())
print(f"\nCows to cull at 900 USD, 3-year horizon: {n_default} with the default rearing cost ({default_cost:,.0f} USD), {n_scenario} with 2,800 USD")
print(f"Median break-even value (3-year): {table['value_3y'].median():,.0f} -> {table_s['value_3y'].median():,.0f} USD")

"""Keep or cull -- the break-even sale value of every cow in the example herd.

    python examples/keep_cull_example.py

keep = expected future value of the cow, her future replacement costs included; replace = value of a heifer bought now, after one replacement
cost; value = keep - replace = the sale value above which culling and replacing her beats keeping her. No sale price is built into the model:
the offer below is an input you choose. The salvage value of a sale is not credited in the model's own comparison (documented limitation).
"""
from cowmrp.data import example_herd_path
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

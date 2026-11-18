"""Whole-farm projection with the example herd -- run this file, or copy it and point HERD at your own records.

    python examples/whole_farm_example.py

What it does (everything comes from the installed package; no file of this repository is needed):
  1. builds the cow model (month resolution; set COW_TIME_UNIT=week or day BEFORE starting Python for another resolution),
  2. loads the 200-cow synthetic example herd (cowmrp.data) -- also try example_raw_herd_path(): the same cows as a farmer would export
     them (cow_id, lactation, dim_days, dp_days only); the loader maps every cow to a model state and reports any record it cannot map,
  3. projects the herd 24 months ahead (expected values; also "stochastic" and "monte_carlo"),
  4. ranks the cows by keep/cull break-even value,
  5. writes CSV tables and the two dashboards into ./farm_example_output/.

YOUR OWN HERD: a CSV with one row per cow and the columns  lactation (1-10), dim_days (days in milk), dp_days (days pregnant, 0 if open)
and optionally cow_id. Replace HERD below by its path.
"""
from cowmrp.data import example_herd_path
from cowmrp.domains.cow.farm import load_herd, project_herd
from cowmrp.domains.cow.keepcull import keep_cull_table
from cowmrp.model import build_model
from cowmrp.viz.farm import write_farm_outputs

HERD = example_herd_path()          # or example_raw_herd_path(), or "my_herd.csv"
OUT = "farm_example_output"

model = build_model("cow")
herd = load_herd(HERD, model)
print(f"{len(herd)} cows loaded; status counts:\n{herd['status'].value_counts().to_string()}")

result = project_herd(herd, model, mode="expected", steps=24)
print("\nHerd by status now and after 24 months:")
print(result["distribution_summary"].query("grouper == 'status'").round(1).to_string(index=False))

table = keep_cull_table(herd, model, years=(1, 3), offer=900)           # CULL when the offer (USD) exceeds the break-even value
print(f"\nKeep/cull at an offer of 900 USD (3-year horizon): {int((table['decision_3y'] == 'CULL').sum())} of {len(table)} cows to cull")
print(table.nsmallest(5, "value_3y")[["cow_id", "value_3y", "decision_3y"]].round(0).to_string(index=False))

files = write_farm_outputs(result, OUT)
table.to_csv(f"{OUT}/keepcull.csv", index=False)
print(f"\nwrote {len(files) + 1} files to {OUT}/ (open dashboard_plotly.html)")

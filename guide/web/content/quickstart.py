from cowmrp.data import example_herd_path            # 200 synthetic cows shipped with the package
from cowmrp.model import build_model
from cowmrp.domains.cow.farm import load_herd, project_herd

model = build_model("cow")                            # month resolution
herd = load_herd(example_herd_path(), model)          # or your own CSV / DataFrame
result = project_herd(herd, model, mode="expected", steps=24)

print(f"{len(model.states):,} states, {len(herd)} cows loaded")
print(result["distribution_summary"].query("grouper == 'status'").round(1).to_string(index=False))

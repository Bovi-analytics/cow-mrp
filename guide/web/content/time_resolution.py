# Run once per resolution:  COW_TIME_UNIT=month|week|day python time_resolution.py
from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS, dim_step_of_days, dp_step_of_days
from cowmrp.model import build_model

model = build_model("cow")                                  # states, probabilities and rewards at the chosen resolution
s = model.states
print(f"{TIME_UNIT}: step = {UNIT_DAYS} d, {len(s):,} states, {model.P.nnz:,} transitions")

# the same cow at every resolution: second lactation, 150 days in milk, 60 days pregnant
dim, dp = dim_step_of_days(150), dp_step_of_days(60)
i = s[(s["lactation"] == 2) & (s["dim"] == dim) & (s["dp"] == dp)]["state_id"].iloc[0]
print(f"cow (lactation 2, 150 d in milk, 60 d pregnant) is state (2, dim={dim}, dp={dp})")
for name in ("milk_income", "feed_cost", "total"):
    r = model.rewards[name][i]
    print(f"  {name:12s} {r:8.2f} USD per step  =  {r / UNIT_DAYS:6.2f} USD per day")

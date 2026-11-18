"""Creates the two SYNTHETIC example data files in data/ (run once; the files are already included).

    cows_1000.csv         1,000 cows: cow_id, lactation (1-10), dim_days (days in milk), dp_days (days pregnant, 0 = open).
                          Sampled with a fixed seed from the 10,000-cow synthetic herd that an earlier version of CowMRP generated from its own
                          model. Only the three farm-style columns are kept: the old state ids are NOT used, the loader maps every cow to a
                          model state from lactation, days in milk and days pregnant.
    youngstock_500.csv    500 young animals, random but fixed seed: animal_id, stage (CALF, HEIFER, BREEDING, PREGNANT), age_months, dp_months.

Neither file contains real animals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OLD_FARM = Path(sys.argv[1]) if len(sys.argv) > 1 else None        # path of the old farm_data.csv (only needed to recreate cows_1000.csv)

if OLD_FARM is not None:
    old = pd.read_csv(OLD_FARM)
    cows = old.sample(1000, random_state=7)[["lactation", "dim_days", "dp_days"]].reset_index(drop=True)
    cows.insert(0, "cow_id", np.arange(1, 1001))
    cows.to_csv(HERE / "data" / "cows_1000.csv", index=False)

rng = np.random.default_rng(11)
stage = rng.choice(["CALF", "HEIFER", "BREEDING", "PREGNANT"], size=500, p=[0.22, 0.48, 0.10, 0.20])
age = np.zeros(500, dtype=int)
dp = np.zeros(500, dtype=int)
for i, s in enumerate(stage):
    if s == "CALF":
        age[i] = rng.integers(0, 3)               # months 0-2
    elif s == "HEIFER":
        age[i] = rng.integers(3, 16)              # months 3-15
    elif s == "PREGNANT":
        dp[i] = rng.integers(1, 11)               # months pregnant 1-10
pd.DataFrame({"animal_id": np.arange(1, 501), "stage": stage, "age_months": age, "dp_months": dp}).to_csv(HERE / "data" / "youngstock_500.csv", index=False)

TIME_UNIT = "month"
UNIT_DAYS = 30
P_WEAR = 0.2          # new -> worn
P_BREAK = 0.5         # worn -> broken
# broken -> new (replacement) with probability 1; "new" stays new with 1 - P_WEAR, "worn" stays worn with 1 - P_BREAK
REWARD = {"new": 10.0, "worn": 4.0, "broken": -20.0}

"""REQUIRED: TIME_UNIT and UNIT_DAYS. Everything else here is this domain's own input.

ALL NUMBERS BELOW ARE ILLUSTRATIVE, chosen to make the example readable. They are not biological or economic estimates and are not
taken from any source. Replace them with your own.
"""
TIME_UNIT = "month"      # 'day' | 'week' | 'month'  (REQUIRED)
UNIT_DAYS = 30           # days per model step       (REQUIRED, a positive integer)

# Probabilities (per month)
CONCEPTION_BY_MONTH = [0.35, 0.33, 0.30, 0.27, 0.24, 0.20]      # open month k = 0..5: chance a bred cow becomes pregnant
LAST_OPEN_MONTH = len(CONCEPTION_BY_MONTH) - 1                   # a cow still open after this month is culled and replaced
GESTATION_MONTHS = 9                                             # pregnant months 1..9; she calves at the end of month 9
PREGNANCY_LOSS_PER_MONTH = 0.01                                  # chance of losing the pregnancy in pregnant months 1..8

# Money (USD)
BREEDING_COST_PER_OPEN_MONTH = 40.0
CALF_VALUE = 500.0
CULL_AND_REPLACE_COST = 900.0
PREGNANCY_LOSS_COST = 300.0

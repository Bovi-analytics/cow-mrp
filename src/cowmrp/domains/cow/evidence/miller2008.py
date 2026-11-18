"""
cow/evidence/miller2008.py
==========================
Miller, Kuhn, Norman & Wright (2008) "Death losses for lactating cows in herds enrolled in Dairy Herd
Improvement test plans", J. Dairy Sci. 91:3710-3715. 15,025,035 lactations, 45,032 herds, US 1995-2005.

Table 2 (class A, printed): observed death frequency per LACTATION by parity.
Only deaths after the first test day and before dry-off are recorded, so these are LOWER bounds on all
deaths; the paper itself says so. Used here ONLY to split Pinedo's ">=4" death hazard across lactations
4..10 (relative shape), never as an absolute level.
"""

DEATH_PER_LACTATION_PCT = {1: 2.05, 2: 2.66, 3: 3.72, 4: 4.38, 5: 4.83, 6: 5.78, 7: 5.92, 8: 6.40}   # 8 means ">=8"
DEATH_SHARE_BY_DIM_STAGE_PCT = {"<=45": 22.7, "46-90": 21.9, "91-150": 17.1, "151-250": 18.0, ">=251": 20.3}   # class A (text p.3713)


def death_relative_to_pinedo_parity4plus():
    """{lactation 4..10: factor} = Miller(L) / mean Miller over 4..8+; 9 and 10 repeat the '>=8' value.
    Class B: a ratio of printed values, used to spread Pinedo's single '>=4' curve over lactations 4-10."""
    vals = [DEATH_PER_LACTATION_PCT[min(L, 8)] for L in range(4, 11)]
    mean = sum(vals) / len(vals)
    return {L: DEATH_PER_LACTATION_PCT[min(L, 8)] / mean for L in range(4, 11)}

"""
cow/evidence/pinedo2010.py
==========================
Pinedo, De Vries & Webb (2010) "Dynamics of culling risk with disposal codes reported by Dairy Herd
Improvement dairy herds", J. Dairy Sci. 93:2250-2261 (docs/Papers for specific task/Mortality/
1-s2.0-S002203021000216X-main.pdf). 3,629,002 Holstein lactations, 2,054 DHI herds, US east of the
Mississippi, calving 2001-2006.

WHAT IS HERE
------------
Figures 4-5 (OPEN cows, risk per day x 100,000 against days after calving) and Figures 6-7 (PREGNANT
cows, same risk against days of PREGNANCY), for parity 1, 2, 3 and >=4, for the disposal codes
  died | feet and legs | injury/other | mastitis | disease | udder.
The codes "low production" and "reproduction" are NOT used: CowMRP represents those reasons with the
explicit milk-floor policy (cow/tables.py cull_p, MILK_FLOOR_KG), and "reason not reported" (5.0 % of
removals) has no hazard curve.

EVIDENCE CLASS: B (adapted). The curves are printed only as graphs, so each value below was READ BY EYE
from the figure at the centre of a 30-day bin (about +/-10-15 % of the read value; the line markers of
parity 1-4 overlap at some ages). The shape, the parity ordering and the open-vs-pregnant contrast are
the paper's; the numbers are a transcription, not a re-analysis. Replace with the authors' data if they
can be obtained. Pinedo et al. state their own interpretation limits (misclassified pregnancy status,
"pregnant" cows are censored at conception so risk near DP 0 is ~0 by construction).

LAYOUT
------
OPEN[code][parity]   -> list over DIM months 0..16 (days 0-30, 30-60, ...) in units of 1e-5 per day
PREGNANT[code][parity] -> list over gestation months 0..9 (days pregnant 0-30, ...) in 1e-5 per day
parity index: 1, 2, 3, 4 (4 means ">=4"). Beyond the last printed month the value is REPEATED.
"""

UNIT = 1e-5   # risk per cow-day = value x UNIT

OPEN = {
    "died": {
        1: [38, 15, 8, 6.5, 6, 6, 6, 6.5, 7, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12],
        2: [25, 22, 15, 12, 11, 10.5, 10, 12.5, 14, 16, 18, 20, 22, 24, 26, 27, 28],
        3: [45, 23, 17, 15, 14.5, 14, 14, 18, 22, 24, 27, 30, 32, 34, 35, 36, 36],
        4: [52, 28, 21, 19, 18, 17.5, 17, 24, 31, 34, 37, 41, 44, 47, 50, 53, 55],
    },
    "feet_legs": {
        1: [4.5, 5, 3.8, 3.4, 3.2, 3, 2.8, 2.5, 2.4, 2.5, 2.7, 2.9, 3, 3.2, 3.5, 3.7, 3.8],
        2: [4, 5, 4.5, 4.5, 4.7, 5, 5.5, 5.5, 5.5, 5.7, 6, 6, 6, 6.2, 6.5, 6.8, 7],
        3: [9, 10.5, 8.5, 8.5, 8.5, 9, 9.5, 10, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5],
        4: [16, 17, 15, 14.5, 15, 15.5, 16.5, 17, 17, 16.5, 16, 15.5, 15, 15, 15, 14.5, 14],
    },
    "injury_other": {
        1: [20, 14, 9, 7.5, 6.5, 6, 5.5, 5, 5, 5, 5.2, 5.5, 6, 6.5, 7.5, 8.5, 9.5],
        2: [12, 14, 10.5, 9.5, 9.3, 9, 9, 9.5, 10, 10.2, 11, 11.8, 12.5, 13.5, 14, 14.5, 15],
        3: [21, 18, 13, 12, 12, 11.5, 12, 13.5, 14.5, 15, 15.5, 16.5, 17, 18, 18.5, 19, 18],
        4: [31, 28, 19, 17.5, 17.5, 17, 17, 21, 22, 21, 22, 23, 23.5, 24, 25, 25, 25],
    },
    "mastitis": {
        1: [9, 6.5, 5.5, 5.2, 5, 5, 4.8, 4.5, 4, 3.8, 3.6, 3.5, 3.5, 3.6, 3.8, 4, 4],
        2: [6.5, 9.5, 10.5, 11.5, 12, 12, 11, 10.5, 9, 8, 7.5, 7.3, 7.2, 7.2, 7, 7, 6.5],
        3: [11, 15, 16, 17, 18, 17.5, 16, 15, 14, 12.5, 11.5, 10.5, 10, 9.5, 9, 9, 8.5],
        4: [17, 22, 23, 23.5, 24, 23, 21.5, 20.5, 19.5, 18, 16.5, 15, 14.5, 14, 13.5, 13, 12.5],
    },
    "disease": {
        1: [6.5, 6, 3, 2.6, 2.5, 2.4, 2.4, 2.4, 2.5, 2.6, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4],
        2: [7, 9, 6, 5.3, 5, 4.7, 4.5, 4.8, 5.2, 5.3, 5.3, 5.3, 5.2, 5.2, 5.2, 5.2, 5],
        3: [10.5, 13.5, 8, 6, 5.6, 5.5, 5.5, 6, 6, 5.9, 5.9, 5.9, 5.8, 5.6, 5.4, 5.2, 5],
        4: [11, 14, 8, 6, 5.6, 5.6, 6, 6.5, 7, 6.5, 6.5, 6.5, 6.2, 6.3, 6.5, 6.5, 6.5],
    },
    "udder": {
        1: [5, 3, 1.6, 1.1, 0.9, 0.8, 0.8, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.5, 1.7, 1.8, 1.8],
        2: [2.5, 1.8, 1.3, 1.3, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.2],
        3: [4.5, 3.5, 2.5, 2.2, 2.2, 2.3, 2.5, 2.8, 3.2, 3.5, 3.8, 4, 4.5, 5.5, 6, 5, 4],
        4: [7, 5.5, 4.2, 3.8, 3.5, 3.7, 4, 5, 6, 6.5, 6.8, 7, 7.2, 7.5, 7.7, 7.5, 7.2],
    },
}

PREGNANT = {
    "died": {
        1: [0.5, 1.8, 3.5, 4.3, 4.6, 5, 6, 8, 11, 16],
        2: [0.8, 3, 5, 6, 6.3, 7, 8.5, 11, 16, 20],
        3: [1, 3.8, 7, 8.5, 9, 10, 12, 15, 21, 27],
        4: [1, 4, 8, 9.5, 10, 11, 13, 17, 25, 32],
    },
    "feet_legs": {
        1: [0.1, 0.6, 1.1, 1.4, 1.4, 1.4, 1.4, 1.2, 1.1, 1.1],
        2: [0.3, 1.2, 2.2, 2.6, 3.0, 2.5, 2.3, 2.0, 1.9, 1.9],
        3: [0.4, 2.2, 3.8, 4.6, 4.5, 4.4, 4.2, 3.5, 2.8, 2.8],
        4: [0.5, 3.5, 6.2, 7.4, 7.6, 6.5, 5.5, 4.2, 4.0, 4.0],
    },
    "injury_other": {
        1: [0.2, 1.2, 2.5, 3, 3, 3, 2.9, 3, 3.2, 3.3],
        2: [0.4, 2, 3.6, 4.3, 4.5, 4.3, 4, 3.8, 3.7, 3.7],
        3: [0.6, 2.8, 5, 6, 5.7, 5.5, 5.5, 5.5, 5.6, 6],
        4: [0.8, 3.5, 6.5, 7.7, 7.3, 7.4, 6.6, 6.2, 6.5, 7],
    },
    "mastitis": {
        1: [0.1, 1.2, 2.5, 2.9, 2.4, 2.2, 1.6, 1.2, 0.9, 0.9],
        2: [0.3, 2.5, 5.5, 6.7, 6, 4.8, 3.5, 2.0, 1.5, 1.5],
        3: [0.4, 3.5, 8, 10.5, 9.5, 7.5, 5, 3.2, 2.5, 2.8],
        4: [0.5, 4.5, 9.5, 11, 10, 8, 5.5, 3.5, 2.8, 3],
    },
    "disease": {
        1: [0.1, 0.5, 1.2, 1.6, 1.7, 1.8, 1.9, 2, 2, 2],
        2: [0.2, 1.2, 2.5, 2.8, 2.8, 2.8, 2.7, 2.5, 2.5, 2.5],
        3: [0.2, 1.5, 3.2, 3.5, 3.4, 3.2, 3, 2.7, 2.6, 2.6],
        4: [0.2, 1.6, 3.3, 3.5, 3.4, 3.3, 3.1, 2.8, 2.7, 2.7],
    },
    "udder": {
        1: [0.05, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        2: [0.1, 0.4, 0.6, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6],
        3: [0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.4, 1.3, 1.1, 1.1],
        4: [0.1, 0.6, 1.2, 1.5, 1.7, 1.9, 2.0, 1.9, 1.5, 1.5],
    },
}

OTHER_CULL_CODES = ("feet_legs", "injury_other", "mastitis", "disease", "udder")

# Printed values (class A) used only as validation references, never as inputs:
TABLE1_ANNUAL = {   # Pinedo 2010 Table 1, annualized rates %/yr
    "live_cull_by_parity": {1: 16.9, 2: 23.3, 3: 30.1, 4: 40.8},
    "death_by_parity": {1: 3.9, 2: 5.6, 3: 8.5, 4: 11.7},
    "pregnant": {"live_cull": 7.2, "death": 3.1}, "open": {"live_cull": 40.4, "death": 11.1},
    "all": {"live_cull": 25.1, "death": 6.6},
}

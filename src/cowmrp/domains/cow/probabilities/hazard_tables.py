"""
cow/probabilities/hazard_tables.py
===================================
Shared builder for the two-clock hazard tables (mortality, other-reason culling).

Two clocks, one table layout (lactation 1..MAX_LACTATION x month x status Open/Pregnant/Dry):
  Open      -> `month` = month since CALVING (the DIM clock)
  Pregnant, Dry -> `month` = month of PREGNANCY (the DP clock; Dry is simply the late part of gestation)
This is the evidence's own structure (Pinedo et al. 2010 Figures 4-7: open cows against days after calving,
pregnant cows against days pregnant) and what the state (lactation, dim, dp) carries: an open cow's risk
cannot depend on DP and a pregnant cow's risk is read on the DP clock, with lactation as the third axis.

Per-step conversion is the one rule of cow/lookup_tables.py (entry-offset survival ratio); pregnant rows use
the gestation convention of abortion.py (step dp=1 starts right after conception).
"""

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import MAX_LACTATION, MAX_DIM, MAX_DP, DRY_OFF_DP, UNIT_DAYS
from cowmrp.domains.cow.lookup_tables import (
    MONTH, N_DIM_MONTHS, STATUSES, full_grid, survival_from_monthly, step_probability_matrix, age_offset_slots,
    table_to_array, status_index,
)
from cowmrp.domains.cow.evidence.pinedo2010 import UNIT, OPEN, PREGNANT

N_PRINTED_OPEN_MONTHS = 17      # Pinedo Figs 4-5 run to ~500 days after calving
N_PRINTED_PREG_MONTHS = 9       # Pinedo Figs 6-7 run to ~260 days of pregnancy (months 0..8)


def _daily_to_monthly(h_day):
    return 1.0 - (1.0 - np.clip(h_day, 0.0, 1.0)) ** MONTH


def pinedo_status_table(codes, lactation_factor=None, factor_note=""):
    """Full table (status x lactation x month) of MONTHLY probabilities = 1-(1-h)^30 with h the summed
    per-day risk of `codes` from Pinedo 2010 Figs 4-7 for parity group min(lactation, 4).
    lactation_factor: optional {lactation: multiplier} (used to spread the '>=4' curve over lactations 4-10)."""
    g = full_grid(N_DIM_MONTHS, STATUSES)
    lac = g["lactation"].values
    month = g["month"].values
    is_open = g["status"].values == "Open"
    parity = np.minimum(lac, 4)
    p = np.zeros(len(g))
    basis = np.empty(len(g), dtype=object)
    for i in range(len(g)):
        par = int(parity[i])
        m = int(month[i])
        if is_open[i]:
            idx = min(m, N_PRINTED_OPEN_MONTHS - 1)
            h = sum(OPEN[c][par][idx] for c in codes) * UNIT
            printed = m < N_PRINTED_OPEN_MONTHS
        else:
            idx = min(m, N_PRINTED_PREG_MONTHS - 1)
            h = sum(PREGNANT[c][par][idx] for c in codes) * UNIT
            printed = m < N_PRINTED_PREG_MONTHS
        if lactation_factor is not None and lac[i] in lactation_factor:
            h *= lactation_factor[lac[i]]
        p[i] = _daily_to_monthly(h)
        # evidence label: parity 1-3 and the ">=4" group (lactation 4) are read from the figure; lactations 5-10 of
        # a grouped source are the same curve repeated unless a lactation factor (another source) separates them
        # mortality with Miller ratios: ratios are printed for parity 1..>=8, so lactations 9 and 10 repeat the '>=8' value
        grouped_repeat = lac[i] > (4 if lactation_factor is None else 8)
        # Dry cows: Pinedo's pregnant-cow curves describe cows still in a lactation record (milked), so they are EXTRAPOLATED to the
        # dry period, not measured there -> "repeated", never "derived".
        is_dry = (not is_open[i]) and g["status"].values[i] == "Dry"
        basis[i] = "repeated" if (not printed or grouped_repeat or is_dry) else "derived"
    g["p"] = p
    g["basis"] = basis
    g["note"] = np.where(
        is_open,
        "Pinedo 2010 Figs 4-5 (open cows, risk per day read from the graph at 30-day bins, parity 1/2/3/>=4); month = months since calving"
        + factor_note + "; beyond ~500 d repeated",
        "Pinedo 2010 Figs 6-7 (pregnant cows, risk per day vs days PREGNANT, parity 1/2/3/>=4); month = months of PREGNANCY"
        + factor_note + "; Dry = late gestation on the same DP clock; beyond ~260 d of pregnancy repeated")
    g["note"] = np.where(g["status"] == "Dry", "EXTRAPOLATED to dry cows: " + g["note"] + " (the source curve is for milked cows still in a lactation record; no hazard was measured for dry cows)", g["note"])
    return g


def gestation_step_probability_matrix(P, k_max):
    """(n_lactations, k_max+1) per-pregnant-step probabilities from monthly probabilities indexed by month of
    pregnancy. Same convention as abortion.py: step 0 unused (0), step 1 carries the risk from conception to the
    first boundary (the chain enters dp=1 with every pregnancy alive), later steps use entry-averaged survival."""
    out = step_probability_matrix(P, k_max, first_age_step=1)
    out[:, 0] = 0.0
    slot = age_offset_slots()
    for i in range(P.shape[0]):
        S = survival_from_monthly(P[i])
        out[i, 1] = 1 - np.mean(S(1 * UNIT_DAYS + slot))
    return out


def status_step_arrays(df):
    """(open[lac, dim], pregnant[lac, dp], dry[lac, dp]) per-step probability arrays from a status table."""
    P = table_to_array(df, N_DIM_MONTHS, STATUSES)
    return (step_probability_matrix(P[0], MAX_DIM, first_age_step=0),
            gestation_step_probability_matrix(P[1], MAX_DP),
            gestation_step_probability_matrix(P[2], MAX_DP))


def lookup_status_arrays(arrays, states):
    """Per-state value from status_step_arrays output: Open -> DIM clock, Pregnant/Dry -> DP clock."""
    op, pr, dr = arrays
    lac = np.clip(states["lactation"].values, 1, MAX_LACTATION) - 1
    dim = np.clip(states["dim"].values, 0, MAX_DIM)
    dp = np.clip(states["dp"].values, 0, MAX_DP)
    s = status_index(states["dp"].values, DRY_OFF_DP)
    return np.where(s == 0, op[lac, dim], np.where(s == 1, pr[lac, dp], dr[lac, dp]))

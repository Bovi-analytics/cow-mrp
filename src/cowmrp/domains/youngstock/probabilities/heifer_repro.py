"""
youngstock/probabilities/heifer_repro.py
=========================================
Heifer breeding -> pregnancy -> first calving probabilities (DAIRY_F line,
stages BREEDING and PREGNANT). All per MODEL STEP, day-exact, resolution-
aware. Rows out of scope return 0.

WHAT EACH PROBABILITY REPRESENTS (and why there is no double count)
--------------------------------------------------------------------
p_conceive      per step, probability that at least one service in the step ends
                in a CONFIRMED PREGNANCY. The literature rate (Norman 2010: 56%
                conventional / 39% sexed per service) is a confirmed-pregnancy
                rate: CDCB defines heifer conception rate as the proportion of
                inseminations that result in pregnancy, established from pregnancy
                diagnoses and subsequent calving confirmations. Embryonic loss
                BEFORE the diagnosis is therefore already a "failed service" inside
                this number.
p_exit_*        all-cause removal (cull/sale/death) from the Fodor 2020 counts,
                per stage. Used as INPUT for removal; nothing else in these stages
                represents removal.
p_abort_heifer  pregnancy loss AFTER a pregnancy was counted as conceived. Because
                the conception rate is already net of early loss, applying the Albaaj
                (2022) 19-90 d windows on top would charge the same biological loss
                twice (measured: removal 14.7% vs the observed 10.5%, age at first
                calving 27.1 vs 25.6 months). DEFAULT ('table' method): no additional
                loss -- heifer pregnancy loss after the diagnosis day is not
                separately quantified by any source read. 'formula' method
                (sensitivity only): the published Albaaj windows applied only from
                HEIFER_PREGNANCY_DIAGNOSIS_DAY (an assumption) to day 90, evaluated
                with the same conception-time-averaged gestation age as the corrected
                cow function; the uncited legacy rates outside the paper's range are
                NOT applied to heifers.
p_calve_heifer  the cow module's own conception-time-averaged calving probability (one convention).

The Fodor cohort (10.53% culled from first insemination to calving; age at first
calving 25.61 months) is an EXTERNAL BENCHMARK for the emergent totals, not a
calibration target: no parameter here was fitted to it.

Order within a step matches the cow chain (cow/transitions.py): removal first,
then pregnancy loss, then calving.
"""

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import UNIT_DAYS, GESTATION_LENGTH_DAYS
import cowmrp.domains.youngstock.parameters as YP
from cowmrp.mrp.engine import validate_lookup_coverage


def _rows(states, stage):
    return (states['line'].values == 'DAIRY_F') & (states['stage'].values == stage)


# ------------------------------------------------------------
# Conception (confirmed pregnancy)
# ------------------------------------------------------------
def heifer_cr_table():
    return pd.DataFrame([{'semen': s, 'p': v} for s, v in YP.HEIFER_CONCEPTION_PER_SERVICE.items()])


def heifer_program_cr():
    """Program-weighted conception risk per service for HEIFERS (parity-'1' semen mix of the active program)."""
    mix = YP.BREEDING_PROGRAMS[YP.BREEDING_PROGRAM]["1"]
    table = heifer_cr_table()
    validate_lookup_coverage(table, ['semen', 'p'], 'heifer_conception')
    cr = dict(zip(table['semen'], table['p']))
    return sum(share * cr[semen] for semen, share in mix.items())


def _conception_per_step(cr_per_service):
    """Per-step conception probability from a per-service risk: services occur every
    HEIFER_SERVICE_INTERVAL_DAYS, so a step of UNIT_DAYS spans UNIT_DAYS/interval services."""
    return 1 - (1 - cr_per_service) ** (UNIT_DAYS / YP.HEIFER_SERVICE_INTERVAL_DAYS)


def p_conception_table(states, context):
    return np.where(_rows(states, 'BREEDING'), _conception_per_step(heifer_program_cr()), 0.0)


def p_conception_formula(states, context):
    """Same construction with the plain conventional-semen CR (ignores the program mix)."""
    return np.where(_rows(states, 'BREEDING'), _conception_per_step(YP.HEIFER_CONCEPTION_PER_SERVICE['conventional']), 0.0)


conception_spec = {
    'name': 'heifer_conception', 'produces': 'p_conceive',
    'depends_on': ['line', 'stage'],
    'method': 'table',
    'methods': {'table': p_conception_table, 'formula': p_conception_formula, 'model': None},
}


# ------------------------------------------------------------
# Removal (all-cause) -- Fodor et al. (2020) cohort
# ------------------------------------------------------------
def _breeding_exit_per_step():
    """Per-step removal probability e so that, competing against conception c per step, the
    cumulative removal-before-conception equals HEIFER_BREEDING_STAGE_REMOVAL (=q). Step order:
    removal, then conception:   q = e / (1 - (1-e)(1-c))   =>   e = q*c / (1 - q*(1-c))."""
    q = YP.HEIFER_BREEDING_STAGE_REMOVAL
    c = _conception_per_step(heifer_program_cr())
    return q * c / (1 - q * (1 - c))


def p_breeding_exit_table(states, context):
    return np.where(_rows(states, 'BREEDING'), _breeding_exit_per_step(), 0.0)


breeding_exit_spec = {
    'name': 'breeding_exit', 'produces': 'p_exit_breeding',
    'depends_on': ['line', 'stage'],
    'method': 'table',
    'methods': {'table': p_breeding_exit_table, 'formula': None, 'model': None},
}


def _pregnant_exit_per_step():
    """Constant hazard over the gestation, so the cumulative removal across GESTATION_LENGTH_DAYS
    equals HEIFER_PREGNANT_STAGE_REMOVAL."""
    daily = 1 - (1 - YP.HEIFER_PREGNANT_STAGE_REMOVAL) ** (1.0 / GESTATION_LENGTH_DAYS)
    return 1 - (1 - daily) ** UNIT_DAYS


def p_pregnant_exit_table(states, context):
    return np.where(_rows(states, 'PREGNANT'), _pregnant_exit_per_step(), 0.0)


pregnant_exit_spec = {
    'name': 'pregnant_exit', 'produces': 'p_exit_pregnant',
    'depends_on': ['line', 'stage'],
    'method': 'table',
    'methods': {'table': p_pregnant_exit_table, 'formula': None, 'model': None},
}


# ------------------------------------------------------------
# Pregnancy loss after the pregnancy is counted as conceived (B)
# ------------------------------------------------------------
def p_heifer_pregnancy_loss_net(states, context):
    """DEFAULT. The conception rate is already net of early loss (confirmed pregnancy), so no
    additional heifer pregnancy loss is applied. See the module docstring."""
    return np.zeros(len(states))


def p_heifer_pregnancy_loss_post_diagnosis(states, context):
    """SENSITIVITY. Published Albaaj (2022) windows, but only for gestation days
    HEIFER_PREGNANCY_DIAGNOSIS_DAY..90, using the cow module's unchanged per-day hazard, evaluated at the
    conception-time-averaged gestation age (same convention as cow.probabilities.abortion)."""
    from cowmrp.domains.cow.probabilities.abortion import _albaaj_daily_hazard
    dp = states['dp'].values
    max_step = int(dp.max()) if len(dp) else 0
    day_max = (max_step + 2) * UNIT_DAYS
    days = np.arange(day_max + 1)
    hazard = _albaaj_daily_hazard(days)
    hazard = np.where((days >= YP.HEIFER_PREGNANCY_DIAGNOSIS_DAY) & (days <= 90), hazard, 0.0)
    S = np.concatenate(([1.0], np.cumprod(1 - hazard)))
    grid = np.arange(len(S))
    from cowmrp.domains.cow.lookup_tables import age_offset_slots
    slot_mid = age_offset_slots()
    T = np.ones(max_step + 2)
    for j in range(1, max_step + 2):
        T[j] = np.mean(np.interp((j - 1) * UNIT_DAYS + slot_mid, grid, S))
    step_p = np.zeros(max_step + 1)
    step_p[1:] = 1 - T[2:max_step + 2] / T[1:max_step + 1]
    return np.where(_rows(states, 'PREGNANT'), step_p[np.clip(dp, 0, max_step)], 0.0)


heifer_abortion_spec = {
    'name': 'heifer_abortion', 'produces': 'p_abort_heifer',
    'depends_on': ['line', 'stage', 'dp'],
    'method': 'table',      # <-- 'formula' = post-diagnosis Albaaj windows (sensitivity)
    'methods': {'table': p_heifer_pregnancy_loss_net, 'formula': p_heifer_pregnancy_loss_post_diagnosis, 'model': None},
}


def p_heifer_calving(states, context):
    """The cow module's own conception-time-averaged calving probability (one gestation convention)."""
    from cowmrp.domains.cow.probabilities.calving import calving_step_probability
    return np.where(_rows(states, 'PREGNANT'), calving_step_probability(states['dp'].values), 0.0)


heifer_calving_spec = {
    'name': 'heifer_calving', 'produces': 'p_calve_heifer',
    'depends_on': ['line', 'stage', 'dp'],
    'method': 'table',
    'methods': {'table': p_heifer_calving, 'formula': None, 'model': None},
}

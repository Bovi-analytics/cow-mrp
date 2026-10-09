"""
cow/probabilities/abortion.py
==============================
Abortion (pregnancy loss) probability: p_abort.

Three methods, most-confident-usable as default (method='formula'):
- table:   the ORIGINAL uncited placeholder table (kept byte-unchanged,
           still selectable -- see abortion_table below).
- formula: Albaaj et al. (2022)-derived, ACTIVE DEFAULT -- see
           p_abortion_albaaj2022 and cow.parameters.ABORTION_ALBAAJ2022 for
           the full published-vs-our-adaptation breakdown (citation,
           published windows, our mathematical conversion, our modeling
           assumptions, confidence/status -- all as separate, inspectable
           fields, never conflated).
- model:   None (unchanged).
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from cow.parameters import scale, UNIT_DAYS, MAX_DP, MAX_LACTATION, ABORTION_ALBAAJ2022, ABORTION_PARITY_RELATIVE_RISK
from cow.lookup_tables import (
    MONTH, N_GEST_MONTHS, full_grid, register_full_table, get_full_table, table_to_array,
    survival_from_monthly, step_probability_matrix,
)
from cow.probabilities.base import lookup, resolve_method, dp_bucket_abort, age_offset_slots


abortion_table = pd.DataFrame([
    {'dp_bucket': '1-60',   'p': 0.040},
    {'dp_bucket': '61-150', 'p': 0.015},
    {'dp_bucket': '151-270','p': 0.005},
    {'dp_bucket': '271+',   'p': 0.000},
])


def p_abortion_table(states, context):
    """Original uncited placeholder table -- UNCHANGED, still selectable via
    method='table'. No citation (see docs/LIMITATIONS_REGISTER.md L-006)."""
    dp = states['dp'].values
    return scale(lookup(abortion_table, {'dp_bucket': dp_bucket_abort(dp)}))


def p_abortion_formula(states, context):
    """Original uncited exponential-decay formula -- UNCHANGED, but no
    longer the active default (superseded in that role by
    p_abortion_albaaj2022 below). Retained for reference/comparison; not
    wired into abortion_spec['methods'] any more since a spec has exactly
    one 'formula' slot -- call this function directly if you need it."""
    dp = states['dp'].values
    dp_days = dp * UNIT_DAYS
    base, decay_days = 0.045, 80
    return scale(base * np.exp(-dp_days / decay_days))


def _albaaj_daily_hazard(dp_days):
    """Per-day abortion hazard at real gestation-day dp_days, built from
    cow.parameters.ABORTION_ALBAAJ2022. See that dict's own 'published' /
    'mathematical_conversion' / 'modeling_assumptions' keys for the full,
    explicit published-vs-adaptation breakdown -- summarized inline here:

    1. Each published window's (loss_rate, D-day span) -> per-day hazard
       via h = 1-(1-loss_rate)**(1/D) [OUR mathematical conversion, an
       exact identity -- not a biological claim].
    2. Days covered by >1 window (the paper's own acknowledged 30-32
       overlap, plus the shared boundaries at day 45 and day 60) use the
       UNWEIGHTED AVERAGE of the covering windows' hazards [OUR modeling
       ADAPTATION -- explicitly not published, not literature-derived].
    3. Outside the paper's studied range (dp_days<19 or >90), fall back to
       the prior uncited table's monthly rates, converted the same way,
       UNCHANGED from before this feature existed.
    """
    windows = ABORTION_ALBAAJ2022['published']['windows']
    window_daily_hazard = np.array([
        1 - (1 - w['loss_rate']) ** (1 / (w['days'][1] - w['days'][0]))
        for w in windows
    ])

    legacy = ABORTION_ALBAAJ2022['outside_published_range']['legacy_monthly_rates']
    def _legacy_daily(monthly_p):
        return 0.0 if monthly_p == 0 else 1 - (1 - monthly_p) ** (1 / 30)
    legacy_daily = {k: _legacy_daily(v) for k, v in legacy.items()}

    hazard = np.zeros_like(dp_days, dtype=float)
    coverage_count = np.zeros_like(dp_days, dtype=float)

    for w, h in zip(windows, window_daily_hazard):
        lo, hi = w['days']
        covered = (dp_days >= lo) & (dp_days <= hi)
        hazard = np.where(covered, hazard + h, hazard)
        coverage_count = np.where(covered, coverage_count + 1, coverage_count)

    covered_any = coverage_count > 0
    hazard = np.where(covered_any, hazard / np.maximum(coverage_count, 1), hazard)

    # Outside the studied range -- legacy fallback, unchanged.
    hazard = np.where((dp_days >= 1) & (dp_days <= 18), legacy_daily['1-18d'], hazard)
    # Days 91-270: Lee & Kim (2007) second / third trimester loss proportions (cow.parameters, 'late_windows').
    for w in ABORTION_ALBAAJ2022['outside_published_range']['late_windows']:
        h_late = 1 - (1 - w['loss_rate']) ** (1 / w['span_days'])
        hazard = np.where((dp_days >= w['days'][0]) & (dp_days <= w['days'][1]), h_late, hazard)
    hazard = np.where(dp_days >= 271, legacy_daily['271+d'], hazard)
    hazard = np.where(dp_days < 1, 0.0, hazard)

    return hazard


# Precomputed day-by-day survival function S(d) = P(no abortion by real
# gestation day d), built ONCE at import from _albaaj_daily_hazard's exact
# per-day values (UNCHANGED above) over every real day 0..(MAX_DP+1)*
# UNIT_DAYS -- the largest real day any reachable dp step could ever need.
# _albaaj_daily_hazard has no closed-form CDF (it's a piecewise/windowed
# function, unlike cow/probabilities/dnb.py's power-law), so S(d) is built
# as an exact cumulative product over real days instead of a formula, but
# it plays the IDENTICAL role as dnb.py's _dnb_cumulative_powerlaw: it
# lets p_abortion_albaaj2022 below derive a per-step probability from the
# ratio of survival at the step's two real-day endpoints, exactly as
# p_dnb_formula already does (2026-08-27 resolution-consistency fix; see
# that function's docstring for the telescoping-product proof this reuses).
_ALBAAJ_MAX_DAY = (MAX_DP + 1) * UNIT_DAYS
_albaaj_hazard_by_day = _albaaj_daily_hazard(np.arange(_ALBAAJ_MAX_DAY + 1))
_ALBAAJ_SURVIVAL = np.concatenate(([1.0], np.cumprod(1 - _albaaj_hazard_by_day)))


def p_abortion_albaaj2022_legacy_offset(states, context):
    """LEGACY -- RETAINED UNCHANGED, no longer wired into abortion_spec
    (superseded 2026-09-19 by the corrected p_abortion_albaaj2022 below; kept
    for direct comparison, same convention as p_calving_table_legacy). Its
    defect: it treats the START of the dp=k pregnant step as gestation day
    k*UNIT_DAYS, but conception happens somewhere INSIDE the preceding step,
    so the true mean age at that point is (k-0.5)*UNIT_DAYS -- an error of
    half a model step (0.5 d at DAY, 3.5 d at WEEK, 15 d at MONTH). At MONTH
    it skips the day 19-30 loss window (cumulative loss to calving 26.6% vs
    the source hazard's 45.1%). Original docstring follows.

    Was ACTIVE DEFAULT. See _albaaj_daily_hazard's docstring and
    cow.parameters.ABORTION_ALBAAJ2022 for the full source/derivation of
    the underlying per-day hazard -- UNCHANGED by this fix.

    RESOLUTION-CONSISTENCY FIX (2026-08-27): the prior conversion
    evaluated h_daily at a SINGLE real day (the step's start, dp*UNIT_DAYS)
    and compounded it as if constant for the entire step
    (1-(1-h_daily)**UNIT_DAYS) -- exact only when the hazard doesn't change
    within the step, which fails at MONTH resolution since Albaaj's
    windows change every 13-30 days (a 30-day MONTH step can straddle 2-3
    windows; a 7-day WEEK step almost never does). Confirmed by direct
    comparison: this previously made MONTH's single-step abortion
    probability ~2.5x WEEK's during the highest-risk early-pregnancy
    window, which was the dominant driver of MONTH's structurally lower
    long-run Pregnant share (see the resolution-consistency diagnosis).

    This now derives the per-step probability from the exact ratio of the
    precomputed day-by-day survival function _ALBAAJ_SURVIVAL at the
    step's two real-day endpoints -- 1 - S(d_next)/S(d_now) -- the
    identical construction cow/probabilities/dnb.py::p_dnb_formula already
    uses, adapted here for a piecewise (not closed-form) hazard curve via
    a precomputed cumulative-product table instead of an analytic CDF.

    At DAY resolution (UNIT_DAYS=1) this is mathematically identical to
    the prior formula (a single day's survival ratio IS that day's
    hazard) -- only WEEK/MONTH's within-step approximation is corrected."""
    dp = states['dp'].values
    d_now = np.clip(dp * UNIT_DAYS, 0, _ALBAAJ_MAX_DAY)
    d_next = np.clip(d_now + UNIT_DAYS, 0, _ALBAAJ_MAX_DAY)
    s_now = _ALBAAJ_SURVIVAL[d_now]
    s_next = _ALBAAJ_SURVIVAL[d_next]
    return np.where(s_now > 0, 1 - s_next / np.where(s_now > 0, s_now, 1.0), 1.0)


def _mean_survival_at_pregnant_step_start(max_step):
    """T[j] = mean survival S(age) at the START of the j-th pregnant step
    (j=1 is the first step after the conception step), averaging over the
    unknown conception time inside the preceding step.

    The cow chain only knows dp=j (steps since conception), not the exact
    conception day. Conception happens uniformly inside the transition step
    that sets dp=1, so the gestation age at the start of pregnant step j is
        age = (j - 1) * UNIT_DAYS + u,   u ~ Uniform(0, UNIT_DAYS).
    Averaging the source survival S over u (slot midpoints, linear
    interpolation between whole days) gives the fraction of pregnancies still
    present at that boundary -- the same resolution-independent biology at
    MONTH, WEEK and DAY (2026-09-19 fix; see docs/MODEL_CHANGELOG.md).
    T[0] = 1 (unused: an Open cow has no pregnancy to lose)."""
    day_max = (max_step + 2) * UNIT_DAYS
    hazard = _albaaj_daily_hazard(np.arange(day_max + 1))
    S = np.concatenate(([1.0], np.cumprod(1 - hazard)))
    grid = np.arange(len(S))
    slot_mid = age_offset_slots()
    T = np.ones(max_step + 2)
    for j in range(1, max_step + 2):
        T[j] = np.mean(np.interp((j - 1) * UNIT_DAYS + slot_mid, grid, S))
    return T


_ABORT_MEAN_SURVIVAL = _mean_survival_at_pregnant_step_start(MAX_DP)
# p_step[k] = P(lose the pregnancy during pregnant step k | still pregnant at its start)
_ABORT_STEP_P = np.zeros(MAX_DP + 1)
_ABORT_STEP_P[1:] = 1 - _ABORT_MEAN_SURVIVAL[2:MAX_DP + 2] / _ABORT_MEAN_SURVIVAL[1:MAX_DP + 1]
# Step 1 must ALSO carry the loss that occurs between the (uniformly timed) conception and the first step
# boundary: the chain enters dp=1 with every pregnancy alive, but the true survival at that boundary is
# T[1] = mean S(u) < 1 whenever the hazard starts inside the conception step (Albaaj loss begins at gestation
# day 19, so at MONTH T[1] ~ 0.95; at WEEK/DAY T[1] = 1 and this is a no-op). Dividing by T[1] here (as in the
# other steps) silently dropped that loss and gave 41.7% cumulative loss at MONTH vs 45.1% in the source
# (2026-09-19 correction). The denominator for step 1 is therefore 1, so survival at boundary 2 is T[2].
_ABORT_STEP_P[1] = 1 - _ABORT_MEAN_SURVIVAL[2]


def p_abortion_albaaj2022(states, context):
    """ACTIVE DEFAULT (corrected 2026-09-19). Per-step pregnancy-loss
    probability from the UNCHANGED Albaaj et al. (2022) per-day hazard
    (_albaaj_daily_hazard; cow.parameters.ABORTION_ALBAAJ2022), evaluated at
    the correct gestation age of each pregnant step: the ratio of average
    survival at the step's two boundaries, where the average is over the
    uniform conception time inside the preceding step (see
    _mean_survival_at_pregnant_step_start). Resolution-consistent: the
    cumulative loss to calving is ~45% at MONTH/WEEK/DAY, matching the source
    hazard, where the legacy mapping gave 26.6% at MONTH. Rows with dp=0
    (Open/DNB) return 0 -- they carry no pregnancy. Row-stochasticity is
    unaffected (p in [0,1] by monotone survival)."""
    dp = np.clip(states['dp'].values, 0, MAX_DP)
    return _ABORT_STEP_P[dp]


# ============================================================
# FULL LOOKUP TABLE (lactation 1..MAX_LACTATION x every native month of GESTATION)
# ============================================================
# Monthly pregnancy-loss probability (conditional on being pregnant at the month's start) by
# lactation and gestation month. Default values are the MONTHLY EQUIVALENT of the active
# Albaaj-derived day hazard (1 - S(30(g+1)) / S(30 g)), replicated across lactations -- the
# source reports no parity dependence, so nothing new is introduced; the table makes the
# gestation-month shape visible and editable per lactation. The within-month shape of the
# day hazard is flattened (constant hazard inside a month) -- that is why 'formula' stays the
# default and 'table' is the editable alternative. Override with cow/data/abortion_full.csv.

def _build_abortion_full():
    g = full_grid(N_GEST_MONTHS)
    days = np.arange(N_GEST_MONTHS * MONTH + 1)
    S = np.concatenate(([1.0], np.cumprod(1 - _albaaj_daily_hazard(days[:-1]))))
    month_p = 1 - S[(np.arange(N_GEST_MONTHS) + 1) * MONTH] / S[np.arange(N_GEST_MONTHS) * MONTH]
    g['p'] = month_p[g['month'].values]
    m = g['month'].values
    albaaj = m <= 2                           # gestation months 0-2 (days 0-89): Albaaj et al. (2022) windows 19-90 d
    leekim = (m >= 3) & (m <= 8)              # months 3-8 (days 90-269): Lee & Kim (2007) trimester proportions
    # the sources are POOLED over parities: the one value is repeated for every lactation (no evidence that lactations differ)
    g['basis'] = np.where(albaaj | leekim, 'repeated', 'placeholder')
    g['note'] = np.where(
        albaaj,
        'monthly equivalent of the Albaaj 2022 windows (days 19-90; days 1-18 use the legacy rate); the source is pooled across parities, so the same value is REPEATED for every lactation (no lactation-specific evidence)',
        np.where(leekim,
                 'monthly equivalent of Lee & Kim 2007 trimester loss proportions (2nd: 34/988 over days 91-180; 3rd: 22/954 over 181-260, repeated to 270); pooled across parities here (parity effect: see abortion model method)',
                 'legacy uncited rate (days 270+: zero), same value for every lactation'))
    return g


register_full_table('abortion_full', _build_abortion_full, N_GEST_MONTHS)


@lru_cache(maxsize=1)
def _abortion_step_arrays():
    """(n_lactations, MAX_DP+1) per-pregnant-step loss probability. Step dp=1 is the first step after
    the conception step (the pregnancy exists at its start: survival there is 1), same convention as
    the active formula; later steps use the entry-averaged boundary survival ratio."""
    P = table_to_array(get_full_table('abortion_full'), N_GEST_MONTHS)[0]
    out = step_probability_matrix(P, MAX_DP, first_age_step=1)
    out[:, 0] = 0.0
    from cow.lookup_tables import age_offset_slots
    slot = age_offset_slots()
    for i in range(P.shape[0]):
        S = survival_from_monthly(P[i])
        out[i, 1] = 1 - np.mean(S(1 * UNIT_DAYS + slot))      # Sbar at start of step 2; start of step 1 := 1
    return out


def p_abortion_full_table(states, context):
    """'table' method: full lactation x gestation-month table (cow/data/abortion_full.csv overrides),
    integrated by the single time-discretization rule. dp = 0 (open) returns 0."""
    lac = np.clip(states['lactation'].values, 1, MAX_LACTATION) - 1
    dp = np.clip(states['dp'].values, 0, MAX_DP)
    return _abortion_step_arrays()[lac, dp]


def p_abortion_parity_model(states, context):
    """UNWIRED sensitivity (2026-10-08 cleanup; not a fitted model): proportional-hazards model -- the pooled DP-clock step probability (the active formula) raised to a
    lactation relative risk, p_L = 1 - (1 - p_base)^RR_L, with RR from ABORTION_PARITY_RELATIVE_RISK (lactation 1 / 2 / >=3 =
    0.81 / 0.96 / 1.23; class B + normalisation, see cow/parameters.py). DP stays the clock; DIM is deliberately NOT a
    covariate (Fernandez-Novo 2020 and Cordeiro 2026 find no independent DIM effect). Evidence on parity is mixed, so this
    is selectable, not the default."""
    keys = sorted(ABORTION_PARITY_RELATIVE_RISK)
    rr = np.array([ABORTION_PARITY_RELATIVE_RISK[min(int(L), keys[-1])] for L in states['lactation'].values])
    return 1.0 - (1.0 - p_abortion_albaaj2022(states, context)) ** rr


abortion_spec = {
    'name': 'abortion',
    'produces': 'p_abort',
    'depends_on': ['lactation', 'dp'],
    'method': 'formula',
    # formula: Albaaj-derived day hazard (ACTIVE, literature-derived) | table: FULL lactation x gestation-month
    # table, switch to it to edit per lactation | model: unavailable -- not yet fitted (parity evidence is mixed; the
    # relative-risk function p_abortion_parity_model below is an UNWIRED sensitivity, not a fitted model).
    # (The old 4-bucket p_abortion_table is retained, unwired.)
    'methods': {'table': p_abortion_full_table, 'formula': p_abortion_albaaj2022, 'model': None},
}

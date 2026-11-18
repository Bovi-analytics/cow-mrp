"""
cow/probabilities/conception.py
================================
Conception probability: p_preg (successful pregnancy start, given Open/DNB).
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from cowmrp.domains.cow.parameters import scale, to_units, UNIT_DAYS, RATE_PERIOD_DAYS, MAX_DIM, MAX_LACTATION, VOLUNTARY_WAITING_PERIOD_DAYS, LAST_BREEDING_DIM_DAYS, CONCEPTION_PARITY_ODDS_RATIO
from cowmrp.domains.cow.lookup_tables import (
    N_DIM_MONTHS, full_grid, register_full_table, get_full_table, table_to_array, step_probability_matrix,
)
from cowmrp.domains.cow.probabilities.base import lookup, resolve_method, lactation_bucket_breeding, dim_bucket_conception, mean_step_probability


conception_table = pd.DataFrame([
    {'lactation_bucket': '1',  'dim_bucket': '0-50',   'p': 0.00},
    {'lactation_bucket': '2+', 'dim_bucket': '0-50',   'p': 0.00},
    {'lactation_bucket': '1',  'dim_bucket': '51-100', 'p': 0.45},
    {'lactation_bucket': '1',  'dim_bucket': '101-200','p': 0.45},
    {'lactation_bucket': '1',  'dim_bucket': '201-350','p': 0.45},
    {'lactation_bucket': '1',  'dim_bucket': '351-449','p': 0.35},
    {'lactation_bucket': '2+', 'dim_bucket': '51-100', 'p': 0.35},
    {'lactation_bucket': '2+', 'dim_bucket': '101-200','p': 0.35},
    {'lactation_bucket': '2+', 'dim_bucket': '201-350','p': 0.35},
    {'lactation_bucket': '2+', 'dim_bucket': '351-449','p': 0.25},
])


def p_conception_table_legacy(states, context):
    """Earlier implementation of the 'table' method, not wired into
    conception_spec['methods']; kept for direct comparison with p_conception_table.
    It looks up ONE conception_table rate for the whole model step, using the
    step's own (to_units-rounded) DIM bucket, which smears a bucket boundary
    across the entire step's real-day span. At MONTH resolution this delays the
    effective start of conception attempts (VWP) to real day 90 instead of 50
    (a WEEK step delays it only to day 56), which is the dominant driver,
    alongside the analogous issue in abortion, of MONTH's structurally higher
    long-run Open share. Same rates and thresholds as p_conception_table; only
    the per-step conversion differs."""
    lac, dim = states['lactation'].values, states['dim'].values
    return scale(lookup(conception_table, {
        'lactation_bucket': lactation_bucket_breeding(lac),
        'dim_bucket': dim_bucket_conception(dim),
    }))


# Real calendar-day bucket edges mirroring conception_table: the same 50/350-day
# values that DIM_CONCEPTION_VWP/MID/HIGH/VHIGH = to_units(50/100/200/350) in
# cow/probabilities/base.py convert to (to_units-rounded) step units for the
# earlier lookup above. ('101-200' and '201-350' share the '51-100' rate in the
# table, so they collapse to one real-day range here; no rate value changes.)
# Kept as raw days (not step-rounded) because _conception_survival_product below
# must evaluate which bucket a given real day falls in independently of the step
# size set by TIME_UNIT.
_CONCEPTION_VWP_DAYS = VOLUNTARY_WAITING_PERIOD_DAYS   # single source: cow/parameters.py
_CONCEPTION_VHIGH_DAYS = 350


def _conception_survival_product(dim_step, lac):
    """Day-exact integration of conception_table's rates over
    every real calendar day in [dim_step*UNIT_DAYS, dim_step*UNIT_DAYS +
    UNIT_DAYS) -- so a 30-day MONTH step represents the same cumulative
    conception exposure as ~4.3 WEEK steps or 30 DAY steps would, instead
    of a single coarse per-step lookup. Each real day's monthly-native
    rate converts to a daily hazard via 1-(1-rate)**(1/RATE_PERIOD_DAYS)
    -- the same identity scale() already encodes -- and the survival
    product is taken across all UNIT_DAYS real days in the step. Fully
    vectorized (no per-row Python loop) since UNIT_DAYS can be up to 30
    and this runs over the whole state table.

    At DAY resolution (UNIT_DAYS=1) this collapses to exactly one day's
    rate lookup, identical to the single-lookup result. Only the
    within-step bucket smearing at WEEK and MONTH is corrected."""
    offsets = np.arange(UNIT_DAYS)
    real_days = dim_step[:, None] * UNIT_DAYS + offsets[None, :]   # shape (n_states, UNIT_DAYS)
    lac_is_1 = (lac == 1)[:, None]

    high_rate = np.where(lac_is_1, 0.45, 0.35)   # '51-100'/'101-200'/'201-350' -- identical rate
    vhigh_rate = np.where(lac_is_1, 0.35, 0.25)  # '351-449'
    monthly_rate = np.select(
        [real_days <= _CONCEPTION_VWP_DAYS, real_days <= _CONCEPTION_VHIGH_DAYS],
        [np.zeros_like(real_days, dtype=float), np.broadcast_to(high_rate, real_days.shape)],
        default=np.broadcast_to(vhigh_rate, real_days.shape),
    )
    daily_hazard = np.where(monthly_rate == 0, 0.0, 1 - (1 - monthly_rate) ** (1.0 / RATE_PERIOD_DAYS))
    survival = np.prod(1 - daily_hazard, axis=1)
    return 1 - survival


def _conception_survival_function(lac1):
    """S(a) = P(not yet conceived at DIM a days) for a first-lactation (lac1) or later cow,
    from conception_table's monthly rates (0 before the voluntary-waiting edge,
    the same 50/350-day edges as _conception_survival_product), linearly interpolated
    between whole days."""
    days = np.arange((MAX_DIM + 3) * UNIT_DAYS + 2)
    high, vhigh = (0.45, 0.35) if lac1 else (0.35, 0.25)
    rate = np.where(days <= _CONCEPTION_VWP_DAYS, 0.0, np.where(days <= _CONCEPTION_VHIGH_DAYS, high, vhigh))
    hazard = np.where(rate == 0, 0.0, 1 - (1 - rate) ** (1.0 / RATE_PERIOD_DAYS))
    S = np.concatenate(([1.0], np.cumprod(1 - hazard)))
    grid = np.arange(len(S))
    return lambda age: np.interp(age, grid, S)


def p_conception_table(states, context):
    """'table' method on the conception_table rates. The DIM at the start of step k is the true
    age k*UNIT_DAYS + u (u uniform on (0, UNIT_DAYS): the cow calved somewhere inside the previous
    step), not exactly k*UNIT_DAYS. The earlier day-exact integration (p_conception_table_legacy_offset
    below) was half a step too young at the first boundary, the mirror image of the gestation-age
    issue in abortion.py. Probability = boundary-survival ratio averaged over u (base.mean_step_probability).
    Days from calving to the conception that reaches calving: 142 at MONTH and 144 at WEEK, against a
    CDCB-derived ~137 (the earlier day-exact integration gives 140 at MONTH only because its
    half-step offset error cancels a separate error elsewhere)."""
    lac = states['lactation'].values
    dim = np.clip(states['dim'].values, 0, MAX_DIM)
    p1 = mean_step_probability(_conception_survival_function(True), MAX_DIM)
    p2 = mean_step_probability(_conception_survival_function(False), MAX_DIM)
    return np.where(lac == 1, p1[dim], p2[dim])


def p_conception_table_legacy_offset(states, context):
    """Earlier day-exact version, not wired into conception_spec; kept for comparison. It integrates
    the rates over the step's real days starting at exactly dim*UNIT_DAYS, i.e. treats a freshly
    calved cow as exactly DIM 0 at the boundary (half a step too young on average)."""
    lac, dim = states['lactation'].values, states['dim'].values
    return _conception_survival_product(dim, lac)


# ============================================================
# FULL LOOKUP TABLE (lactation 1..MAX_LACTATION x every native month since calving)
# ============================================================
# Monthly conception probability per eligible open cow, by lactation and month in milk.
# Default values = the conception_table rates above replicated across lactations 2..MAX
# (the '2+' bucket) and across months; the voluntary waiting period is not baked into the
# rows but applied by day (VOLUNTARY_WAITING_PERIOD_DAYS) so it stays exact at every
# resolution. Month 11 (days 330-359) carries the '201-350' rate, so the 350-day edge of
# conception_table falls on a month boundary (about 10 days off).
# Override by placing cow/data/conception_full.csv.

def _parity_odds_ratio(lac):
    """CONCEPTION_PARITY_ODDS_RATIO per lactation (lactations beyond the last key repeat it)."""
    keys = sorted(CONCEPTION_PARITY_ODDS_RATIO)
    return np.array([CONCEPTION_PARITY_ODDS_RATIO[min(int(L), keys[-1])] for L in lac])


def _build_conception_full():
    """Monthly conception probability by lactation x month since calving. Lactation-1 level = the
    conception_table value (CDCB-consistent aggregate); lactation >= 2 = the same value moved on the odds
    scale by CONCEPTION_PARITY_ODDS_RATIO (Cook & Green 2016, Cordeiro 2026; cow/parameters.py). This builder IS the
    odds-ratio 'model' method; the 'table' method reads it (or cow/data/conception_full.csv)."""
    g = full_grid(N_DIM_MONTHS)
    day_for_bucket = np.maximum(g['month'].values * 30 + 15, _CONCEPTION_VWP_DAYS + 1)
    bucket_day = day_for_bucket
    dim_bucket = np.select(
        [bucket_day <= 100, bucket_day <= 200, bucket_day <= 350],
        ['51-100', '101-200', '201-350'], default='351-449')
    p1 = lookup(conception_table, {
        'lactation_bucket': np.full(len(g), '1'),
        'dim_bucket': dim_bucket,
    })
    odds = p1 / (1.0 - p1) * _parity_odds_ratio(g['lactation'].values)
    g['p'] = odds / (1.0 + odds)
    lac1 = g['lactation'].values == 1
    g['basis'] = np.where(lac1, 'placeholder', np.where(g['lactation'].values <= 3, 'derived', 'repeated'))   # lactation 2: odds ratio; lactation 3: first member of the '>=3' group (derived); 4-10: that grouped ratio repeated
    g['note'] = np.where(
        lac1,
        "conception_table level (consistent with the CDCB days-open aggregate, not a direct estimate); voluntary waiting period applied by day",
        "lactation-1 level (placeholder) x parity odds ratio 0.68 (lactation 2, derived) / 0.47 (lactations >=3: one grouped value repeated for 3-10) from Cook & Green 2016 and Cordeiro 2026")
    return g


register_full_table('conception_full', _build_conception_full, N_DIM_MONTHS)


@lru_cache(maxsize=1)
def _conception_step_arrays():
    """(n_lactations, MAX_DIM+1) per-step conception probabilities."""
    P = table_to_array(get_full_table('conception_full'), N_DIM_MONTHS)[0]
    return step_probability_matrix(P, MAX_DIM, first_age_step=0, zero_until_day=_CONCEPTION_VWP_DAYS,
                                   zero_after_day=LAST_BREEDING_DIM_DAYS)


def conception_entry_steps():
    """DIM steps k (start-of-step index) in which a conception can happen for at least one lactation
    (per-step probability > 0). A pregnant cow's dim - dp stays equal to the step she conceived in, so this
    set is exactly the reachable (dim - dp) values of all pregnant states -- used to build the state space."""
    arr = _conception_step_arrays()
    return np.where((arr > 1e-12).any(axis=0))[0]


def p_conception_full_table(states, context):
    """Default 'table' method: full lactation x month table with the waiting period applied by day,
    integrated by the single time-discretization rule (cow/lookup_tables.py). Breeding is also cut off
    by day (LAST_BREEDING_DIM_DAYS) inside the hazard, so a step straddling the cutoff keeps only the
    share of its days before it. Equals the bucket method (p_conception_table) except for the
    350-day edge, which falls on the month-11 boundary."""
    lac = np.clip(states['lactation'].values, 1, MAX_LACTATION) - 1
    dim = np.clip(states['dim'].values, 0, MAX_DIM)
    return _conception_step_arrays()[lac, dim]


def p_conception_model(states, context):
    """Not wired into conception_spec (it re-derives the default table, so it is not a separate model). Odds-ratio model: logit(p) = logit(p_lactation1(DIM)) + ln(OR_parity), built directly from the
    parameters (ignores any cow/data/conception_full.csv override). Same numbers as the default table by construction;
    it exists so the covariate structure (reference + odds ratios) is explicit and can be re-estimated."""
    df = _build_conception_full()
    P = table_to_array(df, N_DIM_MONTHS)[0]
    arr = step_probability_matrix(P, MAX_DIM, first_age_step=0, zero_until_day=_CONCEPTION_VWP_DAYS, zero_after_day=LAST_BREEDING_DIM_DAYS)
    lac = np.clip(states['lactation'].values, 1, MAX_LACTATION) - 1
    dim = np.clip(states['dim'].values, 0, MAX_DIM)
    return arr[lac, dim]


def p_conception_formula(states, context):
    lac, dim = states['lactation'].values, states['dim'].values
    vwp = to_units(50)
    peak1, peak2 = 0.30, 0.24
    mid, width = to_units(150), to_units(80)
    peak = np.where(lac == 1, peak1, peak2)
    logistic = peak / (1 + np.exp((dim - mid) / width))
    return scale(np.where(dim < vwp, 0.0, logistic))


conception_spec = {
    'name': 'conception',
    'produces': 'p_preg',
    'depends_on': ['lactation', 'dim'],
    'method': 'table',
    # table: full lactation x month table (default; lactation >= 2 = lactation-1 level x parity odds ratio) | formula: logistic placeholder |
    # model: unavailable -- no fitted conception model exists (p_conception_model below re-derives the table; kept unwired)s
    # (the previous bucket-based p_conception_table above is retained, unwired, for comparison)
    'methods': {'table': p_conception_full_table, 'formula': p_conception_formula, 'model': None},
}

"""
cow/parameters.py
=================
Every editable number for the cow domain, in one place.
Run this first -- all other cow files import from here.

HOW TO USE:
- Change TIME_UNIT to "day", "week", or "month"
- Adjust MAX_LACTATION, milk cutoff, health rates, prices, etc.
- All other files read these values automatically.

Note: TIME_UNIT, MAX_LACTATION, MAX_DIM_DAYS, MAX_DP_DAYS can also be set
via environment variables (COW_TIME_UNIT, COW_MAX_LACTATION,
COW_MAX_DIM_DAYS, COW_MAX_DP_DAYS) -- this is how benchmark.py runs
different-sized models without editing this file. If those env vars
aren't set, the hardcoded defaults below are used, same as always.
"""

import os
import numpy as np

# ----------------------------------------------------------
# TIME SETTINGS
# ----------------------------------------------------------
TIME_UNIT = os.environ.get("COW_TIME_UNIT", "month")   # "day", "week", or "month"
UNIT_DAYS = {"day": 1, "week": 7, "month": 30}[TIME_UNIT]
RATE_PERIOD_DAYS = 30                   # lookup tables are natively monthly

def to_units(days):
    """Convert calendar days to the current time unit."""
    return max(1, round(days / UNIT_DAYS))

def last_step_within(days):
    """Last DIM step k whose MID-STEP age is <= `days`. A cow in state dim = k is on average k*U + U/2 days
    into lactation at the step start (she calved at a uniform time inside the previous step), so she spends
    the step at mean age (k+1)*U. 'Allowed up to day T' therefore means steps 0 .. floor(T/U) - 1
    (month: T=300 -> step 9, i.e. the first 10 months)."""
    return max(0, int(days) // UNIT_DAYS - 1)


def first_step_past(days):
    """First DP step whose MID-STEP age (dp*U, same entry convention, gestation counted from the
    conception step) is >= `days`: ceil(days / U). Dry-off at 222 d is dp 8 at month, 32 at week, 222 at day."""
    return max(1, -(-int(days) // UNIT_DAYS))


def dim_step_of_days(days):
    """DIM STEP whose mean age matches `days` since calving: a cow in state dim = k spends the step at mean
    age (k+1)*U (entry-offset convention, cow/lookup_tables.py), so k = round(days/U) - 1 (>= 0)."""
    return max(0, int(round(days / UNIT_DAYS)) - 1)


def dp_step_of_days(days):
    """DP STEP whose mean gestation age matches `days`: mean age of pregnant step dp is dp*U, so dp = round(days/U);
    0 days -> 0 (open), any pregnant query is at least step 1."""
    return 0 if days <= 0 else max(1, int(round(days / UNIT_DAYS)))


def scale(p):
    """Scale a monthly probability to the current time unit."""
    return 1 - (1 - p) ** (UNIT_DAYS / RATE_PERIOD_DAYS)


# ----------------------------------------------------------
# LIFE-CYCLE PARAMETERS
# ----------------------------------------------------------
MAX_LACTATION = int(os.environ.get("COW_MAX_LACTATION", 10))
MAX_DIM_DAYS = int(os.environ.get("COW_MAX_DIM_DAYS", 1000))   # lookup extent; raised below to cover the state table
# STATE_DIM_CAP_DAYS: last day a conception can START in the state table (numerical extent, not biology). Conception past LAST_BREEDING_DIM_DAYS has probability 0 but its states exist.
STATE_DIM_CAP_DAYS = int(os.environ.get("COW_STATE_DIM_CAP_DAYS", 750))
MAX_DP_DAYS = int(os.environ.get("COW_MAX_DP_DAYS", 300))
# the lookup tables must reach the last pregnant DIM of the state table (cap + gestation + the re-opening step)
MAX_DIM_DAYS = max(MAX_DIM_DAYS, STATE_DIM_CAP_DAYS + MAX_DP_DAYS + 2 * UNIT_DAYS)

# Gestation length of Holstein-sired calves, days.
#   Basiel BL, Barragan AA, Felix TL, Dechow CD (2024). "The impact of beef
#   sire breed on dystocia, stillbirth, gestation length, health, and
#   lactation performance of cows that carry beef x dairy calves." J Dairy
#   Sci 107(4):2241-2252. DOI: 10.3168/jds.2023-24112.
#   Holstein-sired calves 277 +/- 0.15 d (least-squares mean +/- SE from a mixed
#   model; 75,256 lactations, 39,249 cows, 10 US herds in the Northeast and
#   Midwest, 2010-2023). Holstein-sired is the assumed dam x sire population
#   (the conventional breeding program).
# Used by the calving 'table' method and by the midpoint of its 'formula'
# method, and by dry-off timing, the body-weight pregnancy gain and the
# youngstock heifer-pregnancy hazard, so changing it here changes all of them.
GESTATION_LENGTH_DAYS = 277
# Standard dry-period length, days. 42-60 days is the commonly recommended
# range and 55 days is the traditional standard practice. No single primary
# study that fixes 55 exactly is cited; it is a standard-practice value.
DRY_OFF_DAYS_BEFORE_CALVING = 55
DRY_OFF_DP_DAYS = GESTATION_LENGTH_DAYS - DRY_OFF_DAYS_BEFORE_CALVING

# Convert to time units
MAX_DIM = to_units(MAX_DIM_DAYS)
MAX_DP = to_units(MAX_DP_DAYS)
DRY_OFF_DP = first_step_past(DRY_OFF_DP_DAYS)   # mid-step rule (see first_step_past)

# ----------------------------------------------------------
# FARMER REPRODUCTIVE-PROGRAM SETTINGS
# ----------------------------------------------------------
# LAST_BREEDING_DIM: last DIM at which an open cow is bred. Past it, conception
# probability is 0 and the breeding cost stops (the cow is "do not breed", a
# REGION of the state space, not a state). Same for every lactation.
# MILK_FLOOR_KG: an open cow past LAST_BREEDING_DIM whose lactation-curve milk
# is at or below this yield is culled (replaced) -- the floor is NEVER looked
# at before LAST_BREEDING_DIM, and never for pregnant cows. Both are
# management settings, not biological constants.
VOLUNTARY_WAITING_PERIOD_DAYS = 50   # no breeding before this DIM (single source: conception + breeding cost)
LAST_BREEDING_DIM_DAYS = 300
MILK_FLOOR_KG = 22.7
LAST_BREEDING_DIM = last_step_within(LAST_BREEDING_DIM_DAYS)   # mid-step rule: month 9 = the first 10 months
ENABLE_CULL_REPRO = True    # milk-floor reproductive cull (replaces health/milk/DNB culling)

# After a pregnancy loss a cow is eligible to be bred again immediately (the model has no detection / re-synchronization
# delay). A loss happens at a uniform time inside a step, but the chain re-opens her only at the step boundary, which
# silently removes on average half a step of re-breeding exposure -- an error proportional to the step length (it
# shortened the MONTH cycle by ~7% and removed ~0.3 calvings per cycle).
# True: a cow that loses her pregnancy keeps the remaining half step of conception exposure (she may conceive again in
# the same step: P = 1 - sqrt(1 - q), q = that step's conception probability). This removes the dependence of the
# re-breeding exposure on TIME_UNIT (the effect vanishes as the step shrinks). False: the cow is re-opened only at the
# next step boundary.
RESUME_BREEDING_AFTER_LOSS = True


# ----------------------------------------------------------
# REPRODUCTION -- CONCEPTION / PREGNANCY RATE (cow/probabilities/conception.py)
# ----------------------------------------------------------
# Source record for the conception_table rates (0.45/0.45/0.45/0.35 for lactation 1 and
# 0.35/0.35/0.35/0.25 for lactation 2+; see conception.py). Unlike ABORTION_ALBAAJ2022 and
# MORTALITY_ANNUAL_ANCHOR below, this dict does not feed a formula; it documents what supports
# the table.
CONCEPTION_RATE_PROVENANCE = {
    "citation_sources": [
        (
            "CDCB (Council on Dairy Cattle Breeding) national DHI "
            "reproductive database, queries.uscdcb.com/publish/dhi/current/"
            "reproall.html: Holstein, 2018-2020 "
            "-- days calving-to-first-breeding 80-83 days; conception rate "
            "for first breedings 0.38, for all breedings 0.35-0.36 "
            "(per-service metrics, national)."
        ),
        (
            "CDCB, First Service to Conception (FSC) trait definition, "
            "uscdcb.com/first-service-to-conception: "
            "mean 55.60 days, Holstein -- days from first "
            "breeding to conception, a per-lactation cumulative metric."
        ),
        (
            "Fodor A, et al. (2019). \"Pregnancy and conception rates by "
            "parity.\" Canadian Journal of Veterinary Research 83(3):"
            "218-227. 23,781 cows, 34 large Hungarian Holstein-Friesian "
            "herds: first-service conception "
            "risk 24.8% (primiparous, n=9,346) vs 17.3% (multiparous, "
            "n=10,448), P<0.001; pregnant-by-200-DIM 65.2% vs 55.4%."
        ),
    ],
    # Aggregate check. CDCB's two official metrics compose into a national days-open
    # benchmark: 80-83 (days to first breeding) + 55.6 (FSC) = about 136-139 days
    # for the current US Holstein average. This is a check of the aggregate outcome,
    # not a source for the individual rates.
    "aggregate_benchmark_check": (
        "With the current conception_table values the model gives a mean "
        "days open (calving to the conception that reaches the next calving, "
        "first-lactation cohort, censored at replacement) of about 142 days at "
        "MONTH and 144 days at WEEK resolution, against the CDCB-derived "
        "national average of about 137 days (see CONCEPTION_INTERPRETATION_2026_09_19)."
    ),
    # No source publishes these exact per-lactation, per-DIM-bucket monthly values.
    "modeling_assumptions": (
        "(1) A single monthly probability, applied uniformly to every "
        "eligible open cow regardless of how many services she has "
        "already had, stands in for the real distinction between "
        "'awaiting first service' and 'awaiting conception after "
        "service'. Both CDCB metrics measure the latter only "
        "(FSC) or a per-service snapshot (conception rate), and the "
        "model has no state variable for service count or for whether a "
        "cow has been bred yet. Only the aggregate (total days open) is "
        "checked against the aggregate benchmark above, not the internal "
        "mechanism against CDCB's internal mechanism. "
        "(2) The lactation 1 versus 2+ split (0.45 vs 0.35) agrees in "
        "direction with the parity gap of Fodor et al. (24.8% vs 17.3% "
        "first-service conception risk), but not in magnitude, since "
        "first-service risk and a recurring monthly probability are "
        "different metrics. (3) The late-DIM decline to 0.35/0.25 "
        "(lactation 1/2+) beyond day 350 has no specific literature "
        "source; it is a plausible modelling choice (worsening "
        "prospects for cows still open very late), not tested against a "
        "study of conception rate versus days already open."
    ),
    "calibration_required": False,
    "user_configurable_parameters": 'cow/probabilities/conception.py::conception_table',
    "confidence_status": (
        "The aggregate days-open outcome is checked against a national "
        "database (CDCB); this is an aggregate check, not a per-value "
        "citation. The split by lactation and the late-DIM decline are "
        "modelling assumptions, supported in direction by Fodor et al. "
        "(2019) but not matched in value."
    ),
}


# ----------------------------------------------------------
# REPRODUCTION -- ABORTION / PREGNANCY LOSS (cow/probabilities/abortion.py)
# ----------------------------------------------------------
# Separates what the source publishes from what the model does with it (see each key's
# comment). Used by abortion.py::p_abortion_albaaj2022 (the default 'formula' method).
# The table method (abortion.py::abortion_table) remains selectable with method='table'.
ABORTION_ALBAAJ2022 = {
    # Source.
    "citation_source": (
        "Albaaj A, Durocher J, LeBlanc SJ, Dufour S (2022). \"Meta-analysis "
        "of the incidence of pregnancy losses in dairy cows at different "
        "stages to 90 days of gestation.\" JDS Communications 4(2):144-148. "
        "DOI: 10.3168/jdsc.2022-0278."
    ),
    # Published values, as printed. Loss rate is conditional on being
    # pregnant at the start of the window (not cumulative from conception).
    # The paper's own words on the overlap:
    # "We set 4 risk periods for PL: early (19 to 32 d) and late (30 to 45
    # d) embryonic periods and early (45 to 60 d) and late (60 to 90 d)
    # fetal periods. These periods were based on the reported data for
    # first and second pregnancy diagnoses, which explains the small
    # overlap between the first 2 risk periods." The paper provides NO
    # day-by-day hazard function and explicitly cautions its own 19-90d
    # aggregate may underestimate the true value.
    "published": {
        "windows": [
            {"days": (19, 32), "loss_rate": 0.27, "stage": "early embryonic"},
            {"days": (30, 45), "loss_rate": 0.13, "stage": "late embryonic"},
            {"days": (45, 60), "loss_rate": 0.07, "stage": "early fetal"},
            {"days": (60, 90), "loss_rate": 0.02, "stage": "later fetal"},
        ],
    },
    # Mathematical conversion: an exact identity between equivalent
    # hazard representations, not a biological claim. Each window's
    # (loss_rate, D-day span) becomes a per-day hazard via
    # h = 1-(1-loss_rate)^(1/D); a per-model-step probability then follows
    # via p_step = 1-(1-h)^UNIT_DAYS -- the same identity
    # cow/probabilities/mortality.py::p_mortality_formula already uses.
    "mathematical_conversion": (
        "h_daily = 1 - (1 - loss_rate) ** (1 / D); "
        "p_step = 1 - (1 - h_daily) ** UNIT_DAYS"
    ),
    # Modelling assumptions (not published, not literature-derived).
    "modeling_assumptions": (
        "(1) Risk is assumed uniform across the days within a "
        "single published window: the paper reports only a window-total "
        "probability, not a within-window shape. "
        "(2) For any day covered by more than one published window (the "
        "acknowledged 19-32/30-45 overlap on days 30-32, plus the single-"
        "day shared boundaries at day 45 and day 60), the "
        "unweighted average of the covering windows' per-day hazards is "
        "used, because the source gives no basis to prefer one over the other."
    ),
    # Outside the paper's studied range (dp_days < 19 or > 90) the rates below are used
    # without extrapolation. Days 1-18 and 271+ use the default monthly rates (no published
    # source); days 91-270 use the Lee & Kim (2007) windows in "late_windows".
    "outside_published_range": {
        "note": (
            "dp_days < 19 or > 90: days 1-18 and 271+ use default monthly rates "
            "that are not derived from a published source; days 91-270 use "
            "the Lee & Kim (2007) windows."
        ),
        "legacy_monthly_rates": {
            "1-18d": 0.040, "91-150d": 0.015, "151-270d": 0.005, "271+d": 0.000,
        },
        # Days 91-270: Lee & Kim (2007), J Vet Sci 8(3):283-288, Table 2
        # (1,001 Holstein pregnancies, 7 Korean herds, ultrasound/palpation diagnosis at 28-60 d): loss of
        # 34/988 pregnancies (3.44 %) in the second trimester (days 91-180) and 22/954 (2.31 %) in the third
        # (days 181-260). The printed proportions are each spread uniformly over their window. The
        # third-trimester value is repeated to day 270 (the source window ends at 260). Cross-checks: De Vries
        # (2006) months 4-8 sum to 2.5 %, Fernandez-Novo (2020) 12.3 % for days 28-110, Corva (2025) 22.5 % for
        # days 28-210 in one grazing herd. These span a wide range, and this value lies between them.
        "late_windows": [
            {"days": (91, 180), "loss_rate": 34 / 988, "span_days": 90, "source": "Lee & Kim 2007 second trimester"},
            {"days": (181, 270), "loss_rate": 22 / 954, "span_days": 80, "source": "Lee & Kim 2007 third trimester (window 181-260, repeated to 270)"},
        ],
    },
    # This is the default 'formula' method of abortion_spec: its rates are
    # peer-reviewed, whereas the table has no citation.
    "default": True,
    # To change the rates, edit the 'windows' list or the outside-range rates above.
    "user_configurable_parameters": "published['windows'], outside_published_range['legacy_monthly_rates']",
    "calibration_required": False,
    "confidence_status": (
        "Published windows and rates: literature-supported. "
        "Mathematical conversion: an exact identity. "
        "Uniform-within-window risk and overlap averaging: modelling "
        "approximations, not the source's."
    ),
}


# ----------------------------------------------------------
# MILK PARAMETERS
# ----------------------------------------------------------
# MilkBot lactation curve parameters. Source: Ehrlich JL (2011) "Quantifying shape of
# lactation curves, and benchmark curves for common dairy breeds and parities", The Bovine
# Practitioner 45(1):88-95, Table 1, Holstein median rows (scale kg/day, ramp d, offset d,
# decay /d): a DHIA data set of 6,459,942 lactations from 17,013 herds, mostly eastern US,
# calving 2005 to mid 2008, DIM 6-400, parities 1, 2 and 3 (third lactation) only;
# lactations 4-10 repeat parity 3 (MILK_CURVE_BASIS 'repeated'). Equation (Ehrlich 2013 PeerJ
# 1:e54 Eq. 5 and Ehrlich 2011 Appendix 1): Y(t) = a * (1 - exp((c - t)/b) / 2) * exp(-d t).
# Table 1 also gives the fitted peak day (88/56/58 d) and M305 (9,296/10,507/10,837 kg), used as
# checks in runners/cow/checks/audit_milk.py.
MILKBOT_EHRLICH2011 = {
    1: {"a": 38.1, "b": 35.2, "c": -3.8, "d": 0.00109},
    2: {"a": 49.5, "b": 27.2, "c": -3.8, "d": 0.00215},
    3: {"a": 53.6, "b": 30.1, "c": -2.4, "d": 0.00243},
}

# Which milk-yield formula cow/states.py::milk_potential_formula dispatches to.
# "wood" (default) | "milkbot" -- see WOOD1967 and MILKBOT_EHRLICH2011. Selecting
# "wood" while WOOD1967['parameters_by_parity'] is uncalibrated raises a clear
# ValueError; it never silently falls back to MilkBot or guesses a value.
MILK_CURVE_METHOD = os.environ.get("COW_MILK_CURVE_METHOD", "wood")

# Lactation-explicit milk curves. The model represents lactations 1..MAX_LACTATION explicitly. Each lactation
# uses the parameters of a source lactation, and the basis of that choice is recorded here:
# 'source' = the cited source reports exactly this lactation; 'derived' = computed from printed values;
# 'repeated' = the source does not distinguish this lactation (it reports lactations 1, 2 and 3 only), so
# lactation 3's parameters are repeated, with no evidence of a difference either way; 'placeholder' = uncited
# modelling value. Replace an entry's source (or add per-lactation parameters to WOOD1967) as evidence appears.
MILK_CURVE_SOURCE_LACTATION = {l: min(l, 3) for l in range(1, MAX_LACTATION + 1)}
MILK_CURVE_BASIS = {
    # Wood: a, time-to-peak and peak are printed (Evangelista 2024) but b and c are derived here, so the basis is 'derived', not 'source'
    "wood": {l: ("derived" if l <= 3 else "repeated") for l in range(1, MAX_LACTATION + 1)},
    "milkbot": {l: ("source" if l <= 3 else "repeated") for l in range(1, MAX_LACTATION + 1)},   # Ehrlich 2011 Table 1 (Holstein median, parities 1-3)
}

# Wood (1967) lactation curve, selectable as an alternative to MilkBot.
# Parameter source: Evangelista et al. (2024), Pesq. Agropec. Bras. 59,
# DOI 10.1590/S1678-3921.pab2024.v59.03366, Table 4: Holstein cows, Parana (Brazil),
# 2010-2019, n = 48,758 / 40,125 / 24,298 (lactations 1 / 2 / 3). a (kg), time to
# peak (days) and peak yield (kg) are the paper's own numbers; b and c are not
# printed in the paper and are derived from them (see _wood_bc). The herd is
# non-US, which is a modelling assumption when applied to other herds.
_WOOD_SOURCE = {
    1: {"a": 15.13, "t_peak_days": 96.0, "peak_kg": 32.58},
    2: {"a": 20.97, "t_peak_days": 63.0, "peak_kg": 40.82},
    3: {"a": 21.37, "t_peak_days": 60.0, "peak_kg": 43.36},
}


def _wood_bc(a, t_peak_days, peak_kg):
    """Derive Wood's b and c from a, time-to-peak (= b/c) and peak yield
    (= a * (b/c)**b * exp(-b)):  b = ln(peak/a) / ln(t_peak/e),  c = b / t_peak."""
    import math
    b = math.log(peak_kg / a) / math.log(t_peak_days / math.e)
    return b, b / t_peak_days


def _wood_parameters_by_parity():
    out = {}
    for k, s in _WOOD_SOURCE.items():
        b, c = _wood_bc(s["a"], s["t_peak_days"], s["peak_kg"])
        out[k] = {"a": s["a"], "b": b, "c": c}
    return out


WOOD1967 = {
    "citation_source": (
        "Wood, P.D.P. (1967). \"Algebraic Model of the Lactation Curve in "
        "Cattle.\" Nature 216:164-165. DOI: 10.1038/216164a0."
    ),
    "published": {
        "formula": "MP_dim = a * t^b * exp(-c*t), where t = days in milk (DIM, real days)",
    },
    "mathematical_conversion": (
        "None for the equation (evaluated at real days). b and c are derived "
        "from the parameter source's a, time-to-peak and peak yield -- see _wood_bc."
    ),
    "modeling_assumptions": (
        "Parameters come from a Brazilian (Parana) Holstein dataset "
        "(Evangelista et al. 2024, Table 4) and are used here as a generic "
        "Holstein curve (lactation buckets 1, 2, 3+). b and c are derived by "
        "this project, not published. Evidence level: page fetch/summary, not "
        "a direct PDF read."
    ),
    "default": True,
    "user_configurable_parameters": "_WOOD_SOURCE (above) / parameters_by_parity",
    "parameters_by_parity": _wood_parameters_by_parity(),
    "calibration_required": False,
    "confidence_status": (
        "Equation form: Type 1, literature-supported. Parameters: a, time to "
        "peak, peak yield literature-sourced (non-US Holstein, fetch-summary "
        "evidence level); b, c derived by this project (modelling assumption)."
    ),
}

# ----------------------------------------------------------
# PREGNANCY MILK DECLINE
# ----------------------------------------------------------
# Milk yield falls linearly as a pregnancy advances, reaching this fraction of
# the open-cow curve value at dry-off (dp == DRY_OFF_DP), and is 0 once dry.
# Open cows (dp == 0) see no decline. Used by cow/tables.py.
PREGNANCY_DECLINE_FLOOR = 0.70




# ----------------------------------------------------------
# MORTALITY -- ANNUAL-RATE ANCHOR CALIBRATION (cow/probabilities/mortality.py)
# ----------------------------------------------------------
# Mirrors ABORTION_ALBAAJ2022 above: separates what the sources publish from what the model
# does with it. Used by mortality.py::p_mortality_anchor_calibrated (the 'formula' method; the
# default 'table' method reads the Pinedo et al. (2010) hazards instead). The mortality_table
# (mortality.py) supplies the relative shape by lactation, DIM and status; only its overall level
# is rescaled to match a corroborated annual rate. Rebuilding the shape itself is not done (see
# 'rejected_alternative').
MORTALITY_ANNUAL_ANCHOR = {
    # Two independent sources, converging on about 5.7 % annual mortality.
    "citation_sources": [
        (
            "Miller RH, Kuhn MT, Norman HD, Wright JR (2008). \"Death "
            "Losses for Lactating Cows in Herds Enrolled in Dairy Herd "
            "Improvement Test Plans.\" Journal of Dairy Science "
            "91(9):3710-3715. DOI: 10.3168/jds.2007-0943. National DHI "
            "dataset, 15,025,035 lactations in 45,032 US herds, "
            "1995-2005: overall death frequency 3.1% per lactation, 5.7% "
            "per cow."
        ),
        (
            "Thomsen PT (2023). \"High risk of dairy cow mortality in "
            "early lactation.\" Veterinary Record 193(5):e3210. DOI: "
            "10.1002/vetr.3210. Cites Danish national statistics: annual "
            "cow mortality 5.4% (2020) and 5.6% (2021)."
        ),
    ],
    # Published value used as the calibration target.
    "anchor_value": 0.057,
    # Calibration method (a modelling choice, not published): the 15
    # (lactation x DIM x status) monthly rates of mortality_table are annualized
    # (1-(1-p)^12), given equal weight (there is no herd occupancy distribution
    # to weight them by), and averaged. A single multiplicative scale factor k is
    # then solved numerically so that the equal-weighted mean of the scaled
    # table's annualized rates equals anchor_value exactly. k is applied
    # uniformly to every cell, so the table's relative shape (by
    # lactation/DIM/status) is preserved and only its overall level moves.
    "calibration_method": (
        "k solves mean_i(1-(1-k*p_i)^12) == anchor_value, over the 15 "
        "equal-weighted mortality_table cells (p_i = original monthly "
        "rate); computed once at import in "
        "cow/probabilities/mortality.py::MORTALITY_CALIBRATION_SCALE."
    ),
    # Modelling assumptions (not published, not literature-derived).
    "modeling_assumptions": (
        "(1) Equal weighting across the 15 table cells stands in for the "
        "true herd occupancy distribution, for which there is no "
        "source. It is a simplification, not a fitted or "
        "population-representative weighting. (2) The table's relative "
        "shape (how lactation, DIM and status change risk relative to "
        "each other) is that of mortality_table; "
        "only the absolute level is literature-anchored."
    ),
    # Alternative that was considered and not adopted.
    "rejected_alternative": (
        "Rebuilding the DIM and parity hazard shape from the death-timing "
        "distribution of Thomsen (2023) is not done. Thomsen reports only "
        "cumulative death timing among cows that died (no at-risk "
        "denominator), at percentile points (7, 18, 30, 90 and 93 d) that "
        "do not align with this table's bucket edges (60/150/300 d), and "
        "DIM shape and parity cannot be separated (higher-parity cows die "
        "earlier). A formula would need a curve interpolated through "
        "sparse points, death-timing density used as an uncorrected hazard "
        "proxy, and an assumed linear parity spread from the two extremes "
        "reported by Miller (2008) (1st vs 8th+ parity, 2 percentage points "
        "apart)."
    ),
    # This is the 'formula' method of mortality_spec. The exponential-decay
    # formula (mortality.py::p_mortality_formula) is not wired
    # into the spec but can still be called directly.
    "default": True,
    "user_configurable_parameters": "anchor_value",
    "calibration_required": False,
    "confidence_status": (
        "Anchor value: literature-supported and empirical, corroborated by "
        "two independent sources (a US DHI-record analysis and Danish "
        "national statistics). Equal weighting and shape preservation: "
        "modelling approximations."
    ),
}


# ----------------------------------------------------------
# BODY WEIGHT (simplified Holstein-like)
# ----------------------------------------------------------
BW_BIRTH_KG = 40.0
BW_MATURE_KG = 680.0
BW_GROWTH_RATE_PER_DAY = 0.0032
BW_MAX_POSTPARTUM_LOSS_KG = 45.0
BW_DAYS_TO_MIN_WEIGHT_DAYS = 30.0
BW_PREGNANCY_GAIN_KG = 40.0
AGE_AT_FIRST_CALVING_DAYS = 730
AVERAGE_CALVING_INTERVAL_DAYS = 400


# ----------------------------------------------------------
# MANAGEMENT POLICY -- CULLING (modular, competing-risk)
# ----------------------------------------------------------
# Culling probability is built from independent "modules". Each module
# contributes its own probability that a cow is culled THIS PERIOD; they
# combine as competing risks:
#
#   p_cull = 1 - (1 - p_age_stage) * (1 - p_repro)
#
# so any single module at 1.0 guarantees culling, a module at 0.0 has no
# effect, and two moderate risks don't simply add past 1.0. Turn a module
# off entirely with the ENABLE_CULL_* switches below -- when off it's
# excluded from the product, not just zeroed.

# Which modules are active
ENABLE_CULL_AGE_STAGE = True   # lactation / DIM / status lookup or formula

# Age/stage module: annual-rate anchor calibration (culling.py). Mirrors
# MORTALITY_ANNUAL_ANCHOR above. Used by culling.py::p_cull_age_stage_anchor_calibrated
# (the 'formula' method of culling_age_stage_spec; the default 'table' method reads the
# Pinedo et al. (2010) hazards instead). The culling_table (culling.py) supplies the
# relative shape by lactation, DIM and status; only its overall level is rescaled.
#
# This module represents the residual share of removals, not a biological age hazard:
#   (1) Disposal-code systems have no "age/stage" category. Pinedo et al. (2010) found that
#       culling risk "varied by parity, stage of lactation... with all reported disposal
#       codes", so age and DIM modulate the hazard of every other cause; they are not a
#       cause themselves.
#   (2) The 29.2 %/yr figure is the full population total across all mutually exclusive
#       disposal-code reasons (Lunak 2024, Penn State Extension: the reason percentages
#       "are mutually exclusive reasons -- each cow appears once in a single category"
#       and sum to the observed total). Anchoring this module to the full total while
#       other culling mechanisms are added on top would count the total twice.
# The module therefore covers the disposal-code categories that no other mechanism of the
# model represents: sold for replacement, temperament, and other voluntary and involuntary
# reasons. The module keeps the name culling_age_stage for continuity. The other culling
# mechanism of the model is the milk-floor reproductive cull (culling_repro).
CULLING_AGE_STAGE_ANNUAL_ANCHOR = {
    # Sources for the full observed total (about 29 %/yr), kept for reference. This is not
    # the calibration target; see 'anchor_value' below.
    "citation_sources": [
        (
            "Pinedo PJ, De Vries A, Webb DW (2010). \"Dynamics of "
            "culling risk with disposal codes reported by Dairy Herd "
            "Improvement dairy herds.\" Journal of Dairy Science "
            "93(5):2250-2261. DOI: 10.3168/jds.2009-2572. National DHI "
            "dataset, 3,629,002 lactations in 2,054 US herds, 2001-2006: "
            "overall annualized culling rate 31.7% (25.1% live culling + "
            "6.6% death culling)."
        ),
        (
            "USDA:APHIS:VS NAHMS Dairy 2018 report (Northeastern US), as "
            "quoted by Lunak M (2024), Penn State Extension, \"Cull "
            "Rates: How is Your Farm Doing?\": total removal rate 37.6% "
            "annually = 31.4% culling (live) + 6.2% deaths."
        ),
        (
            "Pennsylvania DRMS data, 2,570 herds, as quoted by the same "
            "Penn State Extension article: average cull rate 37.2% "
            "annually including deaths (~31.0% live, using the same "
            "~6.2% death share as the NAHMS figure above, since DRMS's "
            "own live/death split was not reported separately in the "
            "source article)."
        ),
    ],
    # Source for the residual apportionment (the calibration target). Only the Lunak (2024) /
    # Penn State Extension figures are used, not an average over the sources above, because
    # that article gives the full mutually exclusive reason breakdown, consistent with the
    # NAHMS Dairy 2018 31.4 % figure. Multiplying the shares of one study (for example NAHMS Dairy
    # 2007: 26.3 % reproductive, 23 % mastitis, 16 % production, 16 % lameness, 3.7 % other
    # disease, 8 % other) by the total of a different study (about 23.6 % of cows left
    # permanently in 2007) is not a valid apportionment.
    "residual_apportionment_source": (
        "Lunak M (2024), Penn State Extension, \"Cull Rates: How is Your "
        "Farm Doing?\". Of the 31.4% NAHMS "
        "Dairy 2018 live-culling rate, reasons are reported as mutually "
        "exclusive categories (each cow counted once): Poor production "
        "18.3%, Infertility 23.3%, Mastitis 18.6%, Lameness 9.1%, "
        "Injuries 3.5%, Respiratory 2.4%, Metritis 2.2%, Displaced "
        "abomasum 2.0%, Sold for replacement 4.6%, Aggressive temper "
        "0.7%, Other (voluntary) 3.2%, Other (involuntary) 12.1%; these sum "
        "to 100% of the 31.4%. The residual bucket assigned to "
        "culling_age_stage is Sold for replacement + Aggressive "
        "temper + Other (voluntary) + Other (involuntary) = "
        "4.6+0.7+3.2+12.1 = 20.6% of 31.4% = 6.47%/yr. "
        "Poor production (18.3%), Infertility (23.3%) and Mastitis (18.6%) "
        "are not part of this residual. Lameness (9.1%) + Injuries (3.5%) + "
        "Respiratory (2.4%) + Metritis (2.2%) + Displaced abomasum (2.0%) "
        "= 19.2% of 31.4% = 6.03%/yr is not represented by any culling "
        "mechanism of the model (see modeling_assumptions, item 4)."
    ),
    # Plain mean of the three full-total sources above (kept for reference; not the
    # calibration target, see 'anchor_value' below).
    "historical_full_total_anchor_value": 0.29166666666666663,
    # Calibration target: the residual share only (see 'residual_apportionment_source'):
    # 20.6% x 31.4% = 6.47%/yr. This represents "unattributed/other" removals only, not
    # every cause combined.
    "anchor_value": 0.0647,
    # Calibration method (identical to MORTALITY_ANNUAL_ANCHOR): the 15
    # (lactation x DIM x status) monthly rates of culling_table are annualized
    # (1-(1-p)^12), given equal weight, and averaged. A single scale factor k is
    # solved numerically so the scaled table's equal-weighted mean annual rate
    # equals anchor_value (0.0647) exactly, applied uniformly to every cell.
    "calibration_method": (
        "k solves mean_i(1-(1-k*p_i)^12) == anchor_value, over the 15 "
        "equal-weighted culling_table cells (p_i = original monthly "
        "rate); computed once at import in "
        "cow/probabilities/culling.py::CULLING_AGE_STAGE_CALIBRATION_SCALE."
    ),
    # Modelling assumptions (not published).
    "modeling_assumptions": (
        "(1) Equal weighting across the 15 table cells, the same "
        "simplification as MORTALITY_ANNUAL_ANCHOR. (2) culling_age_stage "
        "represents a residual, unattributed-cause share, not a "
        "biological age/stage hazard. Its shape (how lactation, DIM and "
        "status change relative risk) is that of culling_table, which was not "
        "derived for 'residual other reasons' specifically; only its level is "
        "residual-targeted. (3) Converting a crude annual cause-specific "
        "proportion (share x total, both crude population-level "
        "proportions from the source survey) into a per-period hazard "
        "with the same 1-(1-p)^(1/12)-style identity used elsewhere in "
        "this package is a standard but approximate conversion: it "
        "assumes competing-cause censoring does not materially distort "
        "the crude cause-specific count, which is reasonable at these "
        "magnitudes but not verified. (4) The unmapped ~6.03%/yr "
        "lameness/injury/respiratory/metritis/displaced-abomasum share is "
        "not folded into this residual, so the combined "
        "culling probability is expected to sit below the full observed "
        "rate by roughly that amount."
    ),
    "default": True,
    "user_configurable_parameters": "anchor_value",
    "calibration_required": False,
    "confidence_status": (
        "Full-total reference value: literature-supported and empirical, "
        "corroborated by three independent sources; it is not the active "
        "target. Active residual anchor_value (0.0647): the underlying "
        "31.4%/18.3%/18.6%/20.6% figures are from Lunak (2024), Penn State "
        "Extension, itself quoting NAHMS Dairy 2018 and Pennsylvania DRMS "
        "data. The crude-proportion conversion, the exclusion of the "
        "unmapped disease categories, and the residual-category grouping "
        "are modelling choices; the source does not define the residual "
        "as a single category."
    ),
}


# ----------------------------------------------------------
# ENVIRONMENTAL REWARDS -- ENABLE FLAGS
# ----------------------------------------------------------
# All default off: turning any of these on is an explicit opt-in and never
# silently changes the existing reward values (see cow/rewards/environment/).
# There is deliberately no ENABLE_REWARD_CO2: respiratory CO2 is tracked only,
# not priced by default (see cow/rewards/environment/co2.py for why).
ENABLE_REWARD_NITROGEN = False
ENABLE_REWARD_ENTERIC_METHANE = False
ENABLE_REWARD_MANURE_METHANE = False
ENABLE_REWARD_N2O = False

# ----------------------------------------------------------
# ENVIRONMENTAL -- DIET COMPOSITION ASSUMPTIONS
# ----------------------------------------------------------
# The model tracks milk yield (kg) but not composition, so there is no
# observed state for these. They are Holstein-typical herd-average default
# assumptions, not measured or derived values. Override them if you have real
# herd data.
DIET_FAT_PCT = 3.8
DIET_PROTEIN_PCT = 3.1
DIET_CP_PCT = 15.5          # % of DM
DIET_ME_MCAL_PER_KG = 3.00
DIET_DE_PCT = 73.0          # Digestible Energy, % of Gross Energy
DIET_ASH_PCT = 8.0

# ----------------------------------------------------------
# ENVIRONMENTAL -- MANURE MANAGEMENT
# ----------------------------------------------------------
# Deliberately no default: IPCC (2019) MCF and EF values depend on climate and
# management system (Table 10.17), so a single hardcoded default would not be
# defensible for all users. Manure CH4 and all N2O pathways are skipped, not
# silently defaulted, until both are set.
MANURE_SYSTEM = os.environ.get("COW_MANURE_SYSTEM")        # e.g. "solid_storage"
MANURE_CLIMATE = os.environ.get("COW_MANURE_CLIMATE")      # "cool" | "temperate" | "warm"
# (The leaching fraction is per manure system, not a single scalar; see
# cow/rewards/environment/n2o.py::FRAC_LEACH_TABLE, from IPCC (2019) Table 10.22.)

# ----------------------------------------------------------
# ENVIRONMENTAL -- IPCC TIER 2 ENTERIC METHANE
# ----------------------------------------------------------
IPCC_TIER2_YM_PERCENT = 6.0  # IPCC (2019) Table 10.12, "High-producing dairy cows (>8500 kg/yr),
                              # NDF>35% DM" row. The model tracks neither diet NDF nor an
                              # annual-yield tier, so this single value is used for all cows (see
                              # cow/rewards/environment/enteric_methane.py for the full table).

# ----------------------------------------------------------
# ENVIRONMENTAL -- REWARD PRICING
# ----------------------------------------------------------
# Economic/policy assumptions (carbon-price-style), NOT biological
# constants. Default to 0.0 -- no arbitrary normalization constant is
# assumed on your behalf; set intentionally for the corresponding
# reward (once enabled above) to have any effect.
PRICE_PER_KG_N_MANURE = 0.0
PRICE_PER_KG_CH4 = 0.0
PRICE_PER_KG_N2O = 0.0

# ----------------------------------------------------------
# NOTE: Economic parameters (milk price, feed cost, etc.)
# are in cow/rewards.py because they are reward-specific.
# Discount factor and evaluation horizon are in mrp_engine.ipynb
# (user settings cell) because they are MRP evaluation parameters.


# ----------------------------------------------------------------------
# CONCEPTION TABLE -- WHAT IT REPRESENTS. Documentation only: nothing here feeds a
# calculation.
# ----------------------------------------------------------------------
CONCEPTION_INTERPRETATION_2026_09_19 = {
    "question": (
        "Is the cow conception probability (0.45 lac1 / 0.35 lac2+ per month, DIM 51-350) an "
        "insemination-to-conception (pregnancy ESTABLISHMENT) rate, a CONFIRMED-pregnancy rate like the "
        "heifer rate (CDCB conception rate), or something else?"
    ),
    "source_definition": (
        "No source defines it. The only documented support is an aggregate check against "
        "CDCB days open (see CONCEPTION_RATE_PROVENANCE). CDCB's own "
        "conception rate (0.36 all breedings, 0.38 first breedings, Holstein 2019-2020; reproall.html) "
        "is a per-service confirmed-pregnancy measure, and it is not what this table was derived from."
    ),
    "correct_metric_and_result": (
        "Days from calving to the conception that reaches calving = (dim - dp + 1) * UNIT_DAYS at calving events of a "
        "lactation-1 cohort censored at replacement. The model gives 142.0 d (MONTH) and 144.2 d (WEEK) against the "
        "CDCB-derived ~137 d (83 d to first breeding + 55.6 d first service to conception)."
    ),
    "interpretation_adopted": (
        "Pregnancy establishment (conception) per month for an open cow past the voluntary waiting period, paired with "
        "the Albaaj (2022) loss windows from gestation day 19 (cumulative ~45% to calving). It is not equivalent to the "
        "heifer parameter, which is a confirmed-pregnancy rate applied with no extra early loss."
    ),
    "alternative_rejected": (
        "Treating the table as a confirmed rate and applying loss only after a diagnosis day (60 d, assumption) gives days "
        "open 107.7 d (MONTH) / 108.7 d (WEEK), ~29 d below the benchmark, and would require lowering the table or adding "
        "the 83-day first-breeding delay to recover it. That would be tuning to the benchmark, so it is not adopted."
    ),
    "recalibration": "Not needed: the existing pairing reproduces the aggregate.",
    "remaining_uncertainty": (
        "Two structurally different decompositions (establishment rate + heavy early loss, vs confirmed rate + first-"
        "breeding delay + small late loss) can produce the same days open, so the table's definition is under-determined by "
        "the aggregate alone. Residuals: MONTH/WEEK days open 142.0/144.2 vs ~137 (+4%); calving interval 422/424 vs CDCB 397 "
        "(+6%; CDCB intervals exist only for cows that calve again, the model cohort is censored differently)."
    ),
}


# ============================================================
# PARITY / GESTATION-LENGTH EVIDENCE -- every number is a documented transformation of printed
# values (see cow/evidence/ for the source tables). Lactations beyond the last key repeat the last
# key's value (the sources group them).
# ============================================================
# Odds of establishing a pregnancy relative to lactation 1 (two concordant studies):
#   Cook & Green (2016) J Dairy Sci 99:4699 Tables 1-2, pregnant by 100 / 150 DIM: parity 2 = 0.67 / 0.68, parity >=3 = 0.53 / 0.47;
#   Cordeiro et al. (2026) Theriogenology 265:118076 Table 2, pregnancy at day 28: parity 2 = 0.69, parity >=3 = 0.46.
# (Remnant et al. 2018 find a much smaller parity effect, OR 0.97 / 0.94 / 0.92 / 0.75 for parity 2/3/4/5+, per insemination
# and conditional on the inter-service interval -- a different quantity.)
# The lactation-1 level is the CDCB-consistent value; only the relative odds come from these studies.
CONCEPTION_PARITY_ODDS_RATIO = {1: 1.00, 2: 0.68, 3: 0.47}

# Relative risk of pregnancy loss by lactation (published ratios plus a normalisation assumption), for the 'model' method of
# the abortion spec only (the default pools parities, like its Albaaj / Lee & Kim sources). Evidence for an increase
# with parity: Fernandez-Novo 2020 (28-110 d: heifers 6.9 %, primiparous 10.8 %, multiparous 15.3 %), Cordeiro 2026
# (days 28-50 loss 18.4 / 23.5 / 27.6 % for parity 1 / 2 / >=3), Lee & Kim 2007 (OR 0.6 / 0.5 for parity 1 / 2 vs >=3);
# against it: Corva 2025 (HR 1.03 / 1.00 for parity 2 / >=3, 22,152 pregnancies) and Lopez-Gatius 2002 (lactation number
# not significant). Raw ratios vs parity >=3 (mean of the three supporting studies): 0.66, 0.78, 1.00; divided by the
# mean 0.81 for a 38/28/34 % lactation 1 / 2 / >=3 mix so that the pooled hazard is preserved.
ABORTION_PARITY_RELATIVE_RISK = {1: 0.81, 2: 0.96, 3: 1.23}

# Gestation length spread: Mwangi et al. (2025) J Dairy Sci 108:8685, 9,750 gestations of 4,256 dairy cows,
# SD 5.87 d (raw 5.89 d), 92.5 % of gestations 270-290 d; Holstein-Friesian-sired calves 277.3 d, parity 2..12 means
# 278.4-279.8 d. The MEAN stays GESTATION_LENGTH_DAYS; the SD is used only by the 'model' method of the calving spec
# (the default 'table' keeps the deterministic trigger).
GESTATION_LENGTH_SD_DAYS = 5.9

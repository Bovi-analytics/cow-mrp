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
MAX_DIM_DAYS = int(os.environ.get("COW_MAX_DIM_DAYS", 1000))
MAX_DP_DAYS = int(os.environ.get("COW_MAX_DP_DAYS", 300))

# Holstein-sired gestation length. UPDATED 2026-09-23, primary-verified:
#   Basiel BL, Barragan AA, Felix TL, Dechow CD (2024). "The impact of beef
#   sire breed on dystocia, stillbirth, gestation length, health, and
#   lactation performance of cows that carry beef x dairy calves." J Dairy
#   Sci 107(4):2241-2252. DOI: 10.3168/jds.2023-24112.
#   Abstract (fetched and read directly, user-supplied PDF): "...Holstein-
#   sired calves (277 +/- 0.15 d)..." -- LSM +/- SE from a mixed model,
#   75,256 lactations, 39,249 cows, 10 US herds (Northeast/Midwest),
#   2010-2023. Holstein-sired is this repo's assumed dam x sire population
#   (the model's default/conventional breeding program). Supersedes the
#   prior value (280, from a 1932 study never independently fetched, only
#   known via secondary summaries -- see docs/REFERENCES.md's `Knott1932`
#   history for that superseded citation).
# CONFIDENCE STATUS: verified, primary source. Used directly by cow/probabilities/
# calving.py's active 'table' method (2026-08-27 resolution-consistency
# fix) and its 'formula' alternative's own midpoint -- both now reference
# this single value, resolving a prior, unreconciled 270-vs-280 mismatch
# between the two methods (see docs/MODEL_CHANGELOG.md). Value updated
# 280->277 2026-09-23 (see above); every downstream reference (dry-off
# timing, body-weight pregnancy gain, youngstock heifer-pregnancy hazard)
# reads this single constant, so nothing else needed a matching edit.
GESTATION_LENGTH_DAYS = 277
# Standard dry-period length. Widely reported as a 42-60 day recommended
# range with 55 days cited as the traditional standard practice (e.g. dry
# period comparisons in the dairy extension/veterinary literature); a
# specific single primary study pinning "55" exactly was not identified
# this pass -- retained as an existing, standard-practice value, not a
# newly-asserted one. See docs/BIOLOGICAL_ASSUMPTIONS.md.
DRY_OFF_DAYS_BEFORE_CALVING = 55
DRY_OFF_DP_DAYS = GESTATION_LENGTH_DAYS - DRY_OFF_DAYS_BEFORE_CALVING

# Convert to time units
MAX_DIM = to_units(MAX_DIM_DAYS)
MAX_DP = to_units(MAX_DP_DAYS)
DRY_OFF_DP = first_step_past(DRY_OFF_DP_DAYS)   # mid-step rule (was to_units: dry-off 1 step early at MONTH)

# ----------------------------------------------------------
# FARMER REPRODUCTIVE-PROGRAM SETTINGS (CowMRP fork)
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
# shortened the MONTH cycle by ~7% and removed ~0.3 calvings per cycle, see docs/COW_MODEL_AUDIT.md experiment E8).
# True: a cow that loses her pregnancy keeps the remaining half step of conception exposure (she may conceive again in
# the same step: P = 1 - sqrt(1 - q), q = that step's conception probability). The DEFINITION of re-eligibility is
# unchanged; only its dependence on TIME_UNIT is removed (the effect vanishes as the step shrinks). False: previous behaviour.
RESUME_BREEDING_AFTER_LOSS = True


# ----------------------------------------------------------
# REPRODUCTION -- CONCEPTION / PREGNANCY RATE (cow/probabilities/conception.py)
# ----------------------------------------------------------
# Provenance record for conception_table's rate values (0.45/0.45/0.45/0.35
# lactation 1; 0.35/0.35/0.35/0.25 lactation 2+, cow/probabilities/
# conception.py). Unlike ABORTION_ALBAAJ2022/MORTALITY_ANNUAL_ANCHOR below,
# this dict does NOT feed an active formula -- conception_table's values are
# UNCHANGED by this record; it exists purely to document a 2026-08-27
# audit's conclusion that these previously-uncited values should be
# RETAINED, not recalibrated or reverted, and why.
CONCEPTION_RATE_PROVENANCE = {
    "history": (
        "These exact values (0.45/0.45/0.45/0.35 lac1, 0.35/0.35/0.35/0.25 "
        "lac2+) were introduced in commit efb2dc2 ('Culling architecture: "
        "literature-apportioned cause-specific calibration', 2026-08-25) as "
        "an UNEXPLAINED change bundled into an unrelated culling-calibration "
        "commit -- that commit's own message discloses it as one of several "
        "'pre-existing working-tree changes', with no rationale given "
        "anywhere for the conception-rate edit specifically. The PRIOR "
        "values (this repo's original, 2026-08-10 commit 72ae1c2) were "
        "0.22/0.17/0.05/0.02 (lac1) and 0.17/0.14/0.04/0.015 (lac2+), a "
        "steeply DIM-declining shape -- also uncited, and not linked to any "
        "source, prior model, or documented calibration either."
    ),
    # Citations actually fetched and read directly (not secondary-summary
    # snippets) during the 2026-08-27 conception-provenance audit.
    "citation_sources": [
        (
            "CDCB (Council on Dairy Cattle Breeding) national DHI "
            "reproductive database, queries.uscdcb.com/publish/dhi/current/"
            "reproall.html (fetched and read directly): Holstein, 2018-2020 "
            "-- days calving-to-first-breeding 80-83 days; conception rate "
            "for first breedings 0.38, for all breedings 0.35-0.36 "
            "(PER-SERVICE metrics, national, current)."
        ),
        (
            "CDCB, First Service to Conception (FSC) trait definition, "
            "uscdcb.com/first-service-to-conception (fetched and read "
            "directly): mean 55.60 days, Holstein -- days from first "
            "breeding to conception, a PER-LACTATION CUMULATIVE metric."
        ),
        (
            "Fodor A, et al. (2019). \"Pregnancy and conception rates by "
            "parity.\" Canadian Journal of Veterinary Research 83(3):"
            "218-227. 23,781 cows, 34 large Hungarian Holstein-Friesian "
            "herds (fetched and read directly): first-service conception "
            "risk 24.8% (primiparous, n=9,346) vs 17.3% (multiparous, "
            "n=10,448), P<0.001; pregnant-by-200-DIM 65.2% vs 55.4%."
        ),
    ],
    # Our aggregate-outcome calibration check -- an EXPLICIT modeling
    # judgment, not a direct source-to-parameter mapping. CDCB's own two
    # official metrics compose into a national days-open benchmark with no
    # unit-conversion ambiguity: 80-83 (days to first breeding) + 55.6
    # (FSC) = ~136-139 days, current US Holstein national average.
    "aggregate_benchmark_check": (
        "Running this model (cow/probabilities/conception.py's day-exact "
        "resolution-consistency integration) with the CURRENT "
        "conception_table values gives an expected days-open (Open+DNB "
        "sojourn, first-lactation cohort, censored at replacement) of "
        "~138-140 days at both MONTH and WEEK resolution -- closely "
        "matching the CDCB-derived national average of ~136-139 days. "
        "Re-running the identical integration with the ORIGINAL "
        "(pre-efb2dc2) declining-DIM values instead gives ~266 days -- "
        "roughly double any benchmark gathered, and the LESS defensible "
        "of the two despite its shape looking superficially more "
        "biologically plausible before being quantified."
    ),
    # EXPLICITLY NOT a claim that CDCB, Fodor et al., or any other source
    # published these exact per-lactation-per-DIM-bucket MONTHLY values --
    # they did not, and no such source was found. See "modeling_
    # assumptions" below for exactly what is, and is not, being asserted.
    "modeling_assumptions": (
        "(1) A single monthly probability, re-applied uniformly to every "
        "eligible Open/DNB cow regardless of how many services she has "
        "already had, stands in for the real distinction between "
        "'awaiting first service' and 'awaiting conception after "
        "service' -- both of CDCB's own metrics measure the LATTER only "
        "(FSC) or a per-service snapshot (conception rate), and this "
        "model has no state variable tracking service count or whether a "
        "cow has been bred yet. This is a genuine, disclosed structural "
        "simplification, not an oversight: only the AGGREGATE (total "
        "days open) is checked against the AGGREGATE benchmark above, "
        "never the internal mechanism against CDCB's internal mechanism. "
        "(2) The lac1-vs-lac2+ split (0.45 vs 0.35, the one lactation "
        "distinction this model implements) is checked only for "
        "DIRECTION against Fodor et al.'s parity gap (24.8% vs 17.3% "
        "first-service conception risk -- a similar-direction, not "
        "identical-magnitude, effect, since first-service risk and a "
        "recurring monthly probability are different metrics). (3) The "
        "late-DIM decline to 0.35/0.25 (lac1/lac2+) beyond day 350 has no "
        "specific literature source identified this pass; retained as an "
        "existing, directionally-plausible modeling choice (worsening "
        "prospects for cows still open very late), not verified against "
        "a dedicated 'conception rate vs. days already open' study."
    ),
    # Decision -- RETAIN, documented, NOT recalibrated to a specific
    # source value, and NOT reverted to the original (worse-performing,
    # per the aggregate check above) values. See
    # docs/MODEL_CHANGELOG.md and docs/BIOLOGICAL_ASSUMPTIONS.md.
    "decision": "retained_and_documented",
    "calibration_required": False,
    "user_configurable_parameters": "cow/probabilities/conception.py::conception_table",
    "confidence_status": (
        "Aggregate days-open outcome: Type 3 (empirical), cross-checked "
        "against a current, official, national-scale database (CDCB), "
        "fetched and read directly, via an explicit AGGREGATE-outcome "
        "calibration check (not a direct per-value citation) -- the same "
        "category of validation this repo's mortality/culling annual-rate "
        "anchors already use (an aggregate target matched by an "
        "originally-uncited table's overall level, internal shape not "
        "independently verified). Per-lactation split and late-DIM "
        "decline shape: Type 4, modeling assumption, directionally "
        "supported by Fodor et al. (2019) but not value-matched."
    ),
}


# ----------------------------------------------------------
# REPRODUCTION -- ABORTION / PREGNANCY LOSS (cow/probabilities/abortion.py)
# ----------------------------------------------------------
# Structure deliberately separates what the source publishes from what we do
# with it -- see each key's own comment. Used by
# cow/probabilities/abortion.py::p_abortion_albaaj2022 (the active default
# 'formula' method). The prior uncited table (cow/probabilities/abortion.py::
# abortion_table) is untouched and still selectable via method='table'.
ABORTION_ALBAAJ2022 = {
    # Citation/source -- fetched and read directly this project (both the
    # PMC full text and its own methods-section wording, not a secondary
    # summary).
    "citation_source": (
        "Albaaj A, Durocher J, LeBlanc SJ, Dufour S (2022). \"Meta-analysis "
        "of the incidence of pregnancy losses in dairy cows at different "
        "stages to 90 days of gestation.\" JDS Communications 4(2):144-148. "
        "DOI: 10.3168/jdsc.2022-0278."
    ),
    # Published values/formula -- VERBATIM, never modified. Loss rate is
    # conditional on being pregnant at the start of the window (not
    # cumulative from conception). The paper's own words on the overlap:
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
    # Our mathematical conversion -- an exact identity between equivalent
    # hazard representations, not a biological claim: each window's
    # (loss_rate, D-day span) becomes a per-day hazard via
    # h = 1-(1-loss_rate)^(1/D); a per-model-step probability then follows
    # via p_step = 1-(1-h)^UNIT_DAYS -- the same identity
    # cow/probabilities/mortality.py::p_mortality_formula already uses.
    "mathematical_conversion": (
        "h_daily = 1 - (1 - loss_rate) ** (1 / D); "
        "p_step = 1 - (1 - h_daily) ** UNIT_DAYS"
    ),
    # Our modeling assumptions/adaptations -- explicitly OURS, not the
    # source's. Never describe these as published or literature-derived.
    "modeling_assumptions": (
        "(1) Risk is assumed locally UNIFORM across the days within a "
        "single published window -- the paper reports only a window-total "
        "probability, not a within-window shape; this is our simplification. "
        "(2) For any day covered by more than one published window (the "
        "acknowledged 19-32/30-45 overlap on days 30-32, plus the single-"
        "day shared boundaries at day 45 and day 60), we take the "
        "UNWEIGHTED AVERAGE of the covering windows' per-day hazards, "
        "because the source gives no basis to prefer one over the other. "
        "This averaging is OUR modeling adaptation -- NOT published, NOT "
        "literature-derived."
    ),
    # Outside the paper's studied range (dp_days < 19 or > 90): the prior
    # uncited default table's rates are retained UNCHANGED, not replaced or
    # extrapolated -- these are not supported by Albaaj et al. (2022) or
    # any other source found this project.
    "outside_published_range": {
        "note": (
            "dp_days < 19 or > 90 -- legacy uncited monthly rates retained "
            "unchanged (not derived from this or any other source)."
        ),
        "legacy_monthly_rates": {
            "1-18d": 0.040, "91-150d": 0.015, "151-270d": 0.005, "271+d": 0.000,
        },
        # Days 91-270 -- REPLACED 2026-10-08 (audit P3) by Lee & Kim (2007), J Vet Sci 8(3):283-288, Table 2
        # (1,001 Holstein pregnancies, 7 Korean herds, ultrasound/palpation diagnosis at 28-60 d): loss of
        # 34/988 pregnancies (3.44 %) in the second trimester (days 91-180) and 22/954 (2.31 %) in the third
        # (days 181-260). Evidence class B: the printed proportions, each spread uniformly over its window.
        # The third-trimester value is repeated to day 270 (source window ends at 260). Days 1-18 and 271+ keep
        # the legacy rates. Cross-checks: De Vries 2006 months 4-8 sum 2.5 %, Fernandez-Novo 2020 12.3 % for
        # days 28-110, Corva 2025 22.5 % for days 28-210 in one grazing herd -- a wide range; this lies between.
        "late_windows": [
            {"days": (91, 180), "loss_rate": 34 / 988, "span_days": 90, "source": "Lee & Kim 2007 second trimester"},
            {"days": (181, 270), "loss_rate": 22 / 954, "span_days": 80, "source": "Lee & Kim 2007 third trimester (window 181-260, repeated to 270)"},
        ],
    },
    # Default -- this becomes the active 'formula' method for abortion_spec,
    # per this project's "most confident usable option is the default" rule
    # (the published rates are peer-reviewed and primary-source-verified;
    # the prior table has zero citation at all).
    "default": True,
    # User-configurable parameters -- edit the 'windows' list or the
    # outside-range fallback rates directly above; same override convention
    # as every other table/constant in this file.
    "user_configurable_parameters": "published['windows'], outside_published_range['legacy_monthly_rates']",
    # Calibration required -- No. The published rates are usable as-is;
    # only the overlap-averaging and outside-range fallback are documented
    # modeling choices, not missing data.
    "calibration_required": False,
    # Confidence/status
    "confidence_status": (
        "Published windows/rates: Type 1, literature-supported, primary "
        "source fetched and read directly this project (see "
        "docs/REFERENCES.md). Mathematical conversion: an exact identity, "
        "not a separate confidence tier. Modeling assumptions (uniform-"
        "within-window, overlap-averaging): Type 4, modeling approximation "
        "-- explicitly ours, not the source's."
    ),
}


# ----------------------------------------------------------
# MILK PARAMETERS
# ----------------------------------------------------------
# MilkBot lactation curve parameters. Citation corrected: Ehrlich (2010),
# Bovine Practitioner 45(1):88-95 (was mis-cited as 2011). Values below
# are UNVERIFIED against that paper's own benchmark table (inaccessible
# this pass) -- a related dataset (Ehrlich 2013, PeerJ 1:e54) reports
# substantially different magnitudes (scale ~39-54 kg, ramp ~26-31 days)
# for a different study population, which may or may not be comparable.
# See docs/BIOLOGICAL_ASSUMPTIONS.md and docs/LIMITATIONS_REGISTER.md L-011.
# MilkBot parameters (audit P6, 2026-10-08): REPLACED the unverified placeholders (peak 28/34/38 kg, ramp 10 d,
# offset -10 d, decay 0.0015/d). Source (class A, printed values, read directly): Ehrlich JL (2011) "Quantifying shape of
# lactation curves, and benchmark curves for common dairy breeds and parities", The Bovine Practitioner 45(1):88-95,
# Table 1, HOLSTEIN MEDIAN rows (scale kg/day, ramp d, offset d, decay /d): a DHIA data set of 6,459,942 lactations from
# 17,013 herds, mostly eastern US, calving 2005-mid 2008, DIM 6-400, parities 1, 2, 3 (third lactation) only;
# lactations 4-10 repeat parity 3 (MILK_CURVE_BASIS 'repeated'). Equation (Ehrlich 2013 PeerJ 1:e54 Eq. 5 and Ehrlich 2011
# Appendix 1): Y(t) = a * (1 - exp((c - t)/b) / 2) * exp(-d t). Table 1 also gives the fitted peak day (88/56/58 d) and
# M305 (9,296/10,507/10,837 kg) used as checks in verify/audit_milk.py.
MILKBOT_EHRLICH2011 = {
    1: {"a": 38.1, "b": 35.2, "c": -3.8, "d": 0.00109},
    2: {"a": 49.5, "b": 27.2, "c": -3.8, "d": 0.00215},
    3: {"a": 53.6, "b": 30.1, "c": -2.4, "d": 0.00243},
}

# Which milk-yield formula cow/states.py::milk_potential_formula dispatches to.
# "milkbot" (default, unchanged) | "wood" -- see WOOD1967 below. Selecting
# "wood" before WOOD1967['parameters_by_parity'] is calibrated raises a clear
# ValueError -- it never silently falls back to MilkBot or guesses a value.
MILK_CURVE_METHOD = os.environ.get("COW_MILK_CURVE_METHOD", "wood")

# LACTATION-EXPLICIT milk curves. The model represents lactations 1..MAX_LACTATION explicitly. Each lactation
# uses the parameters of a SOURCE lactation, and the evidence status of that choice is recorded here, never
# implied: 'source' = the cited source reports exactly this lactation; 'repeated' = the source does not
# distinguish this lactation (it reports lactations 1, 2 and 3 only), so lactation 3's parameters are repeated
# -- NO evidence of a difference either way; 'placeholder' = uncited modelling value.
# Replace an entry's source (or add real per-lactation parameters to WOOD1967 / MILKBOT_PEAK_KG) as evidence appears.
MILK_CURVE_SOURCE_LACTATION = {l: min(l, 3) for l in range(1, MAX_LACTATION + 1)}
MILK_CURVE_BASIS = {
    # Wood: a, time-to-peak and peak are printed (Evangelista 2024) but b and c are DERIVED by this project -> 'derived', not 'source'
    "wood": {l: ("derived" if l <= 3 else "repeated") for l in range(1, MAX_LACTATION + 1)},
    "milkbot": {l: ("source" if l <= 3 else "repeated") for l in range(1, MAX_LACTATION + 1)},   # Ehrlich 2011 Table 1 (Holstein median, parities 1-3)
}

# Wood (1967) lactation curve -- added as a selectable alternative to MilkBot,
# not a replacement. See docs/BIOLOGICAL_ASSUMPTIONS.md and
# docs/ASSUMPTIONS_REGISTER.md for the full verification trail. Structure
# below deliberately separates what the source actually says from what we do
# with it -- see each key's own comment.
# Source inputs for the Wood parameters: Evangelista et al. (2024), Pesq.
# Agropec. Bras. 59, DOI 10.1590/S1678-3921.pab2024.v59.03366, Table 4 --
# Holstein cows, Parana (Brazil), 2010-2019, n = 48,758 / 40,125 / 24,298
# (lactations 1 / 2 / 3). Read via a page fetch that summarizes the page
# (NOT the PDF). a (kg), time to peak (days) and peak yield (kg) are the
# paper's own numbers; b and c are NOT printed in the paper and are DERIVED
# here from them (see _wood_bc). Modelling-assumption flag: non-US herd.
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
# Open cows (dp == 0) see no decline. Used by cow/tables.py. (Existing value;
# previously applied only to an expected-yield ratio, now applied to actual milk
# -- see docs/MODEL_CHANGELOG.md 2026-10-08.)
PREGNANCY_DECLINE_FLOOR = 0.70




# ----------------------------------------------------------
# MORTALITY -- ANNUAL-RATE ANCHOR CALIBRATION (cow/probabilities/mortality.py)
# ----------------------------------------------------------
# Structure mirrors ABORTION_ALBAAJ2022 above -- separates what the sources
# publish from what we do with it. Used by
# cow/probabilities/mortality.py::p_mortality_anchor_calibrated (the new
# active default 'formula' method). The prior uncited table
# (cow/probabilities/mortality.py::mortality_table) is untouched -- its
# relative SHAPE (by lactation/DIM/status) is NOT replaced, only its overall
# LEVEL is rescaled to match a well-corroborated real-world annual rate. A
# fuller alternative (rebuilding the DIM/parity shape itself from Thomsen
# (2023)'s death-timing data) was considered and explicitly rejected as
# requiring too many additional invented assumptions -- see this dict's own
# 'rejected_alternative' key and docs/LIMITATIONS_REGISTER.md L-014.
MORTALITY_ANNUAL_ANCHOR = {
    # Citation/source -- two remaining independent sources, converging on
    # ~5.7% annual mortality. (A third corroborating source's citation was
    # removed 2026-09-23 -- unverified, not independently checked; the
    # 5.7%/4.8%/3.8% (2007/2002/1996) figures it reported are not restated
    # here pending re-sourcing.) Neither of the two remaining full primary
    # texts could be fetched and read directly this pass (both
    # blocked/paywalled -- see 'confidence_status' below); the value is
    # corroborated across independent secondary summaries of each, not a
    # single unverified guess.
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
    # Our calibration method -- NOT published, an explicit modeling choice:
    # the existing uncited mortality_table's 15 (lactation x DIM x status)
    # monthly rates are annualized (1-(1-p)^12), given EQUAL weight (we
    # have no real herd occupancy distribution to weight them by -- an
    # explicit simplification, not a claim of population-
    # representativeness), and averaged. A single multiplicative scale
    # factor k is then solved numerically so that the equal-weighted mean
    # of the SCALED table's annualized rates equals anchor_value exactly.
    # k is applied uniformly to every cell, so the table's relative shape
    # (by lactation/DIM/status) is preserved exactly -- only its overall
    # level moves.
    "calibration_method": (
        "k solves mean_i(1-(1-k*p_i)^12) == anchor_value, over the 15 "
        "equal-weighted mortality_table cells (p_i = original monthly "
        "rate); computed once at import in "
        "cow/probabilities/mortality.py::MORTALITY_CALIBRATION_SCALE."
    ),
    # Our modeling assumptions/adaptations -- explicitly OURS, not the
    # sources'. Never describe these as published or literature-derived.
    "modeling_assumptions": (
        "(1) Equal weighting across the 15 table cells stands in for the "
        "true herd occupancy distribution, which this project has no "
        "source for -- an explicit simplification, not a fitted or "
        "population-representative weighting. (2) The table's original "
        "RELATIVE shape (how lactation/DIM/status change risk relative to "
        "each other) remains exactly the prior uncited table's shape -- "
        "only the absolute LEVEL is now literature-anchored. This is a "
        "deliberately smaller, more defensible correction than rebuilding "
        "the shape itself would have been (see 'rejected_alternative')."
    ),
    # The fuller alternative considered and explicitly rejected this pass.
    "rejected_alternative": (
        "Rebuilding the DIM/parity hazard SHAPE from Thomsen (2023)'s "
        "death-timing distribution (fetched and read directly, open "
        "access) was considered, but Thomsen reports only cumulative "
        "death-timing among cows that died (no at-risk denominator -- an "
        "explicit limitation of that paper's own design), at percentile "
        "points (7d/18d/30d/90d/93d) that don't align with this table's "
        "bucket edges (60/150/300d), and shows DIM-shape and parity are "
        "NOT separable (higher-parity cows die earlier). A workable "
        "formula would have required interpolating a curve through "
        "sparse points, using death-timing density as an uncorrected "
        "hazard proxy, and an invented linear parity-spread from Miller "
        "(2008)'s two reported extremes (1st vs 8th+ parity, 2 "
        "percentage points apart) -- materially more invented "
        "assumption-stacking than ABORTION_ALBAAJ2022 needed. Rejected "
        "by explicit user decision; see docs/LIMITATIONS_REGISTER.md "
        "L-014."
    ),
    # Default -- this becomes the active 'formula' method for
    # mortality_spec, replacing the prior uncited exponential-decay
    # formula (cow/probabilities/mortality.py::p_mortality_formula, kept
    # unchanged, no longer wired into the spec, still callable directly
    # for reference).
    "default": True,
    "user_configurable_parameters": "anchor_value",
    "calibration_required": False,
    "confidence_status": (
        "Anchor value: Type 1/3 (literature-supported/empirical), "
        "corroborated across 3 independent sources (US NAHMS survey, US "
        "DHI-record analysis, Danish national statistics) but NONE of "
        "the 3 full primary texts could be fetched and read directly "
        "this pass (NAHMS PDF: repeated connection resets; Miller et al. "
        "2008: blocked/paywalled across ScienceDirect, "
        "journalofdairyscience.org, ResearchGate, Academia.edu, and "
        "CABI; Thomsen 2023's Danish-statistics citation is background "
        "context within a paper that WAS fetched and read directly) -- "
        "each anchor number is only verified via independent secondary "
        "summaries, not primary-text confirmation (see "
        "docs/REFERENCES.md). Equal-weighting/shape-preservation: Type "
        "4, modeling approximation -- explicitly ours."
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

# Age/stage module -- annual-rate anchor calibration
# (cow/probabilities/culling.py). Structure mirrors MORTALITY_ANNUAL_ANCHOR
# above. Used by cow/probabilities/culling.py::p_cull_age_stage_anchor_
# calibrated (the active default 'formula' method for
# culling_age_stage_spec). The prior uncited table
# (cow/probabilities/culling.py::culling_table) is untouched -- its
# relative SHAPE (by lactation/DIM/status) is NOT replaced, only its
# overall LEVEL is rescaled.
#
# RE-SCOPED this pass (was: anchored to the FULL observed live-culling
# rate, 29.2%/yr -- see git history / docs/MODEL_CHANGELOG.md for the
# prior formulation). A dedicated audit (see docs/BIOLOGICAL_ASSUMPTIONS.md,
# docs/LIMITATIONS_REGISTER.md L-015, L-020) established from the source
# data itself that:
#   (1) real disposal-code systems have NO "age/stage" category at all --
#       Pinedo et al. (2010) found culling risk "varied by parity, stage
#       of lactation... with all reported disposal codes," i.e. age/DIM
#       modulate every OTHER cause's hazard, they are not their own cause;
#   (2) the 29.2%/yr figure previously used here is the FULL population
#       total across ALL mutually-exclusive disposal-code reasons
#       (confirmed directly: Lunak (2024)/Penn State Extension states the
#       reason percentages "are mutually exclusive reasons -- each cow
#       appears once in a single category," summing to the observed
#       total) -- so anchoring culling_age_stage to the FULL total while
#       ALSO multiplying culling_health/culling_milk_policy/culling_dnb on
#       top of it double-books the total (see L-020).
# culling_age_stage is therefore re-scoped as the RESIDUAL/unattributed
# share of removals -- the disposal-code categories that do not map to
# any of this model's other 3 modules (sold-for-replacement, temperament,
# and other unclassified voluntary/involuntary reasons). It is NOT a
# biological "age hazard" despite the module's name (kept unchanged to
# avoid an unrelated rename -- see docs/LIMITATIONS_REGISTER.md L-020).
CULLING_AGE_STAGE_ANNUAL_ANCHOR = {
    # Citation/source, ORIGINAL 3-source total (kept for provenance/
    # history -- this is what the anchor USED to target; it is NOT the
    # calibration target any more, see 'anchor_value' below). None of the
    # 3 full primary texts/reports could be fetched and read directly
    # (JDS/ScienceDirect blocked/paywalled, same wall as Miller (2008) for
    # mortality); the PSU Extension article itself WAS fetched and read
    # directly and is the source for 2 of these 3 numbers.
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
    # Citation/source for the RESIDUAL apportionment (this pass's actual
    # calibration target). Deliberately uses ONLY the Lunak (2024)/Penn
    # State Extension figure (not a 3-source average like above) because
    # it is the one source in the list above for which the FULL
    # mutually-exclusive reason breakdown could be fetched and read
    # directly, and that breakdown is internally consistent with (derived
    # from the same NAHMS Dairy 2018 31.4% figure already used as one of
    # the 3 sources above) -- mixing a different study's total with this
    # breakdown's shares (e.g. the NAHMS Dairy 2007 reason breakdown,
    # 26.3% reproductive/23% mastitis/16% production/16% lameness/3.7%
    # other-disease/8% other) was explicitly rejected because that
    # breakdown comes from a DIFFERENT survey year with its OWN, lower
    # total (~23.6% of cows left permanently in 2007) -- multiplying one
    # study's shares by a different study's total is not a valid
    # apportionment. See docs/BIOLOGICAL_ASSUMPTIONS.md.
    "residual_apportionment_source": (
        "Lunak M (2024), Penn State Extension, \"Cull Rates: How is Your "
        "Farm Doing?\" (fetched and read directly). Of the 31.4% NAHMS "
        "Dairy 2018 live-culling rate, reasons are reported as mutually "
        "exclusive categories (each cow counted once): Poor production "
        "18.3%, Infertility 23.3%, Mastitis 18.6%, Lameness 9.1%, "
        "Injuries 3.5%, Respiratory 2.4%, Metritis 2.2%, Displaced "
        "abomasum 2.0%, Sold for replacement 4.6%, Aggressive temper "
        "0.7%, Other (voluntary) 3.2%, Other (involuntary) 12.1% -- sums "
        "to 100% of the 31.4%. The RESIDUAL bucket assigned to "
        "culling_age_stage here is: Sold for replacement + Aggressive "
        "temper + Other (voluntary) + Other (involuntary) = "
        "4.6+0.7+3.2+12.1 = 20.6% of 31.4% = 6.47%/yr. Poor production "
        "(18.3%) maps to culling_milk_policy and Mastitis (18.6%) maps "
        "to culling_health -- see CULLING_MILK_ANNUAL_ANCHOR and "
        "CULLING_HEALTH_ANNUAL_ANCHOR below. Infertility (23.3%) is "
        "deliberately NOT assigned to culling_dnb -- see "
        "docs/LIMITATIONS_REGISTER.md L-020 for why (culling_dnb is a "
        "structural/deterministic mechanism tied to the model's own "
        "conception/DNB-flag dynamics, not an independently-calibratable "
        "rate; forcing a literature share onto it would double-count an "
        "outcome the model already generates endogenously). Lameness "
        "(9.1%) + Injuries (3.5%) + Respiratory (2.4%) + Metritis (2.2%) "
        "+ Displaced abomasum (2.0%) = 19.2% of 31.4% = 6.03%/yr is a "
        "genuine COVERAGE GAP -- no module in this model represents "
        "these specific physical/metabolic disease categories (folding "
        "them into culling_age_stage's residual, or into culling_health, "
        "was explicitly rejected -- see 'coverage_gap' key below)."
    ),
    # Published value used as the calibration target -- the plain mean of
    # the 3 sources above, KEPT for provenance/history only. This is NOT
    # the active calibration target any more (see 'anchor_value' below).
    "historical_full_total_anchor_value": 0.29166666666666663,
    # ACTIVE calibration target -- the RESIDUAL share only (see
    # 'residual_apportionment_source' above): 20.6% x 31.4% = 6.47%/yr.
    # This is a smaller, more defensible target than the full population
    # total: it represents "unattributed/other" removals only, not every
    # cause combined.
    "anchor_value": 0.0647,
    # Our calibration method -- UNCHANGED mechanism from before (identical
    # to MORTALITY_ANNUAL_ANCHOR): the existing uncited culling_table's 15
    # (lactation x DIM x status) monthly rates are annualized
    # (1-(1-p)^12), given EQUAL weight, and averaged. A single scale
    # factor k is solved numerically so the scaled table's equal-weighted
    # mean annual rate equals anchor_value (now 0.0647, not 0.2917)
    # exactly, applied uniformly to every cell. Only the TARGET changed
    # this pass -- the solver/mechanism in cow/probabilities/culling.py is
    # untouched.
    "calibration_method": (
        "k solves mean_i(1-(1-k*p_i)^12) == anchor_value, over the 15 "
        "equal-weighted culling_table cells (p_i = original monthly "
        "rate); computed once at import in "
        "cow/probabilities/culling.py::CULLING_AGE_STAGE_CALIBRATION_SCALE."
    ),
    # Our modeling assumptions/adaptations -- explicitly OURS.
    "modeling_assumptions": (
        "(1) Equal weighting across the 15 table cells, same "
        "simplification as MORTALITY_ANNUAL_ANCHOR. (2) culling_age_stage "
        "now represents a RESIDUAL/unattributed-cause share, not a "
        "biological age/stage hazard -- its SHAPE (how lactation/DIM/"
        "status change relative risk) is still the original uncited "
        "table's shape, which was never derived to represent 'residual "
        "other reasons' specifically; only its LEVEL is now residual-"
        "targeted. (3) Converting a crude annual cause-specific "
        "proportion (share x total, both crude population-level "
        "proportions from the source survey) into a per-period hazard "
        "via the same 1-(1-p)^(1/12)-style identity used elsewhere in "
        "this repo is a standard but APPROXIMATE conversion -- it "
        "assumes competing-cause censoring does not materially distort "
        "the crude cause-specific count, which is a reasonable but "
        "unverified assumption at these magnitudes (Type 4, modeling "
        "approximation, not an exact identity). (4) The unmapped ~6.03%/"
        "yr lameness/injury/respiratory/metritis/DA share is DELIBERATELY "
        "NOT folded into this residual -- the model's total composed "
        "p_cull is therefore EXPECTED to sit below the full empirical "
        "benchmark by roughly that amount. This is an intentional, "
        "documented coverage gap, not an error to be closed by inflating "
        "this or any other anchor -- see docs/LIMITATIONS_REGISTER.md "
        "L-020."
    ),
    "default": True,
    "user_configurable_parameters": "anchor_value",
    "calibration_required": False,
    "confidence_status": (
        "Historical full-total anchor: Type 1/3 (literature-supported/"
        "empirical), corroborated across 3 independent sources, none of "
        "the full primary texts fetched directly (see docs/REFERENCES.md)"
        " -- kept for provenance only, no longer the active target. "
        "ACTIVE residual anchor_value (0.0647): Type 1/3 for the "
        "underlying 31.4%/18.3%/18.6%/20.6% figures themselves (Lunak "
        "2024/Penn State Extension, fetched and read directly, itself "
        "quoting NAHMS Dairy 2018 and Pennsylvania DRMS data) combined "
        "with Type 4 (modeling approximation) for (a) the crude-"
        "proportion, (b) the choice to exclude the unmapped disease "
        "categories rather than fold them in, and (c) the residual-"
        "category grouping itself, which is our own apportionment "
        "choice, not something the source itself defines as a single "
        "category."
    ),
}


# ----------------------------------------------------------
# ENVIRONMENTAL REWARDS -- ENABLE FLAGS
# ----------------------------------------------------------
# All default OFF. This preserves reward_total/benchmark parity exactly
# as it was before this feature existed -- turning any of these on is an
# explicit opt-in, never a silent change to existing reward values (see
# cow/rewards/environment/). There is deliberately no
# ENABLE_REWARD_CO2 -- respiratory CO2 is tracked only, not priced by
# default (see cow/rewards/environment/co2.py for why).
ENABLE_REWARD_NITROGEN = False
ENABLE_REWARD_ENTERIC_METHANE = False
ENABLE_REWARD_MANURE_METHANE = False
ENABLE_REWARD_N2O = False

# ----------------------------------------------------------
# ENVIRONMENTAL -- DIET COMPOSITION ASSUMPTIONS
# ----------------------------------------------------------
# This repo tracks milk YIELD (kg) but not composition -- there is no
# observed state or existing pipeline value for these. Holstein-typical
# herd-average defaults (scientific-audit-flagged assumptions, not
# measured or derived values). Override via config if you have real
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
# Deliberately NO default: IPCC (2019) MCF/EF values are climate- and
# management-system-dependent (Table 10.17) -- a single hardcoded
# default would not be scientifically defensible for all users (see the
# scientific audit). Manure CH4 and all N2O pathways are SKIPPED, not
# silently defaulted, until both are set.
MANURE_SYSTEM = os.environ.get("COW_MANURE_SYSTEM")        # e.g. "solid_storage"
MANURE_CLIMATE = os.environ.get("COW_MANURE_CLIMATE")      # "cool" | "temperate" | "warm"
# (Leaching fraction is per-manure-system, not a single scalar -- see
# cow/rewards/environment/n2o.py::FRAC_LEACH_TABLE, verified against the
# primary IPCC (2019) Table 10.22. A prior single MANURE_FRAC_LEACH=0.30
# constant here was found, during verification, not to match any real
# per-system value and was removed.)

# ----------------------------------------------------------
# ENVIRONMENTAL -- IPCC TIER 2 ENTERIC METHANE
# ----------------------------------------------------------
IPCC_TIER2_YM_PERCENT = 6.0  # VERIFIED exact: IPCC (2019) Table 10.12, "High-producing dairy
                              # cows (>8500 kg/yr), NDF>35% DM" row specifically (this repo
                              # does not track diet NDF or an annual-yield production tier, so
                              # this single value is used for all cows -- see
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
# CONCEPTION TABLE -- WHAT IT REPRESENTS (2026-09-19). DOCUMENTATION ONLY: no
# value above or in cow/probabilities/conception.py::conception_table changed.
# ----------------------------------------------------------------------
CONCEPTION_INTERPRETATION_2026_09_19 = {
    "question": (
        "Is the cow conception probability (0.45 lac1 / 0.35 lac2+ per month, DIM 51-350) an "
        "insemination-to-conception (pregnancy ESTABLISHMENT) rate, a CONFIRMED-pregnancy rate like the "
        "heifer rate (CDCB conception rate), or something else?"
    ),
    "source_definition": (
        "There is none. The values were introduced without rationale (see CONCEPTION_RATE_PROVENANCE "
        "'history'). The only documented support is an AGGREGATE check against CDCB days open. CDCB's own "
        "conception rate (0.36 all breedings, 0.38 first breedings, Holstein 2019-2020; reproall.html, read) "
        "is a per-service confirmed-pregnancy measure, and it is NOT what this table was derived from."
    ),
    "invalid_earlier_check": (
        "The 2026-08-27 calibration check (A-033) compared 'Open+DNB sojourn' (~138-140 d) with CDCB's 136-139 d. "
        "Open+DNB sojourn omits the time a cow spends in a pregnancy that is later lost, so it is not days open "
        "(calving to the conception that reaches the next calving). The check compared unlike quantities."
    ),
    "correct_metric_and_result": (
        "Days from calving to the conception that reaches calving = (dim - dp + 1) * UNIT_DAYS at calving events of a "
        "lactation-1 cohort censored at replacement. With NO parameter changed: legacy timing 140.3 d (MONTH); after "
        "the abortion fix alone 156.4 d (calibration broken); after the abortion + DIM/conception/DNB/calving timing "
        "fixes 142.0 d (MONTH) and 144.2 d (WEEK) against the CDCB-derived ~137 d (83 d to first breeding + 55.6 d "
        "first service to conception). The legacy agreement came from two half-step errors of opposite sign cancelling."
    ),
    "interpretation_adopted": (
        "PREGNANCY ESTABLISHMENT (conception) per month for an Open cow past the voluntary waiting period, paired with "
        "the Albaaj (2022) loss windows from gestation day 19 (cumulative ~45% to calving). It is NOT equivalent to the "
        "heifer parameter, which is a confirmed-pregnancy rate applied with no extra early loss."
    ),
    "alternative_rejected": (
        "Treating the table as a CONFIRMED rate and applying loss only after a diagnosis day (60 d, assumption) gives days "
        "open 107.7 d (MONTH) / 108.7 d (WEEK), ~29 d below the benchmark, and would require lowering the table or adding "
        "the 83-day first-breeding delay to recover it. Not adopted, because that would be tuning."
    ),
    "recalibration": "Not required and not performed: after the timing corrections the existing pairing reproduces the aggregate.",
    "remaining_uncertainty": (
        "Two structurally different decompositions (establishment rate + heavy early loss, vs confirmed rate + first-"
        "breeding delay + small late loss) can produce the same days open, so the table's definition is under-determined by "
        "the aggregate alone. Residuals: MONTH/WEEK days open 142.0/144.2 vs ~137 (+4%); calving interval 422/424 vs CDCB 397 "
        "(+6%; CDCB intervals exist only for cows that calve again, the model cohort is censored differently). See "
        "docs/LIMITATIONS_REGISTER.md L-030."
    ),
}


# ============================================================
# PARITY / GESTATION-LENGTH EVIDENCE (audit P3, 2026-10-08) -- every number is a documented transformation of
# printed values; see docs/COW_MODEL_AUDIT.md and cow/evidence/ for the source tables. Lactations beyond the last
# key repeat the last key's value (the sources group them).
# ============================================================
# Odds of establishing a pregnancy relative to lactation 1 (class B, two concordant studies):
#   Cook & Green (2016) J Dairy Sci 99:4699 Tables 1-2, pregnant by 100 / 150 DIM: parity 2 = 0.67 / 0.68, parity >=3 = 0.53 / 0.47;
#   Cordeiro et al. (2026) Theriogenology 265:118076 Table 2, pregnancy at day 28: parity 2 = 0.69, parity >=3 = 0.46.
# (Remnant et al. 2018 find a much smaller parity effect, OR 0.97 / 0.94 / 0.92 / 0.75 for parity 2/3/4/5+, per insemination
# and conditional on the inter-service interval -- a different quantity; recorded in docs/COW_MODEL_AUDIT.md.)
# The lactation-1 LEVEL remains the existing CDCB-consistent value (A-033); only the relative odds are evidence.
CONCEPTION_PARITY_ODDS_RATIO = {1: 1.00, 2: 0.68, 3: 0.47}

# Relative risk of pregnancy loss by lactation (class B + normalisation assumption), for the 'model' method of
# the abortion spec only (the default pools parities, like its Albaaj / Lee & Kim sources). Evidence for an increase
# with parity: Fernandez-Novo 2020 (28-110 d: heifers 6.9 %, primiparous 10.8 %, multiparous 15.3 %), Cordeiro 2026
# (days 28-50 loss 18.4 / 23.5 / 27.6 % for parity 1 / 2 / >=3), Lee & Kim 2007 (OR 0.6 / 0.5 for parity 1 / 2 vs >=3);
# against it: Corva 2025 (HR 1.03 / 1.00 for parity 2 / >=3, 22,152 pregnancies) and Lopez-Gatius 2002 (lactation number
# not significant). Raw ratios vs parity >=3 (mean of the three supporting studies): 0.66, 0.78, 1.00; divided by the
# mean 0.81 for a 38/28/34 % lactation 1 / 2 / >=3 mix so that the pooled hazard is preserved.
ABORTION_PARITY_RELATIVE_RISK = {1: 0.81, 2: 0.96, 3: 1.23}

# Gestation length spread (class A): Mwangi et al. (2025) J Dairy Sci 108:8685, 9,750 gestations of 4,256 dairy cows,
# SD 5.87 d (raw 5.89 d), 92.5 % of gestations 270-290 d; Holstein-Friesian-sired calves 277.3 d, parity 2..12 means
# 278.4-279.8 d. The MEAN stays GESTATION_LENGTH_DAYS; the SD is used only by the 'model' method of the calving spec
# (the default 'table' keeps the deterministic trigger).
GESTATION_LENGTH_SD_DAYS = 5.9

"""
youngstock/parameters.py
========================
Biological + structural parameters for the youngstock (calf -> heifer ->
first calving) population module. Sibling of cow/parameters.py.

TIME: this module does NOT define its own time unit. TIME_UNIT / UNIT_DAYS /
GESTATION_LENGTH_DAYS are imported from domains.cow/parameters.py -- the ONE source
of truth. Every duration below is stored in real calendar
DAYS and converted to model steps by the two helpers right below, using the
SAME "which step's real-day span contains this day" rule already used by
cow/probabilities/calving.py (conception-time-averaged calving_step_mass).

SOURCE METADATA: every parameter is created through register(), which stores
its source, evidence tier and placeholder flag in PARAMETER_SOURCES right
beside the value. EVIDENCE TIERS:
  read-full      primary paper/report text read and the number taken from it
  read-abstract  abstract (or report summary table) read and the number taken from it
  derived        computed here from read/secondary anchors (derivation stated)
  secondary      seen only through a search summary / citing text
  assumption     a modeling or management choice, NOT a measured biological value
  placeholder    a stand-in because no usable source was found -- never biology
Nothing tagged 'placeholder' or 'assumption' may be described as literature.
This module is "parameterized from literature", NOT "biologically exact".

PARITY: dam parity at the calving that produces the calf. '1' = first calving
(heifer), '2' = second calving (cow in lactation 1), '3+' = later calvings.
Where a source distinguishes only first vs later calvings, parities '2' and
'3+' carry the SAME value on purpose and the entry says so -- no parity-
specific number is invented.
"""

import numpy as np

from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS, GESTATION_LENGTH_DAYS

PARAMETER_SOURCES = {}


def register(name, value, *, source, evidence, placeholder=False, note=""):
    """Store value + metadata; return the value unchanged."""
    PARAMETER_SOURCES[name] = dict(source=source, evidence=evidence, placeholder=bool(placeholder), note=note)
    return value


# ============================================================
# TIME HELPERS (resolution-aware; the only place days -> steps happens)
# ============================================================

def step_containing_day(day):
    """Index k of the model step whose real-day span [k*UNIT_DAYS,
    (k+1)*UNIT_DAYS) contains `day`. Same rule as calving.py's day-exact
    trigger. Used for events that happen ON a day (e.g. sale)."""
    return int(np.floor(day / UNIT_DAYS))


def first_step_at_or_after_day(day):
    """Smallest k with k*UNIT_DAYS >= day. Used for 'from this age onward'
    thresholds (e.g. first breeding)."""
    return int(np.ceil(day / UNIT_DAYS))


# ============================================================
# USER SELECTIONS (scenario inputs -- not biology)
# ============================================================
PARITY_BUCKETS = ("1", "2", "3+")
SEMEN_TYPES = ("conventional", "sexed", "beef")

# Breeding program: share of CALVINGS by semen type, per dam parity (the semen
# used ~9 months earlier on that dam). See BREEDING_PROGRAMS below.
# CHOOSE: 'conventional' | 'us_dhi_2008' | 'sexed_dairy' | 'beef_on_dairy' | 'custom' (uses BREEDING_MIX below)
BREEDING_PROGRAM = "conventional"

# Fraction of LIVE FEMALE DAIRY calves kept as replacement candidates; the rest
# are sold at birth. 1.0 = keep every eligible female (unconstrained
# production). An INPUT/SCENARIO value -- not optimized, not biological.
# CHOOSE: 0.0 - 1.0
FEMALE_DAIRY_RETENTION = 1.0


# ============================================================
# BREEDING PROGRAMS
# ============================================================
# 'conventional'  -- all conventional dairy semen. Scenario switch.
# 'us_dhi_2008'   -- Norman, Hutchison & Miller (2010) J Dairy Sci 93(8):3880-3890
#                    [read-abstract]: sexed semen was 17.8% of 2008 heifer
#                    breedings and 0.4% of cow breedings (US DHI). Used as a
#                    proxy for the share of CALVINGS (sexed semen conceives
#                    less, so its share of calvings is slightly lower).
#                    The same abstract says 63.1% of the cow sexed-semen use was
#                    on FIRST-PARITY cows, but no parity-specific breeding
#                    counts were read, so that concentration CANNOT be turned
#                    into a parity-2 share; parities 2 and 3+ carry the cow value.
# 'sexed_dairy'   -- all sexed dairy semen. Scenario switch.
# 'beef_on_dairy' -- heifers conventional dairy, all cows beef semen. Scenario
#                    switch (real farms mix; the split is a management choice).
_US2008_COW = {"conventional": 0.996, "sexed": 0.004}
BREEDING_PROGRAMS = register("BREEDING_PROGRAMS", {
    "conventional": {"1": {"conventional": 1.0}, "2": {"conventional": 1.0}, "3+": {"conventional": 1.0}},
    "us_dhi_2008": {"1": {"conventional": 0.822, "sexed": 0.178}, "2": dict(_US2008_COW), "3+": dict(_US2008_COW)},
    "sexed_dairy": {"1": {"sexed": 1.0}, "2": {"sexed": 1.0}, "3+": {"sexed": 1.0}},
    "beef_on_dairy": {"1": {"conventional": 1.0}, "2": {"beef": 1.0}, "3+": {"beef": 1.0}},
}, source="Norman et al. 2010 (us_dhi_2008 only); other programs are scenario switches",
   evidence="assumption", note="us_dhi_2008 = read-abstract shares of BREEDINGS used as calving proxy; the rest are scenario definitions, not literature")

# FARMER INPUT: the share of CALVINGS by semen type, one mix for heifers (dam parity '1') and one for cows
# (parities '2' and '3+'). Each must sum to 1. Select it with BREEDING_PROGRAM = "custom". Defaults are 100%
# conventional (the same as the 'conventional' program) because no national share of beef-on-dairy matings is
# sourced here: NAAB reports sexed semen at ~61% of 2024 US dairy-semen UNITS (Hoard's Dairy Man, 2025; secondary,
# units sold, not calvings; sexed semen conceives less), but not the beef-on-dairy share of dairy breedings.
BREEDING_MIX = {
    "heifer": {"conventional": 1.0, "sexed": 0.0, "beef": 0.0},
    "cow":    {"conventional": 1.0, "sexed": 0.0, "beef": 0.0},
}


def _custom_program():
    for who, mix in BREEDING_MIX.items():
        unknown = set(mix) - set(SEMEN_TYPES)
        if unknown or any(v < 0 for v in mix.values()) or abs(sum(mix.values()) - 1.0) > 1e-9:
            raise ValueError(f"BREEDING_MIX[{who!r}] must use {SEMEN_TYPES}, be non-negative and sum to 1; got {mix}")
    heifer = {s: v for s, v in BREEDING_MIX["heifer"].items() if v > 0}
    cow = {s: v for s, v in BREEDING_MIX["cow"].items() if v > 0}
    return {"1": heifer, "2": dict(cow), "3+": dict(cow)}


BREEDING_PROGRAMS["custom"] = _custom_program()


# ============================================================
# BIRTH OUTCOME PARAMETERS  (per calf / per calving, by dam parity)
# ============================================================
# Fraction of live calves that are FEMALE, by semen type (parity-independent:
# Mendes 2022, Transl Anim Sci 6(3), reports only odds ratios by lactation --
# ~1.08 at lactation 1 falling to ~0.82 at lactation 9 -- so no parity-specific
# FRACTION can be built from it).
FEMALE_FRACTION = register("FEMALE_FRACTION", {"conventional": 0.467, "sexed": 0.90, "beef": 0.50},
    source="conventional: Silva del Rio et al. 2007 J Dairy Sci 90(3):1255-1264 (2.3 M US Holstein calvings, singleton 46.7% F). "
           "sexed: Norman et al. 2010 (~90% F); corroborated by Bittante et al. 2020 J Dairy Sci 103(6):5246-5257 (>=90%). beef: none",
    evidence="read-abstract",
    note="beef=0.50 is a PLACEHOLDER: beef-cross calves are terminal in both sexes, so it only affects reporting, never the dairy flow")
FEMALE_FRACTION_PLACEHOLDER_KEYS = ("beef",)

# Stillbirth probability per calf, by (dam parity, semen type).
#  parity '1' conventional/sexed: Norman 2010 heifers 10.4% / 11.3% [read-abstract]
#     (cross-check Meyer et al. 2001 J Dairy Sci 84(2):515-523: expected 10% primiparous).
#  parities '2','3+' conventional/sexed: Norman 2010 COWS 3.6% / 2.7% -- the source
#     reports heifers vs cows only, so both parities carry the cow value.
#  beef, parities '2','3+': Basiel et al. 2024 J Dairy Sci 107(4):2241-2252 [read-abstract]
#     beef-sired 5% (95% CI 2.9-9.0) vs Holstein-sired 2% (CI 1.5-2.7), 75,256 lactations,
#     10 herds, parity NOT stratified. Used as the ABSOLUTE 5% (their own Holstein
#     baseline, 2%, is below Norman's 3.6% for cows, so the ratio 2.5x would give ~9%,
#     the upper CI edge); the choice is disclosed, evidence moderate.
#  beef, parity '1': NO source read -> PLACEHOLDER equal to the conventional heifer value.
STILLBIRTH = register("STILLBIRTH", {
    ("1", "conventional"): 0.104, ("1", "sexed"): 0.113, ("1", "beef"): 0.104,
    ("2", "conventional"): 0.036, ("2", "sexed"): 0.027, ("2", "beef"): 0.05,
    ("3+", "conventional"): 0.036, ("3+", "sexed"): 0.027, ("3+", "beef"): 0.05,
}, source="Norman et al. 2010; Meyer et al. 2001; Basiel et al. 2024", evidence="read-abstract",
   note="parities 2 and 3+ share one value (sources split only heifer vs cow); ('1','beef') is a PLACEHOLDER")
STILLBIRTH_PLACEHOLDER_KEYS = (("1", "beef"),)

# Twin-vs-singleton stillbirth RATIO (not an absolute rate), applied
# MULTIPLICATIVELY to STILLBIRTH[parity, semen] above: STILLBIRTH[p,s] * STILLBIRTH_TWIN_RATIO[p],
# capped at 1.0. This is the only defensible way to combine two independently-sourced dimensions
# neither paper reports jointly: Norman (2010) stratifies stillbirth by semen type with no twin
# breakdown; Silva del Rio et al. (2007) [below] stratifies mortality by twin/singleton with no
# semen-type breakdown (their population was ~95% conventional semen). DISCLOSED ASSUMPTION: the
# ratio is treated as semen-type-independent -- untestable with the data actually available.
# Silva del Rio et al. 2007 J Dairy Sci 90(3):1255-1264, abstract:
# "Calf mortality for primiparous and multiparous cows was 5.0% after a single birth and 25.5%
# after twin births, whereas for nulliparous heifers, mortality was 10.4% for singletons and
# 38.0% for twins." Ratios: parity '1' = 38.0/10.4; parities '2'/'3+' = 25.5/5.0 (pooled, matching
# how STILLBIRTH above already pools 2/3+ since the source gives heifer-vs-cow only).
STILLBIRTH_TWIN_RATIO = register("STILLBIRTH_TWIN_RATIO", {
    "1": 38.0 / 10.4, "2": 25.5 / 5.0, "3+": 25.5 / 5.0,
}, source="Silva del Rio et al. 2007 J Dairy Sci 90(3):1255-1264", evidence="read-abstract",
   note="RATIO only, not an absolute rate. Twin sex-pair correlation (also reported in this paper: "
        "30.1% MM/43.6% MF/26.3% FF vs ~50% independence-predicted) is not modelled: by linearity of expectation, "
        "within-pair sex correlation has zero effect on this model's expected per-line calf counts, "
        "since the state space tracks no sibling/pairing information.")

# Twinning rate per calving, by dam parity. Silva del Rio et al. 2007 [read-abstract]:
# 1.2% nulliparous vs 5.8% multiparous. Parities 2 and 3+ carry the multiparous value
# (the full text with a parity breakdown was not accessible). Calves per calving event
# = 1 + rate (triplets ignored; freemartins not modeled).
TWINNING = register("TWINNING", {"1": 0.012, "2": 0.058, "3+": 0.058},
    source="Silva del Rio et al. 2007 J Dairy Sci 90(3):1255-1264", evidence="read-abstract",
    note="parities 2 and 3+ share the multiparous value; twinning is known to rise with parity but no parity-specific number was read")


# ============================================================
# CALF / HEIFER SURVIVAL
# ============================================================
# Mean age at weaning: Urie NJ et al. 2018 J Dairy Sci 101(10):9168-9184, Part I
# (NAHMS Dairy 2014): "mean age at weaning ... was 65.7 d" -> 66 d (whole day).
# (The older NAHMS Dairy 2007 reports 8.2 weeks = 57 d; the more recent value is used.)
WEANING_AGE_DAYS = register("WEANING_AGE_DAYS", 66, source="Urie et al. 2018 Part I (read-full)", evidence="read-full",
    note="65.7 d rounded to whole days")

# Preweaning mortality, dairy heifer calves: Urie et al. 2018 Part V, J Dairy Sci 101:9229-9244
# (NAHMS Dairy 2014 cohort, 2,545 calves, 104 operations): 5.0% (128 deaths), mean age at
# death 24.4 d. NOTE the NAHMS Dairy 2014 operation-reported figure for 2013 is 6.4%
# (report read); the cohort figure is used because it carries the age-at-death information.
PREWEANING_MORTALITY_TOTAL = register("PREWEANING_MORTALITY_TOTAL", 0.050,
    source="Urie et al. 2018 Part V (read-full)", evidence="read-full",
    note="cohort study; NAHMS Dairy 2014 operation-reported 2013 figure is 6.4% (sensitivity)")
PREWEANING_MEAN_AGE_AT_DEATH_DAYS = register("PREWEANING_MEAN_AGE_AT_DEATH_DAYS", 24.4,
    source="Urie et al. 2018 Part V (read-full)", evidence="read-full")
PREWEANING_EARLY_WINDOW_END_DAYS = register("PREWEANING_EARLY_WINDOW_END_DAYS", 30,
    source="boundary of the two derived risk windows", evidence="assumption",
    note="only the mean age at death is published; the split point is a modeling choice")


def _preweaning_window_risks():
    """[derived] Split the total risk into an early and a late window so the deaths'
    mean age equals the published 24.4 d, assuming deaths are uniform WITHIN each
    window (midpoints m1, m2):   mean = w1*m1 + (1-w1)*m2   ->   w1 = (m2-mean)/(m2-m1).
    Returns the CONDITIONAL risk within each window."""
    m1 = PREWEANING_EARLY_WINDOW_END_DAYS / 2.0
    m2 = (PREWEANING_EARLY_WINDOW_END_DAYS + WEANING_AGE_DAYS) / 2.0
    w1 = (m2 - PREWEANING_MEAN_AGE_AT_DEATH_DAYS) / (m2 - m1)
    w1 = min(max(w1, 0.0), 1.0)
    d1 = PREWEANING_MORTALITY_TOTAL * w1
    d2 = PREWEANING_MORTALITY_TOTAL * (1.0 - w1)
    return d1, d2 / (1.0 - d1)


# Weaned-heifer mortality, weaning -> calving, per YEAR. USDA APHIS NAHMS Dairy 2014 report
# (Health and Management Practices, 2013 data): "1.9 percent of weaned heifers were euthanized or
# died without assistance"; corroborated by Dairy 2007 (2006 data): 1.8%. Both reports' text read.
# Applied to weaned heifers from weaning to first insemination; the breeding/pregnant stages use
# the Fodor removal below (all-cause) so deaths are not counted twice.
WEANED_HEIFER_ANNUAL_MORTALITY = register("WEANED_HEIFER_ANNUAL_MORTALITY", 0.019,
    source="USDA APHIS NAHMS Dairy 2014 (2013 data; 1.9%), Dairy 2007 (2006 data; 1.8%)", evidence="read-full",
    note="operation-reported annual death loss among weaned heifers; no age structure within weaning->calving is published")


# ============================================================
# HEIFER BREEDING, PREGNANCY, REMOVAL
# ============================================================
# Age at first insemination: Fodor I, Lang Z, Ozsvari L (2020) Asian-Australas J Anim Sci 33:1360-1368,
# 35,128 Holstein-Friesian heifers, 33 Hungarian herds: mean 15.58 months (x30.4375 d). US support:
# Kuhn MT, Hutchison JL, Wiggans GR (2006) J Dairy Sci 89(12):4907-4920 -- heifer conception is
# maximal at 15-16 months (537,938 breedings, 362,512 heifers, 41 states). No US MEAN age at first
# breeding was found; the Hungarian mean is used.
FIRST_BREEDING_AGE_DAYS = register("FIRST_BREEDING_AGE_DAYS", 474,
    source="Fodor et al. 2020 (Hungary, read); Kuhn et al. 2006 (US optimal 15-16 mo, read-abstract)", evidence="read-abstract",
    note="NON-US mean; deterministic here although the source SD is 1.48 months")

# Heifer conception risk per service, by semen type. The rate is defined by CONFIRMED PREGNANCY
# (CDCB heifer conception rate: inseminations resulting in pregnancy, from pregnancy diagnoses and
# subsequent calving confirmations), so it is ALREADY net of early embryonic loss up to the
# diagnosis -- see youngstock/probabilities/heifer_repro.py for why no further early loss is applied.
#  conventional 56% / sexed 39%: Norman et al. 2010 (1.3 M US heifer breedings) [read-abstract];
#     corroborated by Kuhn et al. 2006 (57% overall) and McWhorter et al. 2020 (55.3% Holstein sires).
#  beef 53.6% = 56% x (53.0/55.3): McWhorter TM et al. 2020 J Dairy Sci 103(11):10374-10382
#     [read-abstract] -- Angus sires 53.0% vs Holstein sires 55.3% in Holstein heifers
#     (19,437 breedings). The RATIO is applied to Norman's baseline; the two datasets'
#     baselines differ slightly. Beef sires were used more often on problem breeders in cows;
#     an Italian dataset (Bittante 2020) found the opposite direction, so the effect is small and uncertain.
HEIFER_CONCEPTION_PER_SERVICE = register("HEIFER_CONCEPTION_PER_SERVICE",
    {"conventional": 0.56, "sexed": 0.39, "beef": round(0.56 * 53.0 / 55.3, 4)},
    source="Norman et al. 2010; McWhorter et al. 2020", evidence="read-abstract",
    note="beef value derived from a McWhorter ratio; cow-side beef/sexed conception effects are NOT represented")

# Days between services converts a per-service conception risk into a per-step probability.
# Forde N, Beltman ME, Lonergan P, Diskin M, Roche JF, Crowe MA (2011) Anim Reprod Sci 124(3-4):163-169
# [read-abstract]: "The oestrous cycle in cattle lasts for 18-24 days". 21 d is the MIDPOINT of that range
# (a point value the abstract does not state). The effective interval between services in field data is
# longer when heats are missed (most, not all, returns fall in 18-24 d); for HEIFERS the emergent time from
# first service to conception (about 25 d at DAY resolution) is consistent with Fodor 2020's 25 d derived
# from ages at first service and first calving. Physiological, not calibrated.
HEIFER_SERVICE_INTERVAL_DAYS = register("HEIFER_SERVICE_INTERVAL_DAYS", 21,
    source="Forde et al. 2011 (oestrous cycle 18-24 d; midpoint)", evidence="read-abstract",
    note="midpoint of a stated range; low leverage; field inter-service intervals are longer when heats are missed")

# Heifer removal between first insemination and first calving. Fodor et al. (2020) counts as printed:
HEIFER_FATE_FODOR2020 = register("HEIFER_FATE_FODOR2020", {
    "n_heifers_inseminated": 35128,
    "culled_before_conception": 1784,
    "sold_before_conception": 253,
    "culled_after_conception_before_calving": 1915,
    "sold_after_conception_before_calving": 406,
    "note": ("All-cause removal ('culled' = slaughtered or died; 'sold' = moved to another farm). No separate "
             "death hazard is applied in these two stages (avoids double counting). Hungarian population."),
}, source="Fodor et al. 2020 (read)", evidence="read-full", note="non-US; used as INPUT for removal, and as an external benchmark for the emergent totals")
_F = HEIFER_FATE_FODOR2020
# [derived] cumulative removal risk before conception (per inseminated heifer)
HEIFER_BREEDING_STAGE_REMOVAL = (_F["culled_before_conception"] + _F["sold_before_conception"]) / _F["n_heifers_inseminated"]
# [derived] cumulative removal risk between conception and calving (per heifer that conceived)
HEIFER_PREGNANT_STAGE_REMOVAL = (_F["culled_after_conception_before_calving"] + _F["sold_after_conception_before_calving"]) / (
    _F["n_heifers_inseminated"] - _F["culled_before_conception"] - _F["sold_before_conception"])

# Day of gestation from which any ADDITIONAL heifer pregnancy loss would be applied, used ONLY by the
# optional 'formula' method of heifer_abortion_spec (a sensitivity), never by the default.
# Assumption: pregnancy diagnosis by ~60 days. Fodor 2020 does not state the diagnosis day.
HEIFER_PREGNANCY_DIAGNOSIS_DAY = register("HEIFER_PREGNANCY_DIAGNOSIS_DAY", 60,
    source="none (Fodor 2020 gives no diagnosis day)", evidence="assumption", note="sensitivity input only; default method applies no extra loss")


# ============================================================
# TERMINAL (SOLD) CALVES
# ============================================================
# Shivley CB, Lombard JE, Urie NJ, Weary DM, von Keyserlingk MAG (2019) J Dairy Sci 102(5):4489-4497
# [read-abstract] (NAHMS Dairy 2014, 42 operations, 10 states): bull calves left the operation at a MEAN
# 7.6 d (median 4 d); 97.6% of operations sold them before weaning; 2.4% of bull calves died before leaving.
# The distribution is right-skewed (median 4 d).
# Beef-cross calves: Bigelow RA, Lancaster PA, White BJ, Amachawadi RG, Barnhardt TR, Theurer ME (2025) Transl Anim Sci
# 9:txaf064 [read-abstract; 15 US calf ranches]: ALL ranches received beef-on-dairy calves at < 4 d of age; 33% of
# ranches received calves at 0-1 d and 67% at 2-4 d (arrival age at the ranch, ranch-level, not calf-level).
# No beef-cross pre-sale mortality on the dairy farm was found -> it reuses the dairy-bull 2.4% and is a flagged
# PLACEHOLDER (Pereira 2024's 2.5% to 60 d is a raising-facility figure after sale).
# Bittante et al. 2020 (35.7 d from 125 Italian farms) is not used, because US practice is within the first week.
#
# FARMER INPUT: the age, in days, at which male dairy calves and beef-cross calves (both sexes) leave the farm. One value for
# the three terminal lines. Default 4 d: the median of the NAHMS dairy bull calves (Shivley 2019) and the upper edge of the
# US calf-ranch arrival ages (Bigelow 2025, all < 4 d); modelling choice inside the supported 3-7 d range, not a measurement.
# CHOOSE: 3 - 7 (days; any value > 0 runs, but only 3-7 d is the supported range)
CALF_SALE_AGE_DAYS = register("CALF_SALE_AGE_DAYS", 4.0,
    source="Shivley et al. 2019 (median 4 d, read-abstract); Bigelow et al. 2025 (all ranches < 4 d at arrival, read-abstract)",
    evidence="assumption",
    note="sale age of male dairy and beef-cross calves is a farmer input in the first week (3-7 d); 4 d chosen. "
         "Not used: the Shivley mean of 7.6 d for dairy bulls and 35.7 d for beef-cross calves (Bittante 2020, Italy), "
         "because US practice is within the first week")
if not CALF_SALE_AGE_DAYS > 0:
    raise ValueError(f"CALF_SALE_AGE_DAYS must be > 0 (days), got {CALF_SALE_AGE_DAYS!r}; the supported range is 3-7")
TERMINAL_SALE_AGE_DAYS = {line: CALF_SALE_AGE_DAYS for line in ("DAIRY_M", "BEEF_F", "BEEF_M")}
TERMINAL_PRESALE_MORTALITY = register("TERMINAL_PRESALE_MORTALITY", {"DAIRY_M": 0.024, "BEEF_F": 0.024, "BEEF_M": 0.024},
    source="Shivley et al. 2019 (2.4% of bull calves died before leaving, read-abstract)", evidence="read-abstract",
    note="BEEF_F/BEEF_M reuse the dairy-bull 2.4%: PLACEHOLDER (likely an overstatement at a sale age of about 4 d; Pereira 2024 reports 2.5% to 60 d in a raising facility AFTER sale)")
TERMINAL_PLACEHOLDER_LINES = ("BEEF_F", "BEEF_M")


# ============================================================
# STRUCTURAL CONSTANTS DERIVED FROM DAYS (resolution-aware)
# ============================================================
from cowmrp.domains.cow.lookup_tables import threshold_crossing_mass, crossing_conditional   # one time convention, shared with the cow module

# Age convention (same as the cow module, cow/probabilities/base.py): an animal born inside a step has a TRUE age
# of k*UNIT_DAYS + u at boundary k (u ~ Uniform(0, UNIT_DAYS)), so an age threshold is crossed at boundary
# k*(u) = ceil((T - u)/UNIT_DAYS): usually one boundary, two when T is not a multiple of the step (MONTH).
BREED_CROSSING = threshold_crossing_mass(FIRST_BREEDING_AGE_DAYS)     # {boundary k: share becoming breeding-eligible there}
K_BREED_FIRST, K_BREED = min(BREED_CROSSING), max(BREED_CROSSING)      # K_BREED = last boundary (probability 1)
TERMINAL_CROSSING = {line: threshold_crossing_mass(d) for line, d in TERMINAL_SALE_AGE_DAYS.items()}
# CALF ages 0..K_SALE exist for a terminal line (sold at boundary K_SALE+1 at the latest); shares whose sale
# boundary is 0 (sale age shorter than the remainder of the birth step) never enter the inventory.
K_SALE = {line: max(m) - 1 for line, m in TERMINAL_CROSSING.items()}
# Share of surviving terminal calves still on the farm at the first boundary (the rest were sold inside the birth step).
TERMINAL_CARRY = {line: 1.0 - m.get(0, 0.0) for line, m in TERMINAL_CROSSING.items()}


def _dp_calve_range():
    """(first, last) pregnant step with any calving mass -- taken from the COW module's own calving
    function (cow.probabilities.calving.calving_step_mass) so heifers and cows share ONE gestation
    convention. At DAY and WEEK first == last; at MONTH calving is spread over two steps (9 and 10)."""
    from cowmrp.domains.cow.probabilities.calving import calving_step_mass
    mass = calving_step_mass()
    return min(mass), max(mass)


DP_CALVE_FIRST, DP_CALVE = _dp_calve_range()      # DP_CALVE = LAST pregnant step (calving probability 1)


def parameter_table():
    """One row per registered parameter (for docs and the metadata test)."""
    import pandas as pd
    return pd.DataFrame([dict(parameter=k, **v) for k, v in PARAMETER_SOURCES.items()])

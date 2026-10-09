"""
cow/rewards/base.py
===================
Shared architecture for reward specs: helper functions, economic parameters.
"""

import numpy as np
from cow.parameters import UNIT_DAYS
from mrp.engine import resolve_method


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================
# Literature-grounded 2026-08 pass -- see docs/REWARD_ECONOMICS_REVIEW_
# 2026-08.md (local-only, gitignored) for full sourcing, confidence
# levels, and the paired formula-correctness fixes (F1/F2/F3, in
# reproduction_cost.py/replacement_cost.py/cull_reward.py respectively).
# Values marked UNCHANGED were investigated this pass and found either
# already well-calibrated or left pending further research -- see the
# review doc for the reasoning, not just the inline comment.

# US All Milk Price, 2022-2024 average (USDA NASS All Milk Price;
# farmdoc daily/Zwilling 2024, citing USDA-ERS Dec 2024 Outlook).
MILK_PRICE_PER_KG = 0.474  # was 0.45 (uncited)

# UNCHANGED -- investigated. A direct-source figure (Eastridge & Weiss
# 2007, OSU Buckeye Dairy News 9(3): $0.06-0.08/lb dietary DM),
# vintage-checked against 2007 vs. 2023-2024 corn prices (surprisingly
# little movement: $3.17-4.39/bu then vs. $4.17-4.75/bu now) and
# escalated modestly, suggests ~0.20-0.22 -- already close to this
# value. A separate USDA-ERS-based derivation suggesting higher was
# found to likely overstate (that series allocates replacement-heifer
# feed costs into the milk-production denominator). Left unchanged
# pending a cleaner current-year direct figure.
FEED_PRICE_PER_KG_DM = 0.28

# UNCHANGED -- investigated, evidence inconclusive/conflicting across
# 2024-2026 sources (beef-cross calf premium, kept-replacement-heifer
# valuation gap not captured anywhere in this model). Needs a dedicated
# follow-up (a real herd sex/destination split), not a price swap.
CALF_VALUE = 250.0

# US cull-cow price, DRESSED basis, 2023-2024 (USDA AMS Livestock
# Market News -- cull-cow reports are explicitly dressed-basis, plant-
# delivered quotes). Must be applied to DRESSED weight, not live weight
# -- paired with DRESSING_PCT below (see cull_reward.py).
BEEF_PRICE_PER_KG = 2.30

# Cull dairy cow dressing percentage (carcass weight / live weight),
# ~46.3% (live 701.5 kg / carcass 325 kg, cull-dairy-cow carcass-traits
# study).
DRESSING_PCT = 0.463

# UNCHANGED -- already well-calibrated. Multiple regional sale-barn fee
# schedules plus one producer's reported all-in cost ($55.75) converge
# on the current value.
CULLING_COST = 60.0

# UNCHANGED -- weakest-sourced item in this pass's review (a stale 2002
# US study; Ontario rendering fees, a different country/regulatory
# regime, only loosely converted). Kept pending a current, US-specific
# source.
DISPOSAL_COST = 120.0

# Replacement heifer market price, 2022-2024 avg. CITATION REMOVED
# 2026-09-23 -- unverified, not independently checked. Currently UNUSED by
# replacement_cost_formula under REPLACEMENT_STRATEGY="rear" below --
# kept at a researched value so "buy" can be enabled later without also
# needing a separate price update.
PURCHASE_COST = 2200.0  # was 1800.0

# Cost to raise a dairy heifer birth-to-calving, 2024 ($2,258-2,651
# range, midpoint used). CITATION REMOVED 2026-09-23 -- unverified, not
# independently checked. Alternative to PURCHASE_COST (buy vs. raise,
# not additive -- see REPLACEMENT_STRATEGY below).
REARING_COST = 2450.0

# INCREMENTAL dry-period-specific premium (e.g. a closeup/transition
# ration cost above baseline), NOT the cow's entire dry-period feed cost
# -- feed_cost_formula already charges dry cows for their reduced,
# milk-free DMI separately. Do not reinterpret this as total dry-cow
# ration cost; that would double-count against feed_cost.
DRY_PERIOD_DAILY_COST = 1.50

# Replacement-cost pathway selector (see replacement_cost.py). "rear"
# (default, only implemented value) charges REARING_COST alone. "buy"
# (PURCHASE_COST alone) is a documented future extension, not
# implemented -- selecting it raises NotImplementedError. The old
# additive behavior (both costs summed) is not offered; it was a
# confirmed double-count.
REPLACEMENT_STRATEGY = "rear"


AI_COST = 31.0  # was 25.0 -- general AI-service cost ranges ($20-100
# broad, $30-40 narrower), current.
DIAGNOSIS_COST = 8.0  # UNCHANGED -- already matches basic palpation/
# blood-test diagnosis cost ($5-10/head); would need updating only if
# switching to ultrasound-based diagnosis ($50-150/animal).
SYNCHRONIZATION_COST = 13.0  # was 12.0 -- single Ovsynch protocol, no
# CIDR (GnRH + PGF + handling cost breakdown), current.
import os as _os
# Days-open penalty: OFF by default (audit P1, 2026-10-08). The $3.20-5.10/day of De Vries (2006) is the
# marginal cost of an extra open day computed FROM milk, calving-delay and replacement flows -- flows this MRP
# already books explicitly (milk income, feed, calving chain, culling). Adding it again double counts: with the
# penalty the model's own implied cost is ~$6/open day (outside the published range), without it ~$3.2-3.6
# (inside). Kept only as an optional SENSITIVITY component (set COW_DAYS_OPEN_PENALTY=4 to reproduce the old
# behaviour); see docs/COW_MODEL_AUDIT.md and L-051. MUST stay paired with reproduction_cost.py's per-period
# charging formula.
DAYS_OPEN_PENALTY_PER_DAY = float(_os.environ.get('COW_DAYS_OPEN_PENALTY', 0.0))
from cow.parameters import VOLUNTARY_WAITING_PERIOD_DAYS  # noqa: F401  (single source in cow/parameters.py)
ABORTION_COST = 150.0  # UNCHANGED -- confirmed appropriate for this
# model's actual scope (early-gestation losses, dp_days 19-90 via
# cow.parameters.ABORTION_ALBAAJ2022): De Vries (2006)/Cabrera/Palma et
# al. (2025) show a strongly stage-dependent $90-1,900 range; 150 fits
# the low (early-gestation) end this model covers, not the $555
# overall average across all gestation stages.


# ============================================================
# BODY WEIGHT & FEED (simplified)
# ============================================================

def max_lactation_exit_probability(states, probs):
    """Probability that a cow LEAVES the herd by calving at the maximum modelled lactation. The transition
    (cow/transitions.py) routes these cows to the replacement state exactly like a culled cow, so the economics
    must treat them the same way: she is sold (salvage - culling cost) and replaced (rearing cost). Zero for
    every other lactation."""
    from cow.parameters import MAX_LACTATION
    survive = (1 - probs['p_dead']) * (1 - probs['p_cull'])
    return np.where(np.asarray(states['lactation'].values) >= MAX_LACTATION,
                    survive * (1 - probs['p_abort']) * probs['p_calve'], 0.0)


def mid_step_dim_days(dim):
    """Mean days in milk during DIM step `dim`: (dim + 1) * UNIT_DAYS. A cow in state dim = k calved at a
    uniform time inside the previous step, so she spends step k at mean age k*U + U/2 + U/2
    (same entry convention as cow/lookup_tables.py)."""
    return (np.asarray(dim) + 1) * UNIT_DAYS


def age_days_of(lactation, dim):
    """Approximate age from lactation number and DIM."""
    from cow.parameters import AGE_AT_FIRST_CALVING_DAYS, AVERAGE_CALVING_INTERVAL_DAYS
    dim_days = mid_step_dim_days(dim)
    return AGE_AT_FIRST_CALVING_DAYS + (lactation - 1) * AVERAGE_CALVING_INTERVAL_DAYS + dim_days


def body_weight_kg(age_days, dim_days, dp_days):
    """Simplified body weight model."""
    from cow.parameters import BW_MATURE_KG, BW_MAX_POSTPARTUM_LOSS_KG, BW_DAYS_TO_MIN_WEIGHT_DAYS, BW_PREGNANCY_GAIN_KG, GESTATION_LENGTH_DAYS
    growth = BW_MATURE_KG
    postpartum_loss = BW_MAX_POSTPARTUM_LOSS_KG * np.exp(-dim_days / BW_DAYS_TO_MIN_WEIGHT_DAYS)
    pregnancy_gain = BW_PREGNANCY_GAIN_KG * np.minimum(1.0, dp_days / GESTATION_LENGTH_DAYS)
    return growth - postpartum_loss + pregnancy_gain


def dry_matter_intake_kg(milk_kg, body_weight_kg_, dim_days):
    """Simplified DMI equation."""
    fcm = milk_kg  # fat-corrected milk approximation
    return (0.372 * fcm + 0.0968 * body_weight_kg_ ** 0.75) * (1 - 0.2 * np.exp(-0.1 * dim_days))

"""
cow/rewards/base.py
===================
Shared architecture for reward specs: helper functions, economic parameters.
"""

import numpy as np
from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.mrp.engine import resolve_method


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================
# Each value carries its source or the reasoning behind it in the comment above it.

# US All Milk Price, 2022-2024 average (USDA NASS All Milk Price;
# farmdoc daily/Zwilling 2024, citing USDA-ERS Dec 2024 Outlook).
MILK_PRICE_PER_KG = 0.474

# Feed cost per kg dry matter. Eastridge & Weiss (2007, OSU Buckeye Dairy News
# 9(3)) give $0.06-0.08 per lb of dietary DM. Corn prices changed little between
# 2007 ($3.17-4.39/bu) and 2023-2024 ($4.17-4.75/bu), so a modest escalation
# gives about 0.20-0.22, close to the value used. A USDA-ERS-based derivation
# gives a higher figure but likely overstates it, because that series allocates
# replacement-heifer feed cost into the milk-production denominator.
FEED_PRICE_PER_KG_DM = 0.28

# Blended value of one calf at birth under conventional (unsexed) dairy semen: about 50 % heifer and
# 50 % bull calves, all treated as sold at auction. This is a simplification: no split between kept and
# sold heifers, no beef-cross calves, and no rearing or colostrum cost. 0.5 x ~1,300 (80-85 lb Holstein
# heifer calves, USDA AMS Indiana report of 2026-05-19: ~$1,200-1,375) + 0.5 x ~1,000 (Holstein bull
# calves, ~$800-1,300 across 2026 US auction reports) = ~1,150. Auction reports are regional and weekly,
# not a national average; beef-cross calves ($1,000-2,000) are not represented.
CALF_VALUE = 1150.0

# US cull-cow price, DRESSED basis, 2023-2024 (USDA AMS Livestock
# Market News -- cull-cow reports are explicitly dressed-basis, plant-
# delivered quotes). Must be applied to DRESSED weight, not live weight
# -- paired with DRESSING_PCT below (see cull_reward.py).
BEEF_PRICE_PER_KG = 2.30

# Cull dairy cow dressing percentage (carcass weight / live weight),
# ~46.3% (live 701.5 kg / carcass 325 kg, cull-dairy-cow carcass-traits
# study).
DRESSING_PCT = 0.463

# Cost of culling a cow. Several regional sale-barn fee schedules and one
# producer's reported all-in cost ($55.75) are close to this value.
CULLING_COST = 60.0

# Cost of disposing of a dead cow. This is the weakest-sourced value in this file:
# it rests on a 2002 US study and on Ontario rendering fees (a different country
# and regulatory regime, loosely converted). A current US source would improve it.
DISPOSAL_COST = 120.0

# Replacement heifer market price, 2022-2024 average. Not used by
# replacement_cost_formula while REPLACEMENT_STRATEGY = "rear" (below); it is
# kept so the "buy" strategy can be added later. Modelling assumption: no
# single published source is cited for this value.
PURCHASE_COST = 2200.0

# Cost to raise a dairy heifer from birth to calving, 2024 ($2,258-2,651
# range, midpoint used). Modelling assumption: no single published source is
# cited for this range. It is an alternative to PURCHASE_COST (buy or raise,
# not both; see REPLACEMENT_STRATEGY below).
REARING_COST = 2450.0

# INCREMENTAL dry-period-specific premium (e.g. a closeup/transition
# ration cost above baseline), NOT the cow's entire dry-period feed cost
# -- feed_cost_formula already charges dry cows for their reduced,
# milk-free DMI separately. Do not reinterpret this as total dry-cow
# ration cost; that would double-count against feed_cost.
DRY_PERIOD_DAILY_COST = 1.50

# Replacement-cost pathway selector (see replacement_cost.py). "rear" (the
# default and the only available value) charges REARING_COST alone. "buy"
# (PURCHASE_COST alone) is a possible future extension; selecting it raises
# NotImplementedError. Summing both costs is not offered because it would
# count the replacement twice.
REPLACEMENT_STRATEGY = "rear"


AI_COST = 31.0  # Artificial-insemination service cost; general ranges are $20-100
# broad and $30-40 narrower.
DIAGNOSIS_COST = 8.0  # Basic palpation or blood-test pregnancy diagnosis ($5-10 per
# head); ultrasound would cost $50-150 per animal.
SYNCHRONIZATION_COST = 13.0  # Single Ovsynch protocol without CIDR (GnRH + PGF +
# handling cost breakdown).
import os as _os
# Days-open penalty: off by default. The $3.20-5.10 per day of De Vries (2006) is the marginal cost
# of an extra open day computed from milk, calving-delay and replacement flows, and this model already
# books those flows explicitly (milk income, feed, calving chain, culling). Adding the penalty counts
# them twice: with it the model's implied cost is about $6 per open day (outside the published range),
# without it about $3.2-3.6 (inside). It is kept only as an optional sensitivity component (set
# COW_DAYS_OPEN_PENALTY=4 to switch it on). It must stay paired with the per-period charging formula in
# reproduction_cost.py.
DAYS_OPEN_PENALTY_PER_DAY = float(_os.environ.get('COW_DAYS_OPEN_PENALTY', 0.0))
from cowmrp.domains.cow.parameters import VOLUNTARY_WAITING_PERIOD_DAYS  # noqa: F401  (single source in cow/parameters.py)
ABORTION_COST = 150.0  # Cost of a pregnancy loss. The model covers early-gestation losses only
# (dp_days 19-90, see ABORTION_ALBAAJ2022 in the cow parameters). De Vries (2006), Cabrera and
# Palma et al. (2025) show a strongly stage-dependent range of $90-1,900; 150 fits the low
# (early-gestation) end that the model covers, not the $555 average across all gestation stages.


# ============================================================
# BODY WEIGHT & FEED (simplified)
# ============================================================

def max_lactation_exit_probability(states, probs):
    """Probability that a cow LEAVES the herd by calving at the maximum modelled lactation. The transition
    (cow/transitions.py) routes these cows to the replacement state exactly like a culled cow, so the economics
    must treat them the same way: she is sold (salvage - culling cost) and replaced (rearing cost). Zero for
    every other lactation."""
    from cowmrp.domains.cow.parameters import MAX_LACTATION
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
    from cowmrp.domains.cow.parameters import AGE_AT_FIRST_CALVING_DAYS, AVERAGE_CALVING_INTERVAL_DAYS
    dim_days = mid_step_dim_days(dim)
    return AGE_AT_FIRST_CALVING_DAYS + (lactation - 1) * AVERAGE_CALVING_INTERVAL_DAYS + dim_days


def body_weight_kg(age_days, dim_days, dp_days):
    """Simplified body weight model."""
    from cowmrp.domains.cow.parameters import BW_MATURE_KG, BW_MAX_POSTPARTUM_LOSS_KG, BW_DAYS_TO_MIN_WEIGHT_DAYS, BW_PREGNANCY_GAIN_KG, GESTATION_LENGTH_DAYS
    growth = BW_MATURE_KG
    postpartum_loss = BW_MAX_POSTPARTUM_LOSS_KG * np.exp(-dim_days / BW_DAYS_TO_MIN_WEIGHT_DAYS)
    pregnancy_gain = BW_PREGNANCY_GAIN_KG * np.minimum(1.0, dp_days / GESTATION_LENGTH_DAYS)
    return growth - postpartum_loss + pregnancy_gain


def dry_matter_intake_kg(milk_kg, body_weight_kg_, dim_days):
    """Simplified DMI equation."""
    fcm = milk_kg  # fat-corrected milk approximation
    return (0.372 * fcm + 0.0968 * body_weight_kg_ ** 0.75) * (1 - 0.2 * np.exp(-0.1 * dim_days))

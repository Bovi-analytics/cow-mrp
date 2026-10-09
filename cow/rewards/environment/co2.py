"""
cow/rewards/environment/co2.py
=================================
Respiratory CO2 -- a TRACKED biological quantity only, NOT a reward.

Per the scientific audit: livestock respiratory CO2 is excluded from
standard GHG-inventory accounting (IPCC, US EPA) because it is biogenic
and assumed carbon-neutral over the feed-growth cycle -- only CH4 (which
diverts carbon out of that fast cycle into a longer-lived, higher-GWP
gas) is counted. Computing and exposing this quantity is still useful
(biological/energetic insight), but it is deliberately NOT wired into a
reward_co2 spec or reward_total by default -- see
cow/rewards/environment/__init__.py and cow/rewards/__init__.py. If you
want to price this deliberately (overriding the standard convention),
add a reward_co2.py spec following reward_nitrogen.py's pattern and
document the deviation explicitly.
"""

import numpy as np
from cow.parameters import UNIT_DAYS, DIET_ME_MCAL_PER_KG, DIET_FAT_PCT, DIET_PROTEIN_PCT

# CO2_L_PER_MCAL_HP -- CORRECTED and now VERIFIED this pass, derived
# transparently from Pedersen et al. (2008)'s own primary data (fetched
# and read directly, not via secondary summary):
#   Pedersen S, Blanes-Vidal V, Joergensen H, Chwalibog A, Haeussermann
#   A, Heetkamp MJW, Aarnink AJA (2008). "Carbon Dioxide Production in
#   Animal Houses: A Literature Review." Agricultural Engineering
#   International: CIGR Ejournal, Manuscript BC 08 008, Vol. X, December
#   2008.
# The paper's Table 5 (University of Wageningen) reports three DIRECT
# COW measurements of 0.179, 0.174, 0.181 m3 CO2 h^-1 hpu^-1 (mean
# 0.178 = 178 L/h/hpu) -- distinct from, and more specific than, the
# general cross-species "185 L/h/hpu" figure the paper also cites.
# 1 hpu = 1000 W heat production at 20C (paper's own definition) =
# 1 kJ/s x 3600 s/h = 3.6 MJ/h = 3.6/4.184 = 0.8604 Mcal/h.
#   CO2_L_PER_MCAL_HP = 178 / 0.8604 = 206.9, rounded to 207.
# This is close to (within ~1%, inside the source data's own precision)
# the value previously used here (209) but is now derived transparently
# from cow-specific primary data rather than an unverified constant.
CO2_L_PER_MCAL_HP = 207.0
L_TO_KG_CO2 = 0.00196           # 44 g/mol / 22.4 L/mol at STP -- verified, standard conversion

NE_MAINT_COEFFICIENT = 0.10     # NEmaint (Mcal/d) = 0.10 x BW^0.75. UPDATED 2026-09-23 --
                                 # both editions now primary-verified from user-supplied PDFs
                                 # (docs/ReferenceFiles for Probability and Rewards/), read
                                 # directly, not via secondary summary:
                                 #   NRC (2001) Nutrient Requirements of Dairy Cattle, 7th
                                 #   rev. ed., p.18: "Measured fasting heat production (Flatt
                                 #   et al., 1965) in dry non-pregnant dairy cows averaged
                                 #   0.073 Mcal/kg BW^0.75 ... a 10 percent activity allowance
                                 #   was added ... maintenance requirement for NEL is set at
                                 #   0.080 Mcal/kg BW^0.75 for mature dairy cows."
                                 #   NASEM (2021) Nutrient Requirements of Dairy Cattle, 8th
                                 #   rev. ed. (DOI 10.17226/25806), p.28, "Changes from the
                                 #   seventh revised edition": "The maintenance requirement is
                                 #   increased from 0.08 to 0.10 Mcal per kg of metabolic BW."
                                 # NASEM's own stated revision superseding NRC 2001 for the
                                 # SAME mature-cow maintenance coefficient -- not a differing
                                 # source in dispute, the newer edition's explicit correction
                                 # of the older one. Old value (0.08) retained nowhere else;
                                 # this quantity is a tracked biological output only, never
                                 # wired into reward_total by default (module docstring above),
                                 # so this change does not alter any reward or V value under
                                 # default config. NOTE: NASEM (2021) p.224 has a DIFFERENT
                                 # figure, 0.101 Mcal/kg BW^0.75 -- that is preweaned-calf
                                 # maintenance ME (Ch. 10), a different life stage; not this
                                 # constant, do not conflate.
HP_FLOOR_FRACTION = 0.6         # Numerical safeguard ONLY (prevents negative CO2 during severe
                                 # negative energy balance in early lactation) -- NOT a
                                 # biological finding. No source was offered for this specific
                                 # value in the audited material; retained only as a documented
                                 # modeling safeguard, per the audit's explicit instruction.

# Table H (NEpreg by days pregnant) from the candidate spec, retained as
# a labeled modeling approximation -- no independent citation was found
# for these specific values in this audit pass.

def _ne_preg_mcal(dp_days):
    return np.select(
        [dp_days <= 100, dp_days <= 150, dp_days <= 200, dp_days <= 250, dp_days <= 280],
        [0.0,             0.3,            0.7,            1.4,            2.5],
        default=2.5,  # beyond 280 shouldn't occur pre-calving; safe fallback, not extrapolation
    )


def _energy_balance(states, context):
    """
    HP (Mcal/d) = ME_intake - NE_milk - NE_maint - NE_preg -- the
    standard bioenergetic heat-production definition (NASEM/ARC/INRA-
    consistent: ME = NE_uses + HP).
    CO2 (L/d) = HP x 207 -- derived from Pedersen et al. (2008)'s
    cow-specific CO2/hpu measurements; see CO2_L_PER_MCAL_HP above.
    """
    dp = states['dp'].values

    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS
    milk_kg_per_day = context['milk_yield_kg'] / UNIT_DAYS
    bw_kg = context['bw_kg']

    me_intake = dmi_kg_per_day * DIET_ME_MCAL_PER_KG
    ne_milk = milk_kg_per_day * (0.0929 * DIET_FAT_PCT + 0.0547 * DIET_PROTEIN_PCT / 0.93 + 0.192)
    ne_maint = NE_MAINT_COEFFICIENT * bw_kg ** 0.75
    dp_days = dp * UNIT_DAYS
    ne_preg = _ne_preg_mcal(dp_days)

    hp = me_intake - ne_milk - ne_maint - ne_preg
    hp = np.maximum(hp, HP_FLOOR_FRACTION * me_intake)

    co2_l_per_day = hp * CO2_L_PER_MCAL_HP
    co2_kg_per_day = co2_l_per_day * L_TO_KG_CO2
    return co2_kg_per_day * UNIT_DAYS


respiratory_co2_spec = {
    'name': 'respiratory_co2_kg', 'produces': 'respiratory_co2_kg',
    'depends_on': {
        'state': ['dp'], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['DIET_ME_MCAL_PER_KG', 'DIET_FAT_PCT', 'DIET_PROTEIN_PCT'],
        'external_input': [], 'model_output': ['dmi_kg', 'milk_yield_kg', 'bw_kg'], 'time': [],
    },
    'method': 'energy_balance', 'methods': {'energy_balance': _energy_balance},
}

CO2_METHOD_INFO = {
    'energy_balance': {
        'citation': 'Pedersen S, Blanes-Vidal V, Joergensen H, Chwalibog A, Haeussermann A, '
                    'Heetkamp MJW, Aarnink AJA (2008). "Carbon Dioxide Production in Animal '
                    'Houses: A Literature Review." Agricultural Engineering International: '
                    'CIGR Ejournal, Manuscript BC 08 008, Vol. X -- VERIFIED (fetched and read '
                    'directly this pass; corrects the candidate spec\'s "Biosystems Engineering '
                    '100(3):413-418" citation, which does not match).',
        'applicable_animals': 'Dairy cattle (CO2_L_PER_MCAL_HP derived from the paper\'s own '
                               'cow-specific measurements, not the general cross-species figure)',
        'scope': 'A barn-ventilation ENGINEERING method (estimating airflow from a CO2 mass '
                 'balance), adapted here to estimate a biological quantity -- NOT a '
                 'GHG-inventory method (see module docstring).',
        'required_inputs': ['dmi_kg', 'milk_yield_kg', 'bw_kg', 'dp'],
        'optional_inputs': [],
        'equation': 'HP=ME-NEmilk-NEmaint-NEpreg; CO2(L/d)=HP*207',
        'units': 'kg CO2/day',
        'parameter_sources': 'Pedersen et al. (2008) Table 5 (cow CO2/hpu measurements); '
                              'NASEM (2021) p.28 (NE maintenance coefficient, primary-verified '
                              '2026-09-23; supersedes NRC (2001) p.18\'s 0.08 -- see '
                              'NE_MAINT_COEFFICIENT above).',
        'assumptions': 'RQ (respiratory quotient) ~= 1.0 (implicit in using a single L/Mcal '
                        'ratio); HP floored at 0.6xME as a numerical safeguard, not biology.',
        'limitations': 'CO2_L_PER_MCAL_HP=207 is derived from only 3 direct cow measurements '
                        'in the primary source (range 0.174-0.181 m3/h/hpu) -- a small sample. '
                        'NE maintenance coefficient (0.10) now uses the current NASEM (2021) '
                        'value (primary-verified, p.28), which superseded the prior NRC (2001) '
                        'value of 0.08 (see NE_MAINT_COEFFICIENT above). NOT GHG-inventory-eligible -- '
                        'IPCC/EPA exclude respiratory CO2 as biogenic/carbon-neutral (see '
                        'module docstring).',
    },
}

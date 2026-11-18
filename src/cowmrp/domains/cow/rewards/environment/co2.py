"""
cow/rewards/environment/co2.py
=================================
Respiratory CO2: a tracked biological quantity, not a reward.

Livestock respiratory CO2 is excluded from standard greenhouse-gas
inventories (IPCC, US EPA) because it is biogenic and assumed
carbon-neutral over the feed-growth cycle; only CH4 is counted. The
quantity is still computed because it describes the cow's energy use, but
it is not wired into a reward or into ``reward_total`` by default. To price
it deliberately (a departure from the standard convention), add a
``reward_co2.py`` spec following the pattern of ``reward_nitrogen.py`` and
document the departure.
"""

import numpy as np
from cowmrp.domains.cow.parameters import UNIT_DAYS, DIET_ME_MCAL_PER_KG, DIET_FAT_PCT, DIET_PROTEIN_PCT

# CO2_L_PER_MCAL_HP: litres of CO2 per Mcal of heat production, derived from
#   Pedersen S, Blanes-Vidal V, Joergensen H, Chwalibog A, Haeussermann
#   A, Heetkamp MJW, Aarnink AJA (2008). "Carbon Dioxide Production in
#   Animal Houses: A Literature Review." Agricultural Engineering
#   International: CIGR Ejournal, Manuscript BC 08 008, Vol. X, December
#   2008.
# Table 5 (University of Wageningen) reports three direct cow measurements of
# 0.179, 0.174 and 0.181 m3 CO2 per hour per hpu (mean 0.178 = 178 L/h/hpu).
# This is more specific than the cross-species figure of 185 L/h/hpu that the
# paper also cites.
# 1 hpu = 1000 W of heat production at 20 C = 3.6 MJ/h = 3.6/4.184 = 0.8604 Mcal/h.
#   CO2_L_PER_MCAL_HP = 178 / 0.8604 = 206.9, rounded to 207.
CO2_L_PER_MCAL_HP = 207.0
L_TO_KG_CO2 = 0.00196           # 44 g/mol / 22.4 L/mol at STP

NE_MAINT_COEFFICIENT = 0.10     # NEmaint (Mcal/d) = 0.10 x BW^0.75.
                                 # NASEM (2021) Nutrient Requirements of Dairy Cattle, 8th
                                 # rev. ed. (DOI 10.17226/25806), p.28, "Changes from the
                                 # seventh revised edition": "The maintenance requirement is
                                 # increased from 0.08 to 0.10 Mcal per kg of metabolic BW."
                                 # The previous edition, NRC (2001) p.18, set the NEL
                                 # maintenance requirement at 0.080 Mcal/kg BW^0.75 for mature
                                 # dairy cows. This quantity is a tracked output only and is
                                 # not part of the reward total by default.
                                 # Not to be confused with NASEM (2021) p.224, 0.101 Mcal/kg
                                 # BW^0.75, which is preweaned-calf maintenance ME (Ch. 10).
HP_FLOOR_FRACTION = 0.6         # Numerical safeguard only: prevents negative CO2 during severe
                                 # negative energy balance in early lactation. It is a modelling
                                 # choice, not a measured value.

# NEpreg (Mcal/d) by days pregnant: a modelling approximation. These step
# values have no independent citation.

def _ne_preg_mcal(dp_days):
    return np.select(
        [dp_days <= 100, dp_days <= 150, dp_days <= 200, dp_days <= 250, dp_days <= 280],
        [0.0,             0.3,            0.7,            1.4,            2.5],
        default=2.5,  # beyond 280 shouldn't occur pre-calving; safe fallback, not extrapolation
    )


def _energy_balance(states, context):
    """
    HP (Mcal/d) = ME_intake - NE_milk - NE_maint - NE_preg, the standard
    bioenergetic definition of heat production (ME = NE uses + HP).
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
                    'CIGR Ejournal, Manuscript BC 08 008, Vol. X.',
        'applicable_animals': 'Dairy cattle (CO2_L_PER_MCAL_HP derived from the paper\'s own '
                               'cow-specific measurements, not the general cross-species figure)',
        'scope': 'A barn-ventilation engineering method (estimating airflow from a CO2 mass '
                 'balance), adapted here to estimate a biological quantity. It is not a '
                 'greenhouse-gas inventory method (see module docstring).',
        'required_inputs': ['dmi_kg', 'milk_yield_kg', 'bw_kg', 'dp'],
        'optional_inputs': [],
        'equation': 'HP=ME-NEmilk-NEmaint-NEpreg; CO2(L/d)=HP*207',
        'units': 'kg CO2/day',
        'parameter_sources': 'Pedersen et al. (2008) Table 5 (cow CO2/hpu measurements); '
                              'NASEM (2021) p.28 (NE maintenance coefficient 0.10, which '
                              'replaces the 0.08 of NRC (2001) p.18).',
        'assumptions': 'RQ (respiratory quotient) ~= 1.0 (implicit in using a single L/Mcal '
                        'ratio); HP floored at 0.6xME as a numerical safeguard, not biology.',
        'limitations': 'CO2_L_PER_MCAL_HP = 207 rests on only 3 direct cow measurements in the '
                        'source (range 0.174-0.181 m3/h/hpu), a small sample. The NE maintenance '
                        'coefficient (0.10) is the NASEM (2021) value; NRC (2001) used 0.08. Not '
                        'eligible for greenhouse-gas inventories: IPCC and EPA exclude respiratory '
                        'CO2 as biogenic and carbon-neutral (see module docstring).',
    },
}

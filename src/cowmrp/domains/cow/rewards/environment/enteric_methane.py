"""
Enteric methane (rumen fermentation), g CH4 per model step.

Two methods are available:

- ``charmley_2016``: empirical dry-matter-intake (DMI) yield factor.
- ``ipcc_tier2``: energy-balance method (gross-energy intake x Ym / 55.65 MJ per kg CH4).

Both methods use the herd-level diet averages (the ``DIET_*`` parameters); the diet is
not described by individual feed ingredients.
"""

from cowmrp.domains.cow.parameters import (
    UNIT_DAYS, DIET_ME_MCAL_PER_KG, DIET_DE_PCT, IPCC_TIER2_YM_PERCENT,
)

MCAL_TO_MJ = 4.184
ME_OVER_DE = 0.82              # ME = 0.82 x DE. NRC (2001) Chapter 2, following NRC (1996).
                                # Caveat: the ratio comes from high-forage cattle fed near
                                # maintenance; diets rich in concentrate usually have a higher
                                # ratio (often above 0.90). The assumed diet here
                                # (DIET_DE_PCT = 73 %) is closer to that regime, so 0.82
                                # probably understates the true ME/DE.
CH4_ENERGY_MJ_PER_KG = 55.65   # Energy content of methane, MJ per kg CH4.
                                # IPCC (2019) Vol 4 Ch 10, Equation 10.21.


def _charmley_2016(states, context):
    """
    CH4 (g/d) = 20.7 x DMI(kg/d).

    Charmley E, Williams SRO, Moate PJ, Hegarty RS, Herd RM, Oddy VH,
    Reyenga P, Staunton KM, Anderson A, Hannah MC (2016). "A universal
    equation to predict methane production of forage-fed cattle in
    Australia." Animal Production Science 56:169-180.

    The equation was fitted on 1034 individual records of Australian
    forage-fed cattle (beef and dairy, diets above 70 % forage). It has not
    been shown to apply to confinement dairy herds fed a high-concentrate
    total mixed ration, as the DMI and milk model here implies, so treat it
    as a forage-diet estimate.
    """
    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS
    ch4_g_per_day = 20.7 * dmi_kg_per_day
    return ch4_g_per_day * UNIT_DAYS


def _ipcc_tier2(states, context):
    """
    CH4 (kg/d) = GE_intake(MJ/d) x (Ym/100) / 55.65 MJ/kg CH4.

    IPCC (2019), 2019 Refinement to the 2006 IPCC Guidelines, Vol 4 Ch
    10, Equation 10.21 (Tier 2 enteric fermentation), with the energy
    content of methane of 55.65 MJ per kg CH4.

    Ym (methane conversion factor, % of gross energy lost as CH4) defaults
    to IPCC_TIER2_YM_PERCENT = 6.0, the value in IPCC (2019) Table 10.12
    for high-producing dairy cows (above 8500 kg per head per year) on
    diets with NDF above 35 % of DM. The same table gives 5.7 % (high-producing,
    NDF 35 % or less), 6.3 % (medium-producing, 5000-8500 kg per year) and
    6.5 % (low-producing, below 5000 kg per year). The model tracks neither
    diet NDF nor an annual-yield tier, so one value is used for every cow.

    Gross energy (GE) intake is not observed in this model. It is derived
    from DIET_ME_MCAL_PER_KG and DIET_DE_PCT with ME = 0.82 x DE, then
    GE = DE / (DE% / 100). This adds one approximation beyond a Tier 2
    calculation that uses measured diet GE.
    """
    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS

    de_mcal_per_kg = DIET_ME_MCAL_PER_KG / ME_OVER_DE
    ge_mcal_per_kg = de_mcal_per_kg / (DIET_DE_PCT / 100.0)
    ge_mj_per_day = dmi_kg_per_day * ge_mcal_per_kg * MCAL_TO_MJ

    ch4_kg_per_day = ge_mj_per_day * (IPCC_TIER2_YM_PERCENT / 100.0) / CH4_ENERGY_MJ_PER_KG
    ch4_g_per_day = ch4_kg_per_day * 1000.0
    return ch4_g_per_day * UNIT_DAYS


enteric_methane_spec = {
    'name': 'enteric_ch4_g', 'produces': 'enteric_ch4_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['DIET_ME_MCAL_PER_KG', 'DIET_DE_PCT', 'IPCC_TIER2_YM_PERCENT'],
        'external_input': [], 'model_output': ['dmi_kg'], 'time': [],
    },
    'method': 'charmley_2016',
    'methods': {
        'charmley_2016': _charmley_2016,
        'ipcc_tier2': _ipcc_tier2,
    },
}

ENTERIC_METHANE_METHOD_INFO = {
    'charmley_2016': {
        'citation': 'Charmley et al. (2016), Animal Production Science 56:169-180, '
                    '"A universal equation to predict methane production of forage-fed '
                    'cattle in Australia."',
        'applicable_animals': 'Beef and dairy cattle, Australia',
        'scope': '1034 individual animal records, forage-fed (diets above 70 % forage). '
                 'Not shown to apply to confinement, high-concentrate TMR systems.',
        'required_inputs': ['dmi_kg'],
        'optional_inputs': [],
        'equation': 'CH4 (g/d) = 20.7 x DMI(kg/d)',
        'units': 'g CH4/day',
        'parameter_sources': 'Charmley et al. (2016)',
        'assumptions': 'Methane yield per kg DMI is constant across diet/animal variation '
                        'within the calibration dataset.',
        'limitations': 'Empirical DMI-only relationship; does not respond to diet fat/oil '
                        'supplementation, 3-NOP, ionophores, or other methane mitigants.',
    },
    'ipcc_tier2': {
        'citation': 'IPCC (2019), 2019 Refinement to the 2006 IPCC Guidelines for National '
                    'Greenhouse Gas Inventories, Vol 4 Ch 10, Equation 10.21, Table 10.12.',
        'applicable_animals': 'Cattle, Tier 2 enteric fermentation method',
        'scope': 'Ym = 6.0 % is the Table 10.12 value for high-producing dairy cows '
                 '(above 8500 kg per year) on diets with NDF above 35 % of DM. GE is '
                 'derived from the herd-level ME and DE defaults, not measured diet GE.',
        'required_inputs': ['dmi_kg', 'DIET_ME_MCAL_PER_KG', 'DIET_DE_PCT', 'IPCC_TIER2_YM_PERCENT'],
        'optional_inputs': [],
        'equation': 'CH4 (kg/d) = GE_intake(MJ/d) x Ym/100 / 55.65',
        'units': 'g CH4/day',
        'parameter_sources': 'IPCC (2019) Table 10.12 (Ym); NRC (2001) Chapter 2 (ME = 0.82 x DE); '
                              'diet ME and DE values from the diet assumptions in '
                              'domains/cow/parameters.py.',
        'assumptions': 'ME = 0.82 x DE. This ratio comes from high-forage cattle; diets rich in '
                        'concentrate usually have a higher ratio (often above 0.90), so 0.82 '
                        'probably understates the true ME/DE for the assumed diet.',
        'limitations': 'GE is derived (ME / 0.82 / DE%), not measured. Ym is a single default: the '
                        'NDF-specific 5.7 % / 6.0 % split in Table 10.12 cannot be selected because '
                        'diet NDF is not tracked. ME/DE = 0.82 may understate the ratio for the '
                        'assumed concentrate level.',
    },
}

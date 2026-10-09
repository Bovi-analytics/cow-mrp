"""
cow/rewards/environment/enteric_methane.py
============================================
Enteric methane (rumen fermentation), g CH4 per model step. Two methods:

- charmley_2016: empirical DMI-based yield factor.
- ipcc_tier2: energy-balance method (GE intake x Ym / 55.65 MJ/kg CH4).

"cncps" is NOT implemented -- see the comment in the methods dict below
for exactly what inputs it would require and why they are unavailable.
"""

from cow.parameters import (
    UNIT_DAYS, DIET_ME_MCAL_PER_KG, DIET_DE_PCT, IPCC_TIER2_YM_PERCENT,
)

MCAL_TO_MJ = 4.184
ME_OVER_DE = 0.82              # CITATION REMOVED 2026-09-23 -- unverified, not independently
                                # checked (previously attributed to a 1965 source, carried
                                # through later editions; user will supply NASEM (2021) to
                                # re-source this). ME = 0.82 x DE.
                                # CAVEAT (retained, source-independent of the removed citation):
                                # derived from high-forage, maintenance-level-intake cattle;
                                # high-concentrate diets are reported elsewhere as needing a
                                # true ME/DE ratio typically >0.90. This repo's assumed diet
                                # (DIET_DE_PCT=73%, moderate-high concentrate) is closer to
                                # that higher-MDR regime, so 0.82 likely understates true ME/DE
                                # here -- kept as the long-standing default per this pass's
                                # "verify or label, do not invent a replacement" instruction.
CH4_ENERGY_MJ_PER_KG = 55.65   # VERIFIED exact: IPCC (2019) Vol 4 Ch 10, Equation 10.21
                                # ("The factor 55.65 (MJ/kg CH4) is the energy content of
                                # methane") -- confirmed against the primary PDF.


def _charmley_2016(states, context):
    """
    CH4 (g/d) = 20.7 x DMI(kg/d).

    Charmley E, Williams SRO, Moate PJ, Hegarty RS, Herd RM, Oddy VH,
    Reyenga P, Staunton KM, Anderson A, Hannah MC (2016). "A universal
    equation to predict methane production of forage-fed cattle in
    Australia." Animal Production Science 56:169-180.

    n=1034 individual animal records (NOT 2,909 -- that figure was not
    found in the primary source and appears to be a mischaracterization;
    corrected here per the scientific audit). Dataset is Australian
    forage-fed cattle (beef AND dairy, >70% forage diets) -- whether it
    applies to a high-concentrate, confinement TMR dairy model (as this
    repo's DMI/milk pipeline implies) is UNCONFIRMED. Use with that
    caveat, not as a validated TMR-dairy default.
    """
    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS
    ch4_g_per_day = 20.7 * dmi_kg_per_day
    return ch4_g_per_day * UNIT_DAYS


def _ipcc_tier2(states, context):
    """
    CH4 (kg/d) = GE_intake(MJ/d) x (Ym/100) / 55.65 MJ/kg CH4.

    IPCC (2019), 2019 Refinement to the 2006 IPCC Guidelines, Vol 4 Ch
    10, Equation 10.21, Tier 2 enteric fermentation method -- equation
    form and the 55.65 MJ/kg CH4 constant VERIFIED exact against the
    primary PDF this pass.

    Ym (methane conversion factor, % of GE lost as CH4) defaults to
    IPCC_TIER2_YM_PERCENT = 6.0. VERIFIED against the primary Table
    10.12 ("Cattle/Buffalo Methane Conversion Factors"): this is
    precisely the "High-producing cows (>8500 kg/head/yr), NDF>35% DM"
    row -- not a generic "typical" value as previously labeled. The full
    table has four dairy-cow tiers depending on production level and
    diet NDF: 5.7% (high-producing, NDF<=35%), 6.0% (high-producing,
    NDF>35%), 6.3% (medium-producing, 5000-8500 kg/yr), 6.5%
    (low-producing, <5000 kg/yr). This repo tracks neither diet NDF nor
    an explicit annual-yield-based production tier, so a single value
    (6.0%) is used and flagged rather than selected per cow.

    GE (gross energy) intake is NOT a tracked/observed value in this
    model -- it is DERIVED from DIET_ME_MCAL_PER_KG and DIET_DE_PCT via
    the standard approximation ME ~= 0.82 x DE, then GE = DE / (DE%/100).
    This derivation is an extra approximation layer beyond a
    "gold-standard" Tier 2 implementation using directly-measured diet
    GE -- documented here per the audit rather than presented as
    equivalent to a measured-GE Tier 2 estimate.
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
        # 'cncps': NOT implemented. CNCPS enteric methane requires
        # feed-library-level carbohydrate/protein fractionation (NDF,
        # non-fiber carbohydrate, starch degradation rate, rumen
        # passage rate) and a full rumen-fermentation submodel -- none
        # of which exist in this repo's production-class-level diet
        # representation (DIET_CP_PCT/DIET_ME_MCAL_PER_KG/DIET_DE_PCT
        # are herd-level averages, not feed-ingredient-level inputs).
        # Implementing CNCPS would require: (1) a feed-ingredient
        # library, (2) ration formulation per cow class, (3) a rumen
        # kinetics submodel -- none in scope here. This dict's structure
        # is ready to accept a 'cncps' entry once those inputs exist.
    },
}

ENTERIC_METHANE_METHOD_INFO = {
    'charmley_2016': {
        'citation': 'Charmley et al. (2016), Animal Production Science 56:169-180, '
                    '"A universal equation to predict methane production of forage-fed '
                    'cattle in Australia."',
        'applicable_animals': 'Beef and dairy cattle, Australia',
        'scope': 'n=1034 individual animal records, forage-fed (>70% forage diets). '
                 'NOT confirmed applicable to confinement/high-concentrate TMR systems.',
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
                    'Greenhouse Gas Inventories, Vol 4 Ch 10, Equation 10.21, Table 10.12 -- '
                    'equation, 55.65 constant, and Ym=6.0% VERIFIED exact against the primary PDF.',
        'applicable_animals': 'Cattle, Tier 2 enteric fermentation method',
        'scope': 'Ym=6.0% is Table 10.12\'s "high-producing dairy cows (>8500 kg/yr), '
                 'NDF>35% DM" row specifically -- not a generic default. GE is derived here '
                 'from herd-level ME/DE defaults, not measured diet GE.',
        'required_inputs': ['dmi_kg', 'DIET_ME_MCAL_PER_KG', 'DIET_DE_PCT', 'IPCC_TIER2_YM_PERCENT'],
        'optional_inputs': [],
        'equation': 'CH4 (kg/d) = GE_intake(MJ/d) x Ym/100 / 55.65',
        'units': 'g CH4/day',
        'parameter_sources': 'IPCC (2019) Table 10.12 (Ym, verified); (ME=0.82xDE, citation '
                              'removed 2026-09-23 -- unverified, see ME_OVER_DE below); '
                              'ME/DE concentration values from cow/parameters.py diet-composition '
                              'assumptions (not independently sourced).',
        'assumptions': 'ME ~= 0.82 x DE -- citation removed 2026-09-23, unverified pending '
                        're-sourcing (NASEM 2021 to be supplied); the retained caveat reports '
                        'high-concentrate diets as needing a true ratio '
                        '(often >0.90 there); this repo\'s assumed diet is closer to that regime, so '
                        '0.82 likely understates true ME/DE -- kept as the long-standing '
                        'default, not corrected to an unverified alternative.',
        'limitations': 'GE is derived (ME/0.82/DE%), not measured; Ym uses a single default '
                        '(NDF-specific 5.7/6.0% split from Table 10.12 not selectable -- no '
                        'NDF tracking in this repo); ME_OVER_DE=0.82 may understate ME/DE for '
                        'this repo\'s assumed concentrate level (see assumptions above).',
    },
}

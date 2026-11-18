"""
cow/rewards/environment/manure_methane.py
============================================
Manure methane via IPCC (2019) Tier 2: VS -> B0 -> MCF -> CH4. Kept
separate from enteric_methane.py: a different biological pathway (manure
decomposition, not rumen fermentation) with different inputs.

MCF (methane conversion factor) depends on both climate and management
system (IPCC 2019 Table 10.17), so there is no single defensible default.
MANURE_SYSTEM and MANURE_CLIMATE in the cow parameters are therefore unset by
default; this quantity (and N2O, which needs the same inputs) is skipped,
not silently defaulted, until both are configured (see
``_manure_system_configured`` in the environment package ``__init__``).
"""

from cowmrp.domains.cow.parameters import UNIT_DAYS, DIET_DE_PCT, DIET_ASH_PCT, MANURE_SYSTEM, MANURE_CLIMATE

B0_DAIRY_M3_CH4_PER_KG_VS = 0.24   # IPCC (2019) Table 10.16A (Updated), "Dairy cattle" row; the
                                    # same in all regions (non-dairy cattle vary 0.13-0.19).
CH4_KG_PER_M3 = 0.67               # IPCC (2019) Equation 10.23: conversion factor of m3 CH4 to
                                    # kg CH4.

# IPCC (2019) Table 10.17 methane conversion factors (MCF), percent.
#
# The table is richer than this file's (system, climate) layout:
# liquid/slurry and uncovered-lagoon MCFs vary by 10 climate sub-zones (not
# just cool/temperate/warm) and, for liquid/slurry, by storage retention
# time (1/3/4/6/12 months). One representative sub-zone and duration is
# used per cool/temperate/warm band:
#   - retention time: 6 months, the retention time IPCC itself uses for a
#     single representative value (Table 10.14, footnote 5: liquid-system
#     emission factors "are calculated from manure management systems with a
#     6 month retention time").
#   - climate sub-zone: the "...Moist" column of each band (Cool Temp Moist,
#     Warm Temp Moist, Tropical Moist).
MCF_TABLE = {
    # (system, climate): MCF percent
    ('liquid_slurry', 'cool'):            21.0,  # Table 10.17, "Liquid/Slurry... 6 Month" row:
    ('liquid_slurry', 'temperate'):       37.0,  # Cool Temp Moist / Warm Temp Moist /
    ('liquid_slurry', 'warm'):            80.0,  # Tropical Moist columns.
    ('solid_storage', 'cool'):             2.0,  # Table 10.17 "Solid storage" row (one value per
    ('solid_storage', 'temperate'):        4.0,  # climate band, no sub-zone split).
    ('solid_storage', 'warm'):             5.0,
    ('dry_lot', 'cool'):                   1.0,  # Table 10.17 "Dry lot" row.
    ('dry_lot', 'temperate'):              1.5,
    ('dry_lot', 'warm'):                   2.0,
    ('daily_spread', 'cool'):              0.1,  # Table 10.17 "Daily spread" row.
    ('daily_spread', 'temperate'):         0.5,
    ('daily_spread', 'warm'):              1.0,
    ('anaerobic_lagoon', 'cool'):         60.0,  # Table 10.17 "Uncovered anaerobic lagoon" row:
    ('anaerobic_lagoon', 'temperate'):    73.0,  # Cool Temp Moist / Warm Temp Moist /
    ('anaerobic_lagoon', 'warm'):         80.0,  # Tropical Moist columns.
    ('anaerobic_digester', 'cool'):       10.0,  # Table 10.17 gives six digester rows by leakage
    ('anaerobic_digester', 'temperate'):  10.0,  # and storage quality, not by climate. 10.0 % is
    ('anaerobic_digester', 'warm'):       10.0,  # "High leakage, low quality technology, low
                                                   # quality gastight storage technology", a
                                                   # poorly performing digester (see note below).
}
# Anaerobic digesters: the Table 10.17 range is 1.00 % (low leakage,
# high-quality gastight storage, best industrial technology) to 13.17 %
# (high leakage, low-quality technology, open storage, warm climate). The
# flat 10.0 % used here is near the poorly performing end, so it will
# overstate emissions for a well-run digester. Selecting digester quality
# would need an extra dimension in this table. The table also does not
# distinguish gross methane generation from methane captured, flared or used.


def _ipcc_2019(states, context):
    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS
    vs_kg_per_day = dmi_kg_per_day * (1 - DIET_DE_PCT / 100.0) * (1 - DIET_ASH_PCT / 100.0)

    key = (MANURE_SYSTEM, MANURE_CLIMATE)
    if key not in MCF_TABLE:
        raise ValueError(
            f"manure_methane 'ipcc_2019': no MCF entry for "
            f"(COW_MANURE_SYSTEM={MANURE_SYSTEM!r}, COW_MANURE_CLIMATE={MANURE_CLIMATE!r}). "
            f"Available: {sorted(MCF_TABLE.keys())}"
        )
    mcf_percent = MCF_TABLE[key]

    ch4_kg_per_day = vs_kg_per_day * B0_DAIRY_M3_CH4_PER_KG_VS * CH4_KG_PER_M3 * (mcf_percent / 100.0)
    return ch4_kg_per_day * UNIT_DAYS


manure_methane_spec = {
    'name': 'manure_ch4_kg', 'produces': 'manure_ch4_kg',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['DIET_DE_PCT', 'DIET_ASH_PCT', 'MANURE_SYSTEM', 'MANURE_CLIMATE'],
        'external_input': [], 'model_output': ['dmi_kg'], 'time': [],
    },
    'method': 'ipcc_2019', 'methods': {'ipcc_2019': _ipcc_2019},
}

MANURE_METHANE_METHOD_INFO = {
    'ipcc_2019': {
        'citation': 'IPCC (2019), 2019 Refinement to the 2006 IPCC Guidelines, Vol 4 Ch 10, '
                    'Eq 10.23, Tables 10.16A (B0) and 10.17 (MCF).',
        'applicable_animals': 'Dairy cattle',
        'scope': 'All manure management systems. MCF is climate-dependent per Table 10.17; '
                 'this file collapses the table\'s 10 climate sub-zones and (for liquid/slurry) '
                 '5 retention-time tiers down to 3 representative cool/temperate/warm values -- '
                 'see MCF_TABLE comments for exactly which sub-zone/duration each represents.',
        'required_inputs': ['dmi_kg', 'DIET_DE_PCT', 'DIET_ASH_PCT', 'MANURE_SYSTEM', 'MANURE_CLIMATE'],
        'optional_inputs': [],
        'equation': 'VS=DMI*(1-DE%/100)*(1-Ash%/100); CH4(kg/d)=VS*B0*0.67*MCF/100',
        'units': 'kg CH4/day',
        'parameter_sources': 'IPCC (2019) Tables 10.16A and 10.17.',
        'assumptions': 'B0 = 0.24 m3 CH4/kg VS (IPCC dairy default); VS approximated '
                        'from DE%/Ash%, not measured; liquid/slurry and lagoon MCFs use a '
                        '6-month retention / "Moist" sub-zone representative selection (see '
                        'MCF_TABLE comments); anaerobic digester MCF represents a poor-performing '
                        'digester specifically, not a typical one (see note below MCF_TABLE).',
        'limitations': 'Does not distinguish gross generation from captured or flared '
                        'methane for digesters.',
    },
}

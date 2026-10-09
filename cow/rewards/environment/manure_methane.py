"""
cow/rewards/environment/manure_methane.py
============================================
Manure methane via IPCC (2019) Tier 2: VS -> B0 -> MCF -> CH4. Kept
entirely separate from enteric_methane.py -- different biological
pathway (manure decomposition vs. rumen fermentation), different inputs.

MCF (methane conversion factor) is climate- AND management-system-
dependent per IPCC (2019) Table 10.17 -- there is no single scientifically
defensible default. cow/parameters.py's MANURE_SYSTEM/MANURE_CLIMATE are
therefore unset by default; this quantity (and N2O, which needs the same
inputs) is SKIPPED, not silently defaulted, until both are configured --
see cow/rewards/environment/__init__.py::_manure_system_configured().
"""

from cow.parameters import UNIT_DAYS, DIET_DE_PCT, DIET_ASH_PCT, MANURE_SYSTEM, MANURE_CLIMATE

B0_DAIRY_M3_CH4_PER_KG_VS = 0.24   # VERIFIED exact: IPCC (2019) Table 10.16A (Updated),
                                    # "Dairy cattle" row = 0.24, uniform across ALL regions
                                    # (unlike non-dairy cattle, which varies 0.13-0.19 by region).
CH4_KG_PER_M3 = 0.67               # VERIFIED exact: IPCC (2019) Equation 10.23's own constant
                                    # ("0.67 = conversion factor of m3 CH4 to kilograms CH4").

# IPCC (2019) Table 10.17 methane conversion factors (MCF), percent --
# RE-VERIFIED this pass directly against the primary PDF (previously
# only checked via a secondary search summary, which turned out to be
# imprecise for several rows -- see corrections below).
#
# Table 10.17's real structure is richer than this file's (system,
# climate) schema: liquid/slurry and uncovered-lagoon MCFs vary by BOTH
# 10 climate sub-zones (not just cool/temperate/warm) AND, for
# liquid/slurry, by storage RETENTION TIME (1/3/4/6/12 months) -- a
# third dimension this file deliberately does not add (per this pass's
# "do not expand the implementation" instruction). Instead, one
# representative sub-zone/duration is selected per cool/temperate/warm
# bucket and documented explicitly below, rather than silently
# collapsing the table:
#   - retention time: 6 months (matches IPCC's OWN choice when a single
#     representative value is needed -- Table 10.14's footnote 5 states
#     liquid-system emission factors "are calculated from manure
#     management systems with a 6 month retention time").
#   - climate sub-zone: the "...Moist" column in each of Cool/Temperate/
#     Warm (Cool Temp Moist / Warm Temp Moist / Tropical Moist), for
#     naming consistency across the three bands.
MCF_TABLE = {
    # (system, climate): MCF percent
    ('liquid_slurry', 'cool'):            21.0,  # CORRECTED: Table 10.17, "Liquid/Slurry...
    ('liquid_slurry', 'temperate'):       37.0,  # 6 Month" row, Cool Temp Moist / Warm Temp
    ('liquid_slurry', 'warm'):            80.0,  # Moist / Tropical Moist columns (was 10/35/65,
                                                   # not traceable to any real cell in the table).
    ('solid_storage', 'cool'):             2.0,  # CORRECTED: Table 10.17 "Solid storage" row
    ('solid_storage', 'temperate'):        4.0,  # (single value per climate band, no sub-zone
    ('solid_storage', 'warm'):             5.0,  # split) -- was 1.0/2.0/4.0, table says 2/4/5.
    ('dry_lot', 'cool'):                   1.0,  # CONFIRMED exact: Table 10.17 "Dry lot" row.
    ('dry_lot', 'temperate'):              1.5,
    ('dry_lot', 'warm'):                   2.0,
    ('daily_spread', 'cool'):              0.1,  # CONFIRMED exact: Table 10.17 "Daily spread" row.
    ('daily_spread', 'temperate'):         0.5,
    ('daily_spread', 'warm'):              1.0,
    ('anaerobic_lagoon', 'cool'):         60.0,  # CORRECTED: Table 10.17 "Uncovered anaerobic
    ('anaerobic_lagoon', 'temperate'):    73.0,  # lagoon" row, Cool/Warm Temp Moist and Tropical
    ('anaerobic_lagoon', 'warm'):         80.0,  # Moist columns (was 35/70/80).
    ('anaerobic_digester', 'cool'):       10.0,  # Table 10.17 gives SIX digester rows by
    ('anaerobic_digester', 'temperate'):  10.0,  # leakage/storage quality, not by climate (mostly
    ('anaerobic_digester', 'warm'):       10.0,  # flat). 10.0% exactly matches "High leakage, low
                                                   # quality technology, low quality gastight
                                                   # storage technology" -- kept, but see note below:
                                                   # this represents a POOR-PERFORMING digester, not
                                                   # a typical/average one.
}
# NOTE on anaerobic digesters: Table 10.17's real range for digesters is
# 1.00% (best: low leakage, high-quality gastight storage, best complete
# industrial technology) to 13.17% (worst: high leakage, low-quality
# technology, open storage, warm climate). This file's flat 10.0% sits
# near the poor-performing end, not a typical/average digester -- kept
# because it is an exact match to a real table row (not invented), but
# users modeling a well-run digester should expect this to overstate
# emissions; a config option to select digester quality would require
# expanding this table's schema, out of scope for this verification pass.
# This table also does NOT distinguish gross methane generation from
# methane captured/flared/utilized.


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
                    'Eq 10.23, Tables 10.16A (B0) and 10.17 (MCF) -- all VERIFIED exact '
                    'against the primary PDF this pass, except as noted below.',
        'applicable_animals': 'Dairy cattle',
        'scope': 'All manure management systems. MCF is climate-dependent per Table 10.17; '
                 'this file collapses the table\'s 10 climate sub-zones and (for liquid/slurry) '
                 '5 retention-time tiers down to 3 representative cool/temperate/warm values -- '
                 'see MCF_TABLE comments for exactly which sub-zone/duration each represents.',
        'required_inputs': ['dmi_kg', 'DIET_DE_PCT', 'DIET_ASH_PCT', 'MANURE_SYSTEM', 'MANURE_CLIMATE'],
        'optional_inputs': [],
        'equation': 'VS=DMI*(1-DE%/100)*(1-Ash%/100); CH4(kg/d)=VS*B0*0.67*MCF/100',
        'units': 'kg CH4/day',
        'parameter_sources': 'IPCC (2019) Tables 10.16A/10.17, verified against the primary PDF.',
        'assumptions': 'B0=0.24 m3 CH4/kg VS (VERIFIED exact dairy default); VS approximated '
                        'from DE%/Ash%, not measured; liquid/slurry and lagoon MCFs use a '
                        '6-month retention / "Moist" sub-zone representative selection (see '
                        'MCF_TABLE comments); anaerobic digester MCF represents a poor-performing '
                        'digester specifically, not a typical one (see note below MCF_TABLE).',
        'limitations': 'All MCF_TABLE values are now VERIFIED against the primary Table 10.17 '
                        '(dry_lot/daily_spread already matched exactly; liquid_slurry/'
                        'solid_storage/anaerobic_lagoon corrected this pass). Does not '
                        'distinguish gross generation from captured/flared methane for digesters.',
    },
}

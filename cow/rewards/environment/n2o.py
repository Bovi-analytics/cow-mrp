"""
cow/rewards/environment/n2o.py
=================================
N2O from manure management, split into IPCC (2019) pathways: direct,
indirect (volatilization), indirect (leaching/runoff), and total --
computed as three SEPARATE, separately-inspectable specs (per the
scientific audit's requirement not to fold pathways together silently).

The N2O-N -> N2O molecular-weight conversion (44/28) is applied exactly
once per pathway. Because it's a linear scalar, applying it per-pathway
and then summing is mathematically identical to summing N2O-N first and
converting once -- so this satisfies both "expose each pathway
separately in its final (N2O, not N2O-N) unit" and "never double-convert".

Requires the same manure system/climate configuration as
manure_methane.py -- see
cow/rewards/environment/__init__.py::_manure_system_configured().

All tables in this file were RE-VERIFIED against the primary IPCC PDFs
in a follow-up verification pass (Chapter 10 Table 10.21/10.22; Chapter
11 Table 11.3), correcting several values that had only been checked via
secondary search summaries previously. EF3/Frac_gas/Frac_leach are all
per-manure-management-SYSTEM (Chapter 10, storage-stage losses) --
distinct from Chapter 11's Frac_GASM/Frac_LEACH(H), which govern a
DIFFERENT, later stage of the N cycle (losses after manure is applied to
soil, "N2O Emissions from Managed Soils") and are NOT used here.
"""

from cow.parameters import UNIT_DAYS, MANURE_SYSTEM

N2O_OVER_N2ON = 44.0 / 28.0   # N2O-N -> N2O molecular-weight conversion, verified

# IPCC (2019) Table 10.21 (Updated): direct emission factors for direct
# N2O emissions from manure management, EF3, kg N2O-N (kg N excreted)^-1.
# VERIFIED against the primary PDF this pass -- corrections from the
# previous (search-summary-based) version are noted per row.
EF_DIRECT_TABLE = {
    'daily_spread':       0.000,   # CORRECTED (was 0.01): Table 10.21 states daily-spread
                                     # N2O during storage/treatment is assumed zero -- manure
                                     # is applied to land within 24h, so those emissions are
                                     # covered under Chapter 11 "Agricultural Soils" instead.
    'solid_storage':      0.010,   # CONFIRMED: Table 10.21 "Solid storage" (no
                                     # bulking-agent/additive modification) row.
    'dry_lot':            0.020,   # CONFIRMED: Table 10.21 "Dry lot" row, exact.
    'liquid_slurry':      0.005,   # CORRECTED (was 0.001): Table 10.21 "Liquid/Slurry, with
                                     # natural crust cover" (or "Cover") sub-row = 0.005;
                                     # "without natural crust cover" = 0. This file's schema
                                     # has no crust-cover dimension, so the "with cover"
                                     # variant is used as the representative choice
                                     # (uncovered/actively-agitated slurry would use 0 instead).
    'anaerobic_lagoon':   0.000,   # CONFIRMED: Table 10.21 "Uncovered anaerobic lagoon" = 0
                                     # (emissions considered negligible -- low nitrification/
                                     # denitrification potential in these systems).
    'anaerobic_digester': 0.0006,  # CORRECTED (was 0.000): Table 10.21 "Anaerobic digester"
                                     # row, exact value.
}

# IPCC (2019) Table 10.22 (Updated), "Dairy Cow" column specifically:
# Frac_GasMS (fraction of managed manure N lost as NH3/NOx during
# STORAGE) and Frac_LeachMS (fraction lost to leaching/runoff during
# STORAGE). VERIFIED against the primary PDF this pass. Where a system
# has sub-variants (liquid/slurry crust cover; solid storage bulking/
# additives), the base/"with cover" variant is used for consistency with
# EF_DIRECT_TABLE's own sub-variant choice above.
FRAC_GAS_TABLE = {
    'daily_spread':       0.07,   # CORRECTED (was 0.20): Table 10.22, Dairy Cow column.
    'solid_storage':      0.30,   # CORRECTED (was 0.25): base ("-") variant, Dairy Cow column.
    'dry_lot':             0.30,  # CONFIRMED: Dairy Cow column, exact.
    'liquid_slurry':       0.30,  # CORRECTED (was 0.35): "with natural crust cover" variant,
                                    # Dairy Cow column (0.10 for "with cover", 0.48 "without
                                    # natural crust cover" -- see Table 10.22 for the other two).
    'anaerobic_lagoon':    0.35,  # CORRECTED (was 0.45): Dairy Cow column, exact.
    'anaerobic_digester':  0.50,  # Table 10.22 gives only a RANGE for digesters (0.05-0.50),
                                    # no single point default -- 0.50 (top of range) kept as the
                                    # representative choice; relabeled from an implied point
                                    # value to an explicit "top of published range" choice.
}
FRAC_LEACH_TABLE = {
    'daily_spread':       0.000,   # IPCC (2019) Table 10.22, Dairy Cow "Frac_leach_MS" column.
    'solid_storage':      0.020,   # base ("-") variant.
    'dry_lot':             0.035,
    'liquid_slurry':      0.000,   # "with natural crust cover" variant (all liquid/slurry
                                     # sub-variants and uncovered lagoon are 0 in the primary
                                     # table -- leaching only applies where manure is
                                     # uncovered/exposed to runoff, per Table 10.22's own scope
                                     # note).
    'anaerobic_lagoon':    0.000,
    'anaerobic_digester':  0.000,
}
# NOTE: this file previously used a single global MANURE_FRAC_LEACH=0.30
# constant (in cow/parameters.py) applied to every system uniformly.
# VERIFIED this pass to be WRONG: the real per-system values above are
# all <=0.035 -- 0.30 does not correspond to any cell in Table 10.22 for
# manure-management-stage leaching (it may have been conflated with
# Chapter 11's Frac_LEACH(H)=0.24, a DIFFERENT, later-stage quantity for
# N already applied to soil -- see this file's module docstring). The
# scalar parameter has been removed from cow/parameters.py and replaced
# with this per-system table, matching how EF_DIRECT_TABLE/
# FRAC_GAS_TABLE already work.

EF4_VOLATILIZATION = 0.010    # VERIFIED exact: IPCC (2019) Chapter 11, Table 11.3, aggregated
                                # default, kg N2O-N (kg NH3-N + NOx-N volatilised)^-1.
                                # (Disaggregated by climate: wet=0.014, dry=0.005 -- not
                                # selectable here, no climate-zone-for-volatilization tracking.)
EF5_LEACHING = 0.011          # VERIFIED exact: IPCC (2019) Chapter 11, Table 11.3, aggregated
                                # default, kg N2O-N (kg N leached/runoff)^-1.


def _manure_n_kg_per_year(context):
    manure_n_g_per_day = context['manure_n_g'] / UNIT_DAYS
    return manure_n_g_per_day * 365.0 / 1000.0


def _direct(states, context):
    ef = EF_DIRECT_TABLE.get(MANURE_SYSTEM)
    if ef is None:
        raise ValueError(f"n2o 'ipcc_2019': no direct EF for COW_MANURE_SYSTEM={MANURE_SYSTEM!r}. "
                          f"Available: {sorted(EF_DIRECT_TABLE.keys())}")
    n_ex_kg_yr = _manure_n_kg_per_year(context)
    n2o_kg_yr = n_ex_kg_yr * ef * N2O_OVER_N2ON
    return (n2o_kg_yr / 365.0) * UNIT_DAYS


def _indirect_volatilization(states, context):
    frac_gas = FRAC_GAS_TABLE.get(MANURE_SYSTEM)
    if frac_gas is None:
        raise ValueError(f"n2o 'ipcc_2019': no Frac_gas for COW_MANURE_SYSTEM={MANURE_SYSTEM!r}. "
                          f"Available: {sorted(FRAC_GAS_TABLE.keys())}")
    n_ex_kg_yr = _manure_n_kg_per_year(context)
    n2o_kg_yr = n_ex_kg_yr * frac_gas * EF4_VOLATILIZATION * N2O_OVER_N2ON
    return (n2o_kg_yr / 365.0) * UNIT_DAYS


def _indirect_leaching(states, context):
    frac_leach = FRAC_LEACH_TABLE.get(MANURE_SYSTEM)
    if frac_leach is None:
        raise ValueError(f"n2o 'ipcc_2019': no Frac_leach for COW_MANURE_SYSTEM={MANURE_SYSTEM!r}. "
                          f"Available: {sorted(FRAC_LEACH_TABLE.keys())}")
    n_ex_kg_yr = _manure_n_kg_per_year(context)
    n2o_kg_yr = n_ex_kg_yr * frac_leach * EF5_LEACHING * N2O_OVER_N2ON
    return (n2o_kg_yr / 365.0) * UNIT_DAYS


n2o_direct_spec = {
    'name': 'n2o_direct_kg', 'produces': 'n2o_direct_kg',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['MANURE_SYSTEM'], 'external_input': [], 'model_output': ['manure_n_g'], 'time': [],
    },
    'method': 'ipcc_2019', 'methods': {'ipcc_2019': _direct},
}
n2o_indirect_volatilization_spec = {
    'name': 'n2o_indirect_volatilization_kg', 'produces': 'n2o_indirect_volatilization_kg',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['MANURE_SYSTEM'], 'external_input': [], 'model_output': ['manure_n_g'], 'time': [],
    },
    'method': 'ipcc_2019', 'methods': {'ipcc_2019': _indirect_volatilization},
}
n2o_indirect_leaching_spec = {
    'name': 'n2o_indirect_leaching_kg', 'produces': 'n2o_indirect_leaching_kg',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['MANURE_SYSTEM'], 'external_input': [], 'model_output': ['manure_n_g'], 'time': [],
    },
    'method': 'ipcc_2019', 'methods': {'ipcc_2019': _indirect_leaching},
}

N2O_METHOD_INFO = {
    'ipcc_2019': {
        'citation': 'IPCC (2019), 2019 Refinement to the 2006 IPCC Guidelines, Vol 4 Ch 10, '
                    'Eq 10.25-10.27, Table 10.21 (EF3) and Table 10.22 (Frac_gas/Frac_leach, '
                    'Dairy Cow column); EF4/EF5 from Chapter 11, Table 11.3 (aggregated '
                    'defaults) -- ALL VERIFIED exact against the primary PDFs this pass.',
        'applicable_animals': 'Dairy cattle manure management',
        'scope': 'Direct + indirect (volatilization, leaching) pathways during manure STORAGE '
                 '(not the later, separate "N2O from managed soils" stage after land '
                 'application, which uses different Chapter 11 fractions not modeled here).',
        'required_inputs': ['manure_n_g', 'MANURE_SYSTEM'],
        'optional_inputs': [],
        'equation': 'N2O(kg/yr) = Nex,yr * (EFdirect + Frac_gas*EF4 + Frac_leach*EF5) * 44/28, '
                    'pathways computed separately then summed',
        'units': 'kg N2O/day',
        'parameter_sources': 'IPCC (2019) Tables 10.21, 10.22, 11.3 -- verified against the '
                              'primary PDFs.',
        'assumptions': 'Liquid/slurry and solid-storage sub-variants (crust cover, bulking '
                        'agent, etc.) are collapsed to one representative variant per system '
                        '(see EF_DIRECT_TABLE/FRAC_GAS_TABLE/FRAC_LEACH_TABLE comments); '
                        'anaerobic digester Frac_gas uses the top of IPCC\'s published range '
                        '(0.05-0.50), no single point default exists in the primary table.',
        'limitations': 'All three per-system tables now verified against the primary source. '
                        'The single-representative-variant simplification (no crust-cover/'
                        'bulking-agent dimension) remains a real, documented simplification, '
                        'not an error.',
    },
}

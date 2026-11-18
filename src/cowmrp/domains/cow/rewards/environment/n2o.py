"""
cow/rewards/environment/n2o.py
=================================
N2O from manure management, split into IPCC (2019) pathways: direct,
indirect (volatilization), indirect (leaching/runoff), and total. Each
pathway is a separate spec, so each can be inspected on its own.

The N2O-N to N2O molecular-weight conversion (44/28) is applied exactly
once per pathway. It is a linear scalar, so converting each pathway and
then summing equals summing N2O-N first and converting once.

Requires the same manure system and climate configuration as
manure_methane.py (see ``_manure_system_configured`` in the environment
package ``__init__``).

Sources: IPCC (2019) Chapter 10 Table 10.21 (direct emission factor EF3)
and Table 10.22 (Frac_gas and Frac_leach, Dairy Cow column); Chapter 11
Table 11.3 (EF4 and EF5). EF3, Frac_gas and Frac_leach apply per manure
management system during storage. They are different from Chapter 11's
Frac_GASM and Frac_LEACH(H), which describe losses after manure is applied
to soil ("N2O emissions from managed soils") and are not used here.
"""

from cowmrp.domains.cow.parameters import UNIT_DAYS, MANURE_SYSTEM

N2O_OVER_N2ON = 44.0 / 28.0   # N2O-N -> N2O molecular-weight conversion

# IPCC (2019) Table 10.21 (Updated): direct emission factors for direct
# N2O emissions from manure management, EF3, kg N2O-N (kg N excreted)^-1.
EF_DIRECT_TABLE = {
    'daily_spread':       0.000,   # Table 10.21: N2O during storage is assumed zero for daily
                                     # spread (manure goes to land within 24 h), so those emissions
                                     # fall under Chapter 11 "Agricultural Soils".
    'solid_storage':      0.010,   # Table 10.21 "Solid storage" (no bulking agent or additive).
    'dry_lot':            0.020,   # Table 10.21 "Dry lot".
    'liquid_slurry':      0.005,   # Table 10.21 "Liquid/Slurry, with natural crust cover" = 0.005;
                                     # without a natural crust cover it is 0. The model has no
                                     # crust-cover dimension, so the "with cover" value is used.
    'anaerobic_lagoon':   0.000,   # Table 10.21 "Uncovered anaerobic lagoon" = 0 (emissions
                                     # considered negligible).
    'anaerobic_digester': 0.0006,  # Table 10.21 "Anaerobic digester".
}

# IPCC (2019) Table 10.22 (Updated), "Dairy Cow" column specifically:
# Frac_GasMS (fraction of managed manure N lost as NH3/NOx during
# STORAGE) and Frac_LeachMS (fraction lost to leaching/runoff during
# STORAGE). Where a system has sub-variants (liquid/slurry crust cover;
# solid storage bulking/additives), the base or "with cover" variant is
# used, consistent with EF_DIRECT_TABLE above.
FRAC_GAS_TABLE = {
    'daily_spread':       0.07,   # Table 10.22, Dairy Cow column.
    'solid_storage':      0.30,   # Base ("-") variant, Dairy Cow column.
    'dry_lot':             0.30,  # Dairy Cow column.
    'liquid_slurry':       0.30,  # "With natural crust cover" variant, Dairy Cow column (the
                                    # table also gives 0.10 "with cover" and 0.48 "without
                                    # natural crust cover").
    'anaerobic_lagoon':    0.35,  # Dairy Cow column.
    'anaerobic_digester':  0.50,  # Table 10.22 gives only a range for digesters (0.05-0.50) and
                                    # no point default; the top of the range is used.
}
FRAC_LEACH_TABLE = {
    'daily_spread':       0.000,   # IPCC (2019) Table 10.22, Dairy Cow "Frac_leach_MS" column.
    'solid_storage':      0.020,   # Base ("-") variant.
    'dry_lot':             0.035,
    'liquid_slurry':      0.000,   # "With natural crust cover" variant. All liquid/slurry
                                     # sub-variants and the uncovered lagoon are 0 in Table 10.22;
                                     # leaching applies only where manure is exposed to runoff.
    'anaerobic_lagoon':    0.000,
    'anaerobic_digester':  0.000,
}
# Leaching during storage is system-specific and small (at most 0.035 above).
# Chapter 11's Frac_LEACH(H) (0.24) is a different, later-stage quantity for N
# already applied to soil and is not used here.

EF4_VOLATILIZATION = 0.010    # IPCC (2019) Chapter 11, Table 11.3, aggregated default,
                                # kg N2O-N (kg NH3-N + NOx-N volatilised)^-1. The table also
                                # gives wet climates 0.014 and dry climates 0.005; they cannot
                                # be selected because climate is not tracked for volatilization.
EF5_LEACHING = 0.011          # IPCC (2019) Chapter 11, Table 11.3, aggregated default,
                                # kg N2O-N (kg N leached/runoff)^-1.


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
                    'defaults).',
        'applicable_animals': 'Dairy cattle manure management',
        'scope': 'Direct + indirect (volatilization, leaching) pathways during manure STORAGE '
                 '(not the later, separate "N2O from managed soils" stage after land '
                 'application, which uses different Chapter 11 fractions not modeled here).',
        'required_inputs': ['manure_n_g', 'MANURE_SYSTEM'],
        'optional_inputs': [],
        'equation': 'N2O(kg/yr) = Nex,yr * (EFdirect + Frac_gas*EF4 + Frac_leach*EF5) * 44/28, '
                    'pathways computed separately then summed',
        'units': 'kg N2O/day',
        'parameter_sources': 'IPCC (2019) Tables 10.21, 10.22 and 11.3.',
        'assumptions': 'Liquid/slurry and solid-storage sub-variants (crust cover, bulking '
                        'agent, etc.) are collapsed to one representative variant per system '
                        '(see EF_DIRECT_TABLE/FRAC_GAS_TABLE/FRAC_LEACH_TABLE comments); '
                        'anaerobic digester Frac_gas uses the top of IPCC\'s published range '
                        '(0.05-0.50), no single point default exists in the primary table.',
        'limitations': 'Each system is represented by one variant (no crust-cover or '
                        'bulking-agent dimension), which is a simplification of the IPCC tables.',
    },
}

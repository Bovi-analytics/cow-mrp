"""
cow/rewards/environment/nitrogen.py
====================================
Nitrogen partitioning: intake -> milk N / fecal N / urinary N -> manure N.

Biological quantities only (g N per model step) -- NOT rewards. See
reward_nitrogen.py for the $ transformation.

Milk yield and DMI are NOT recomputed here -- they are read from context
as 'dmi_kg'/'milk_yield_kg', computed once by
cow/rewards/environment/__init__.py from the repo's EXISTING pipeline
values (cow.states.milk_yield_kg, cow.rewards.base.dry_matter_intake_kg).
This keeps a cow's nitrogen calculations consistent with her feed cost
and milk income -- no separate, disagreeing DMI/yield model.

CP% (diet crude protein, % of DM) and Protein% have no observed state or
existing pipeline value in this repo (no diet-composition tracking
exists) -- DIET_CP_PCT/DIET_PROTEIN_PCT in cow/parameters.py are
explicit, labeled default assumptions (Holstein-typical TMR), not
measured or derived values.
"""

from cowmrp.domains.cow.parameters import UNIT_DAYS, DIET_CP_PCT, DIET_PROTEIN_PCT


# ============================================================
# Nitrogen intake
# ============================================================

def _ni_standard(states, context):
    """
    NI (g N/d) = DMI(kg/d) x 1000 x CP%/100 / 6.25.
    6.25 = 100/16 -- crude protein is defined as N x 6.25 (protein is
    ~16% N by mass). Standard, uncontroversial nutrition conversion --
    not diet- or breed-specific.
    """
    dmi_kg_per_day = context['dmi_kg'] / UNIT_DAYS
    ni_g_per_day = dmi_kg_per_day * 1000.0 * (DIET_CP_PCT / 100.0) / 6.25
    return ni_g_per_day * UNIT_DAYS


nitrogen_intake_spec = {
    'name': 'nitrogen_intake_g', 'produces': 'nitrogen_intake_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['DIET_CP_PCT'], 'external_input': [], 'model_output': ['dmi_kg'], 'time': [],
    },
    'method': 'standard', 'methods': {'standard': _ni_standard},
}


# ============================================================
# Fecal N / Urinary N -- po_valley_2026
# ============================================================
# Source: "Modelling Nitrogen Excretion in Dairy Cows: An Application to
# Farms in the Po Valley (Italy)", Animals 16(2):294 (2026),
# doi:10.3390/ani16020294. Calibrated on 10 commercial Holstein TMR farms in
# Lombardy (Brescia/Cremona), Italy, June 2021 to June 2023. The authors
# describe the model as "context-specific" and "tailored to Italian Holstein
# cows fed TMR", so it should not be read as a general default. It is the
# only regression-based, peer-reviewed fecal/urinary N split used here.

def _fecal_n_po_valley_2026(states, context):
    ni_g_per_day = context['nitrogen_intake_g'] / UNIT_DAYS
    fecal_g_per_day = 33.4 + 0.245 * ni_g_per_day
    return fecal_g_per_day * UNIT_DAYS


fecal_n_spec = {
    'name': 'fecal_n_g', 'produces': 'fecal_n_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': ['nitrogen_intake_g'], 'time': [],
    },
    'method': 'po_valley_2026', 'methods': {'po_valley_2026': _fecal_n_po_valley_2026},
}


def _urinary_n_po_valley_2026(states, context):
    ni_g_per_day = context['nitrogen_intake_g'] / UNIT_DAYS
    urinary_g_per_day = 15.2 + 0.275 * ni_g_per_day
    return urinary_g_per_day * UNIT_DAYS


urinary_n_spec = {
    'name': 'urinary_n_g', 'produces': 'urinary_n_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': ['nitrogen_intake_g'], 'time': [],
    },
    'method': 'po_valley_2026', 'methods': {'po_valley_2026': _urinary_n_po_valley_2026},
}


# ============================================================
# Milk N -- standard protein->N conversion. Kept as its own quantity so that
# N intake, N in milk, N in feces and N in urine are all distinguishable and
# milk N is not folded into another term.
# ============================================================

def _milk_n_standard(states, context):
    """
    Milk N (g N/d) = milk_yield(kg/d) x Protein%/100 x 1000 / 6.38.
    6.38 is the standard milk-protein-to-N conversion factor (distinct
    from the general 6.25 CP factor -- milk true protein is ~15.68% N
    by mass). Standard, textbook conversion.
    """
    milk_kg_per_day = context['milk_yield_kg'] / UNIT_DAYS
    milk_n_g_per_day = milk_kg_per_day * (DIET_PROTEIN_PCT / 100.0) * 1000.0 / 6.38
    return milk_n_g_per_day * UNIT_DAYS


milk_n_spec = {
    'name': 'milk_n_g', 'produces': 'milk_n_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': ['DIET_PROTEIN_PCT'], 'external_input': [], 'model_output': ['milk_yield_kg'], 'time': [],
    },
    'method': 'standard', 'methods': {'standard': _milk_n_standard},
}


# ============================================================
# Manure N = fecal + urinary (NOT + milk N -- milk N leaves via the
# milk, not the manure; manure N is not the same as total N intake or
# excretion).
# ============================================================

def _manure_n_sum(states, context):
    return context['fecal_n_g'] + context['urinary_n_g']


manure_n_spec = {
    'name': 'manure_n_g', 'produces': 'manure_n_g',
    'depends_on': {
        'state': [], 'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [],
        'model_output': ['fecal_n_g', 'urinary_n_g'], 'time': [],
    },
    'method': 'sum', 'methods': {'sum': _manure_n_sum},
}


NITROGEN_METHOD_INFO = {
    'po_valley_2026': {
        'citation': 'Modelling Nitrogen Excretion in Dairy Cows: An Application to Farms in '
                    'the Po Valley (Italy). Animals 16(2):294 (2026). doi:10.3390/ani16020294',
        'applicable_animals': 'Lactating Holstein dairy cows, TMR-fed',
        'scope': 'Calibrated on 10 commercial farms, Lombardy (Brescia/Cremona) Italy, '
                 'Jun 2021-Jun 2023 -- explicitly "context-specific" per the authors, '
                 'not presented as a universal equation.',
        'required_inputs': ['nitrogen_intake_g'],
        'optional_inputs': [],
        'equation': 'NexF(g/d)=33.4+0.245*NI; NexU(g/d)=15.2+0.275*NI (NI in g N/d)',
        'units': 'g N/day',
        'parameter_sources': 'None beyond nitrogen_intake_g',
        'assumptions': 'Linear N-partitioning response to intake across the calibration range.',
        'limitations': 'Regional/dietary calibration may not transfer to non-TMR, '
                        'non-Holstein, or non-Italian systems.',
    },
}

"""
cow/tables.py
=============
Everything that depends only on (lactation bucket, DIM) or DP is computed
ONCE here, before any state is generated, so the state/probability/
transition/reward code does array lookups only (no per-state curve
evaluation).

Contents of get_tables():
- milk_curve  (3, MAX_DIM+1): mean daily milk (kg) of an OPEN cow over the real
              days she spends in DIM step k, per lactation bucket (1, 2, 3+), averaged
              over the uniform entry offset (true age at step start = k*U + u, so the
              mean age during step k is (k+1)*U). Keeps day/week/month consistent.
- cull_p      (3, MAX_DIM+1): probability that an open cow of that lactation bucket is
              culled in DIM step k (milk-floor reproductive cull), spread over the
              steps around the crossing with the calving entry-offset rule so the mean
              cull age is the same at day / week / month. The floor is evaluated ONLY
              from LAST_BREEDING_DIM_DAYS on (same for every lactation) and only ever
              applied to open cows; the curve must be non-increasing past the cutoff.
- cull_dim    (3,): last step of the cull window (cull_p == 1 from there on).
- cull_day    (3,): real day at which the open-cow curve first reaches MILK_FLOOR_KG.
- dim_max, dp_max : largest DIM / DP that can be reached (see the comment in get_tables). They are the
              axes of the state space (capped at MAX_DIM / MAX_DP).
- entry_steps : DIM steps in which a conception can happen (pregnant dim - dp always equals one of them).
- preg_factor (MAX_LACTATION, MAX_DP+1): milk multiplier by DP from the pregnancy_milk_factor_full table
              (1 when open, declining while pregnant, 0 once dry; dry-off applied by day).
"""

from functools import lru_cache
import numpy as np
import pandas as pd

from cow.parameters import (
    UNIT_DAYS, MAX_DIM, MAX_DP, MAX_LACTATION, DRY_OFF_DP, DRY_OFF_DP_DAYS, LAST_BREEDING_DIM, LAST_BREEDING_DIM_DAYS, MILK_FLOOR_KG,
    VOLUNTARY_WAITING_PERIOD_DAYS,
    PREGNANCY_DECLINE_FLOOR, VOLUNTARY_WAITING_PERIOD_DAYS,
)
from cow.curves import curve_kg
from cow.lookup_tables import (
    age_offset_slots, threshold_crossing_mass, crossing_conditional, N_GEST_MONTHS, full_grid,
    register_full_table, get_full_table, table_to_array,
)


# ============================================================
# PREGNANCY MILK FACTOR -- full lookup table (lactation 1..MAX x month of gestation)
# ============================================================
# Multiplier on an open cow's milk once she is pregnant. Default = a LINEAR decline from 1 at conception to
# PREGNANCY_DECLINE_FLOOR (0.70) at dry-off, evaluated at each gestation month's midpoint -- an uncited modelling
# placeholder (docs/ASSUMPTIONS_REGISTER.md A-066), the same value for every lactation. NONE of the
# papers reviewed for this model gives the shape (the early linear decline is NOT validated; a smaller early effect is plausible but unsupported);
# replace with real values (per lactation if available) in cow/data/pregnancy_milk_factor_full.csv. Dry-off itself is applied by day (DRY_OFF_DP_DAYS).

def _build_pregnancy_milk_factor_full():
    g = full_grid(N_GEST_MONTHS)
    age_mid = g['month'].values * 30 + 15
    g['factor'] = 1 - (1 - PREGNANCY_DECLINE_FLOOR) * np.minimum(age_mid, DRY_OFF_DP_DAYS) / DRY_OFF_DP_DAYS
    g['basis'] = 'placeholder'
    g['note'] = ('linear decline 1 -> PREGNANCY_DECLINE_FLOOR at dry-off, evaluated at month midpoints; uncited modelling '
                 'placeholder, the same for every lactation; dry-off applied by day')
    return g


register_full_table('pregnancy_milk_factor_full', _build_pregnancy_milk_factor_full, N_GEST_MONTHS, value_col='factor')


def lactation_index(lactation):
    """0 .. MAX_LACTATION-1 for lactation 1 .. MAX_LACTATION (every lactation has its own row)."""
    return np.clip(np.asarray(lactation), 1, MAX_LACTATION) - 1


def _entry_ages(first_step, last_step, steps_before_first_age=0):
    """True ages (days) sampled across each step: step k starts at k*U + u (u ~ Uniform(0, U): the cow
    entered the phase at a uniform time inside the previous step) and she lives through U days of it.
    Shape (n_steps, n_u_slots, U). `steps_before_first_age` = 1 for gestation (dp = 1 is the first step
    AFTER the conception step, so its start age is u, not U + u)."""
    ks = np.arange(first_step, last_step + 1) - steps_before_first_age
    return (ks[:, None, None] * UNIT_DAYS + age_offset_slots()[None, :, None]
            + (np.arange(UNIT_DAYS)[None, None, :] + 0.5))


def _step_mean_curve(first_dim, last_dim):
    """Mean daily milk of an OPEN cow over the real days she spends in each DIM step, averaged over the
    uniform entry offset -- the SAME convention as the probability tables (cow/lookup_tables.py), so the
    mean age during step k is (k+1)*U and day / week / month describe the same lactation."""
    ages = _entry_ages(first_dim, last_dim)
    return np.stack([curve_kg(ages, l).mean(axis=(1, 2)) for l in range(1, MAX_LACTATION + 1)])


@lru_cache(maxsize=1)
def get_tables():
    curve = _step_mean_curve(0, MAX_DIM)

    # ---- milk-floor reproductive cull --------------------------------------------------------------
    # The floor is evaluated ONLY from LAST_BREEDING_DIM_DAYS on (never before; same for every lactation),
    # on the continuous open-cow curve. The first real day c at/after the cutoff with milk <= MILK_FLOOR_KG
    # is the crossing age. A cow crosses it at a uniform point inside a step, so the cull is spread over
    # the adjacent steps with the SAME entry-offset rule as calving (threshold_crossing_mass):
    # cull_p[bucket, k] = P(crossing inside step k | not yet crossed), 1.0 from the last step on.
    fine = np.arange(LAST_BREEDING_DIM_DAYS, (MAX_DIM + 2) * UNIT_DAYS) + 0.5
    cull_p = np.zeros((MAX_LACTATION, MAX_DIM + 1))
    cull_dim = np.zeros(MAX_LACTATION, dtype=int)
    cull_day = np.zeros(MAX_LACTATION)
    for i in range(MAX_LACTATION):
        y = curve_kg(fine, i + 1)
        if (np.diff(y) > 1e-12).any():
            raise ValueError(
                "The milk curve is still rising after LAST_BREEDING_DIM_DAYS -- the milk floor is only "
                "defined past the lactation peak. Raise LAST_BREEDING_DIM_DAYS (cow/parameters.py)."
            )
        below = np.where(y <= MILK_FLOOR_KG)[0]
        if len(below) == 0:
            raise ValueError(
                f"Lactation {i + 1}: milk never falls to MILK_FLOOR_KG={MILK_FLOOR_KG} kg/d "
                f"within MAX_DIM -- open cows would never leave. Raise MILK_FLOOR_KG or "
                f"COW_MAX_DIM_DAYS (cow/parameters.py)."
            )
        c = float(fine[below[0]])
        mass = threshold_crossing_mass(c)
        k_last = max(mass) - 1
        if k_last > MAX_DIM:
            raise ValueError(f"Lactation {i + 1}: milk-floor crossing (day {c:.0f}) is beyond MAX_DIM.")
        cull_day[i], cull_dim[i] = c, k_last
        for k in range(MAX_DIM + 1):
            cull_p[i, k] = crossing_conditional(mass, k + 1)

    # Reachable bounds. dp: calving has probability 1 from its last step on, so no pregnant cow is ever beyond
    # that step. dim: an open cow is culled by cull_dim; a pregnant cow conceived in an entry step c and loses
    # (aborts) or calves at most dp_max steps later, re-opening at dim <= c_max + dp_max + 1.
    from cow.probabilities.calving import active_calving_mass
    from cow.probabilities.conception import conception_entry_steps
    dp_max = int(min(MAX_DP, max(active_calving_mass())))
    entry_steps = conception_entry_steps()
    # last pregnant step in which a loss is possible under the ACTIVE abortion method (a loss re-opens the cow
    # one step later, at dim = c + dp + 1)
    import pandas as pd
    from mrp.engine import resolve_method
    from cow.probabilities.abortion import abortion_spec
    p_ab = resolve_method(abortion_spec)(pd.DataFrame({'lactation': 1, 'dim': 0, 'dp': np.arange(dp_max + 1)}), {})
    dp_abort_last = int(np.where(np.asarray(p_ab) > 1e-12)[0].max())
    dim_max = int(min(MAX_DIM, max(cull_dim.max(), entry_steps.max() + dp_abort_last + 1)))

    # Pregnancy milk factor by (lactation, dp): mean over the real gestation days of the step of the TABLE value
    # (native monthly, cow/data/pregnancy_milk_factor_full.csv overrides) -- 0 once the cow is dry (dry-off is
    # applied by DAY, DRY_OFF_DP_DAYS, like the waiting period); steps from DRY_OFF_DP on are the dry stage
    # (majority of the step past dry-off) and carry 0 milk. dp = 0 (open) is 1.
    gest_age = _entry_ages(1, MAX_DP, steps_before_first_age=1)           # (MAX_DP, S, U)
    month_of_age = np.minimum((gest_age // 30).astype(int), N_GEST_MONTHS - 1)
    fac = table_to_array(get_full_table('pregnancy_milk_factor_full'), N_GEST_MONTHS, value_col='factor')[0]   # (L, M)
    preg = np.ones((MAX_LACTATION, MAX_DP + 1))
    for li in range(MAX_LACTATION):
        f = np.where(gest_age < DRY_OFF_DP_DAYS, fac[li][month_of_age], 0.0)
        preg[li, 1:] = f.mean(axis=(1, 2))
    preg[:, np.arange(MAX_DP + 1) >= DRY_OFF_DP] = 0.0

    # Share of an open cow's days in each DIM step that fall inside the breeding window (after the voluntary
    # waiting period, up to the farmer's last breeding day) and after the waiting period -- same entry-offset
    # averaging as everything else. Used by reproduction_cost.
    ages = _entry_ages(0, MAX_DIM)
    breed_frac = ((ages > VOLUNTARY_WAITING_PERIOD_DAYS) & (ages <= LAST_BREEDING_DIM_DAYS)).mean(axis=(1, 2))
    after_vwp_frac = (ages > VOLUNTARY_WAITING_PERIOD_DAYS).mean(axis=(1, 2))

    return {"milk_curve": curve, "breed_frac": breed_frac, "after_vwp_frac": after_vwp_frac, "cull_p": cull_p, "cull_dim": cull_dim, "cull_day": cull_day,
            "dim_max": dim_max, "dp_max": dp_max, "entry_steps": entry_steps, "preg_factor": preg}


def state_milk_kg(dim, lactation, dp):
    """Milk (kg/day) for state arrays: precomputed curve x pregnancy factor."""
    t = get_tables()
    li = lactation_index(lactation)
    d = np.clip(np.asarray(dim), 0, MAX_DIM)
    p = np.clip(np.asarray(dp), 0, MAX_DP)
    return t["milk_curve"][li, d] * t["preg_factor"][li, p]


def export_tables(out_dir):
    """Write milk_table.csv and repro_cull_table.csv (the visible record of the precompute)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    t = get_tables()
    from cow.parameters import MILK_CURVE_BASIS, MILK_CURVE_SOURCE_LACTATION, MILK_CURVE_METHOD
    rows = []
    for i in range(MAX_LACTATION):
        for k in range(MAX_DIM + 1):
            rows.append({"lactation": i + 1, "dim_step": k, "dim_days_start": k * UNIT_DAYS,
                         "milk_kg_per_day": t["milk_curve"][i, k], "basis": MILK_CURVE_BASIS[MILK_CURVE_METHOD][i + 1],
                         "source_lactation": MILK_CURVE_SOURCE_LACTATION[i + 1]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "milk_table.csv"), index=False)
    rows = []
    for i in range(MAX_LACTATION):
        for k in range(MAX_DIM + 1):
            rows.append({"lactation": i + 1, "dim_step": k, "dim_days_start": k * UNIT_DAYS,
                         "cull_probability_if_open": t["cull_p"][i, k]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "repro_cull_by_step.csv"), index=False)
    rows = []
    for i in range(MAX_LACTATION):
        rows.append({"lactation": i + 1, "last_breeding_dim_days": LAST_BREEDING_DIM_DAYS,
                     "milk_floor_kg": MILK_FLOOR_KG, "floor_crossing_day": t["cull_day"][i],
                     "first_step_with_cull": int(np.argmax(t["cull_p"][i] > 0)),
                     "last_step_of_cull_window": int(t["cull_dim"][i])})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "repro_cull_table.csv"), index=False)
    return out_dir

"""
cow/rewards/reproduction_cost.py
=================================
Breeding costs + days-open penalty + abortion.

The days-open penalty below is a constant per-period marginal charge
(not a cumulative "total days open so far" amount). DAYS_OPEN_PENALTY_
PER_DAY (cow/rewards/base.py) is calibrated to De Vries (2006) assuming
this per-period formula -- the two must not be changed independently of
each other (see docs/REWARD_ECONOMICS_REVIEW_2026-08.md for history).
"""
import numpy as np
from cow.parameters import UNIT_DAYS, RATE_PERIOD_DAYS
from cow.rewards.base import AI_COST, DIAGNOSIS_COST, SYNCHRONIZATION_COST, DAYS_OPEN_PENALTY_PER_DAY, VOLUNTARY_WAITING_PERIOD_DAYS, ABORTION_COST

# UNIT CONSISTENCY (2026-09-20, L-041). AI + diagnosis + synchronization is a per-SERVICE (event) cost. The model
# has no explicit service process; the pre-existing implicit assumption is ONE service per native 30-day rate
# period while an Open cow is past the VWP (it was charged once per step, which is one per period only at
# MONTH). The event cost itself is NOT scaled; the EXPECTED NUMBER OF SERVICES PER STEP is made an explicit
# rate: BREEDING_SERVICES_PER_RATE_PERIOD * UNIT_DAYS / RATE_PERIOD_DAYS (exactly 1 at MONTH, 7/30 at WEEK).
# 1.0 is the existing implicit value, NOT a sourced service frequency (a ~21-day service interval would give
# ~1.4 per 30 d); it is registered as an assumption (docs/ASSUMPTIONS_REGISTER.md A-059). The days-open penalty
# (per day) and the abortion cost (per event x per-step probability) were already unit-consistent.
BREEDING_SERVICES_PER_RATE_PERIOD = 1.0


def reproduction_cost_formula(states, probs):
    dim = states['dim'].values
    dp = states['dp'].values
    from cow.tables import get_tables
    from cow.parameters import MAX_DIM
    tab = get_tables()
    k = np.clip(dim, 0, MAX_DIM)
    # Expected time an OPEN cow spends in this step, in the breeding window (after the waiting period, up to
    # the last breeding day) and, for the days-open penalty, after the waiting period:
    #   days = U * (share of the step's true ages in the window, entry-offset averaged: cow/tables.py)
    #          * (expected share of the step she is still open).
    # A cow that conceives / is culled / dies at a uniform time inside the step is open for only part of it:
    # expected open share = (1 + P(stay open this step)) / 2  (trapezoid over the step). Without this factor
    # every open step is charged in full and the charge grows with the step length (see docs/MODEL_CHANGELOG.md
    # 2026-10-08 (3), E3: variants compared against the DAY-resolution event count).
    survive = (1 - probs['p_dead']) * (1 - probs['p_cull'])
    open_share = (1 + survive * (1 - probs['p_preg'])) / 2.0
    services_per_day = BREEDING_SERVICES_PER_RATE_PERIOD / RATE_PERIOD_DAYS
    breeding_cost = np.where(dp == 0, (AI_COST + DIAGNOSIS_COST + SYNCHRONIZATION_COST) * services_per_day
                             * UNIT_DAYS * tab['breed_frac'][k] * open_share, 0.0)
    days_open_penalty = np.where(dp == 0, DAYS_OPEN_PENALTY_PER_DAY * UNIT_DAYS * tab['after_vwp_frac'][k] * open_share, 0.0)
    abortion = np.where(dp > 0, survive * probs['p_abort'] * ABORTION_COST, 0.0)   # event happens only if she survives the step (as in the transition)
    return breeding_cost + days_open_penalty + abortion

reproduction_cost_spec = {
    'name': 'reproduction_cost', 'sign': -1,
    'depends_on': {
        'state': ['dim', 'dp'],
        'derived_state': [], 'probability_output': ['p_dead', 'p_cull', 'p_preg', 'p_abort'], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': reproduction_cost_formula, 'model': None},
}

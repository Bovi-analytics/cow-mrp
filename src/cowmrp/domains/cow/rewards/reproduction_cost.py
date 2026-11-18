"""
cow/rewards/reproduction_cost.py
=================================
Breeding costs + days-open penalty + abortion.

The days-open penalty below is a constant per-period marginal charge
(not a cumulative "total days open so far" amount). DAYS_OPEN_PENALTY_
PER_DAY (cow/rewards/base.py) is calibrated to De Vries (2006) assuming
this per-period formula -- the two must not be changed independently of
each other.
"""
import numpy as np
from cowmrp.domains.cow.parameters import UNIT_DAYS, RATE_PERIOD_DAYS
from cowmrp.domains.cow.rewards.base import AI_COST, DIAGNOSIS_COST, SYNCHRONIZATION_COST, DAYS_OPEN_PENALTY_PER_DAY, VOLUNTARY_WAITING_PERIOD_DAYS, ABORTION_COST

# UNIT CONSISTENCY. AI + diagnosis + synchronization is a per-SERVICE (event) cost. The model has no explicit
# service process; it assumes ONE service per native 30-day rate period while an open cow is past the VWP. The
# event cost itself is not scaled; the expected number of services per step is an explicit rate:
# BREEDING_SERVICES_PER_RATE_PERIOD * UNIT_DAYS / RATE_PERIOD_DAYS (exactly 1 at MONTH, 7/30 at WEEK).
# 1.0 is an assumption, not a sourced service frequency (a ~21-day service interval would give ~1.4 per 30 d).
# The days-open penalty (per day) and the abortion cost (per event x per-step probability) are unit-consistent.
BREEDING_SERVICES_PER_RATE_PERIOD = 1.0


def reproduction_cost_formula(states, probs):
    dim = states['dim'].values
    dp = states['dp'].values
    from cowmrp.domains.cow.tables import get_tables
    from cowmrp.domains.cow.parameters import MAX_DIM
    tab = get_tables()
    k = np.clip(dim, 0, MAX_DIM)
    # Expected time an OPEN cow spends in this step, in the breeding window (after the waiting period, up to
    # the last breeding day) and, for the days-open penalty, after the waiting period:
    #   days = U * (share of the step's true ages in the window, entry-offset averaged: cow/tables.py)
    #          * (expected share of the step she is still open).
    # A cow that conceives / is culled / dies at a uniform time inside the step is open for only part of it:
    # expected open share = (1 + P(stay open this step)) / 2  (trapezoid over the step). Without this factor
    # every open step is charged in full and the charge grows with the step length.
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

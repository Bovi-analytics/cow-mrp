"""
youngstock/probabilities/birth_outcomes.py
===========================================
The OFFSPRING FATE TREE at calving. For one calving EVENT, by dam parity
('1' first calving, '2' second, '3+' later), returns the expected number of
calves landing in each youngstock state:

    calving event (dam parity p)
      -> split singleton (1-twinning[p] events x 1 calf) vs twin (twinning[p] events x 2 calves)
      -> semen type       = BREEDING_PROGRAMS[program][p]          (share w)
      -> stillborn?       = STILLBIRTH[p, semen] for the singleton stream;
                            min(1, STILLBIRTH[p, semen] * STILLBIRTH_TWIN_RATIO[p]) for the twin
                            stream (ratio-based, see youngstock/parameters.py) -> STILLBORN
      -> live-calf sex    = FEMALE_FRACTION[semen], applied to the RECOMBINED live pool (singleton
                            and twin survivors are no longer distinguished past this point -- neither
                            sex split nor terminal-line routing depends on twin-vs-singleton origin,
                            and twin same-sex correlation is provably inert here, see
                            STILLBIRTH_TWIN_RATIO's docstring)
            dairy semen : female -> DAIRY_F (kept: FEMALE_DAIRY_RETENTION; rest SOLD at birth;
                          kept calves that die inside the birth step -> DIED)
                          male   -> DAIRY_M   (terminal)
            beef semen  : female -> BEEF_F, male -> BEEF_M (terminal)
      -> terminal calf    : dies before sale with TERMINAL_PRESALE_MORTALITY -> DIED (booked at birth);
                            survivors: TERMINAL_CARRY share are still on the farm at the first
                            step boundary (-> CALF age 0, sold next step) and the rest were sold
                            inside the birth step (-> SOLD directly)

The result is a BIRTH-FLOW MIX, not a transition-matrix row: it is the placement
in  y_{t+1} = y_t Q + sum_p (calving events of parity p) x mix_p.  Vectors have
one entry per youngstock state; each sums to calves per event (>= 1), which is
why births live OUTSIDE the row-stochastic Q.

Parity structure: three explicit buckets. Sources that distinguish only first
vs later calvings give parities '2' and '3+' the SAME value (stated in
youngstock/parameters.py beside each entry) -- the code path is per parity so
a future parity-specific source drops in without structural change.

Every table is rebuilt from cowmrp.domains.youngstock.parameters at CALL time, so a scenario
change (BREEDING_PROGRAM, FEMALE_DAIRY_RETENTION) takes effect without a reimport.

Methods: 'table' (active); 'formula'/'model' None (no defensible closed-form
fate model -- selecting them raises via mrp_engine.resolve_method).
"""

import numpy as np
import pandas as pd

from cowmrp.mrp.core import lookup, find_state_id
from cowmrp.mrp.engine import validate_lookup_coverage
import cowmrp.domains.youngstock.parameters as YP
from cowmrp.domains.youngstock.probabilities.mortality import birth_step_survival


def _tables():
    """Lookup tables (key columns + 'p') from the CURRENT parameters."""
    program = YP.BREEDING_PROGRAMS[YP.BREEDING_PROGRAM]
    return dict(
        semen_share=pd.DataFrame([{'dam_parity': par, 'semen': s, 'p': w} for par, mix in program.items() for s, w in mix.items()]),
        stillbirth=pd.DataFrame([{'dam_parity': par, 'semen': s, 'p': v} for (par, s), v in YP.STILLBIRTH.items()]),
        twinning=pd.DataFrame([{'dam_parity': par, 'p': v} for par, v in YP.TWINNING.items()]),
        female_fraction=pd.DataFrame([{'semen': s, 'p': v} for s, v in YP.FEMALE_FRACTION.items()]),
    )


def _get(table, **keys):
    return float(lookup(table, {k: np.array([v]) for k, v in keys.items()})[0])


def _birth_mix_for_parity(states, parity, T):
    for name, cols in (('semen_share', ['dam_parity', 'semen', 'p']), ('stillbirth', ['dam_parity', 'semen', 'p']),
                       ('twinning', ['dam_parity', 'p']), ('female_fraction', ['semen', 'p'])):
        validate_lookup_coverage(T[name], cols, 'birth_outcomes')
    retention = YP.FEMALE_DAIRY_RETENTION
    if not 0.0 <= retention <= 1.0:
        raise ValueError(f"FEMALE_DAIRY_RETENTION must be in [0, 1], got {retention}")

    mix = np.zeros(len(states))
    # Split calves_per_event (= 1 + twin_rate) into a singleton stream (1-twin_rate events x 1 calf)
    # and a twin stream (twin_rate events x 2 calves), each with its OWN stillbirth rate -- see
    # STILLBIRTH_TWIN_RATIO's docstring in youngstock/parameters.py for why. The two
    # streams are recombined into one 'live' pool before sex/line routing, since neither the sex split
    # (FEMALE_FRACTION) nor the terminal-line routing depends on twin-vs-singleton origin.
    twin_rate = _get(T['twinning'], dam_parity=parity)
    singleton_calves_per_event = 1.0 - twin_rate
    twin_calves_per_event = twin_rate * 2.0
    twin_still_ratio = YP.STILLBIRTH_TWIN_RATIO[parity]
    shares = T['semen_share'][T['semen_share']['dam_parity'] == parity]
    if abs(shares['p'].sum() - 1.0) > 1e-9:
        raise ValueError(f"BREEDING_PROGRAM {YP.BREEDING_PROGRAM!r}: semen shares for parity {parity!r} "
                         f"sum to {shares['p'].sum()}, expected 1.0")

    sid = lambda line, stage, **kw: int(find_state_id(states, line=line, stage=stage, **kw))

    def book_terminal(line, calves):
        """Terminal calves of `line`: pre-sale death, immediate sale, or still on the farm at the boundary."""
        died = calves * YP.TERMINAL_PRESALE_MORTALITY[line]
        alive = calves - died
        carry = YP.TERMINAL_CARRY[line]
        mix[sid(line, 'DIED')] += died
        mix[sid(line, 'SOLD')] += alive * (1.0 - carry)
        mix[sid(line, 'CALF', age=0)] += alive * carry

    for _, row in shares.iterrows():
        semen, w = row['semen'], row['p']
        still = _get(T['stillbirth'], dam_parity=parity, semen=semen)
        still_twin = min(1.0, still * twin_still_ratio)
        live_singleton = singleton_calves_per_event * w * (1.0 - still)
        live_twin = twin_calves_per_event * w * (1.0 - still_twin)
        live = live_singleton + live_twin
        f = _get(T['female_fraction'], semen=semen)

        mix[sid('ALL', 'STILLBORN')] += singleton_calves_per_event * w * still + twin_calves_per_event * w * still_twin
        if semen in ('conventional', 'sexed'):
            kept = live * f * retention
            s0 = birth_step_survival()                                   # share alive at the first boundary
            mix[sid('DAIRY_F', 'CALF', age=0)] += kept * s0
            mix[sid('DAIRY_F', 'DIED')] += kept * (1.0 - s0)             # calves that died inside the birth step
            mix[sid('DAIRY_F', 'SOLD')] += live * f - kept              # surplus females sold at birth
            book_terminal('DAIRY_M', live * (1.0 - f))
        else:  # beef semen -> terminal beef-cross calves
            book_terminal('BEEF_F', live * f)
            book_terminal('BEEF_M', live * (1.0 - f))
    return mix


def birth_outcomes_table(states, context):
    """{'1': mix, '2': mix, '3+': mix}; each vector sums to calves per calving event."""
    T = _tables()
    return {par: _birth_mix_for_parity(states, par, T) for par in YP.PARITY_BUCKETS}


birth_outcomes_spec = {
    'name': 'birth_outcomes',
    'depends_on': ['line', 'stage', 'age'],
    'method': 'table',
    'methods': {'table': birth_outcomes_table, 'formula': None, 'model': None},
}

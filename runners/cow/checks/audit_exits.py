"""
audit_exits.py
==============
Exit decomposition: for a replacement-to-replacement cycle, every way a cow leaves each
lactation 1..MAX_LACTATION -- death, age/stage ("other") culling, milk-floor reproductive cull,
calving at the maximum lactation -- as (a) the share of cows ENTERING the lactation that leave in it
and (b) an annualized rate per cow-year, next to the published reference values.

    python runners/cow/checks/audit_exits.py [--unit week]        (default week; COW_TIME_UNIT also honoured)

Reference values (published mortality and culling studies):
  Miller 2008      death frequency per lactation, parity 1..>=8: 2.05 2.66 3.72 4.38 4.83 5.78 5.92 6.40 %
  Pinedo 2010 T1   annualized death rate  parity 1,2,3,>=4: 3.9 5.6 8.5 11.7 %/yr
                   annualized live-cull   parity 1,2,3,>=4: 16.9 23.3 30.1 40.8 %/yr
  Pinedo 2010 T1   pregnant vs open: live culling 7.2 vs 40.4 %/yr, death 3.1 vs 11.1 %/yr
These are reference points, never tuning targets (V is an output).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if "--unit" in sys.argv:
    os.environ["COW_TIME_UNIT"] = sys.argv[sys.argv.index("--unit") + 1]
elif "COW_TIME_UNIT" not in os.environ:
    os.environ["COW_TIME_UNIT"] = "week"

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import cowmrp.domains.cow.parameters as P
from cowmrp.mrp import core as mc
from cowmrp.domains.cow.states import build_cow_states
from cowmrp.domains.cow.probabilities import compute_probabilities
from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
import cowmrp.domains.cow.probabilities.culling as C

MILLER_DEATH = [2.05, 2.66, 3.72, 4.38, 4.83, 5.78, 5.92, 6.40, 6.40, 6.40]
PINEDO_DEATH = [3.9, 5.6, 8.5, 11.7] + [11.7] * 6
PINEDO_CULL = [16.9, 23.3, 30.1, 40.8] + [40.8] * 6


def exit_table():
    states = build_cow_states()
    U = P.UNIT_DAYS
    probs = compute_probabilities(states)
    r, c, v = compute_transitions(states, probs)
    M = mc.assemble_matrix(states, r, c, v).tocsr()
    rid = get_replacement_state_id(states)
    n = len(states)
    Q = M.copy().tolil()
    Q[:, rid] = 0
    e = np.zeros(n)
    e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsr().tocsc()).T.tocsc(), e)   # expected visits per cycle
    lac, dim, dp = (states[k].values for k in ("lactation", "dim", "dp"))
    pd_ = probs["p_dead"]
    pr = C.p_cull_repro(states, {})
    pa = C.p_culling_age_stage_full_table(states, {})
    survive = (1 - pd_) * (1 - probs["p_cull"])
    f_death = x * pd_
    f_age = x * (1 - pd_) * pa * (1 - pr)
    f_repro = x * (1 - pd_) * pr
    f_max = x * survive * (lac >= P.MAX_LACTATION) * (1 - probs["p_abort"]) * probs["p_calve"]
    rows = []
    for L in range(1, P.MAX_LACTATION + 1):
        m = lac == L
        entry = x[(lac == L) & (dim == 0) & (dp == 0)].sum()
        cow_years = x[m].sum() * U / 365.0
        if entry < 1e-12:
            rows.append((L, entry, cow_years, 0, 0, 0, 0))
            continue
        rows.append((L, entry, cow_years, f_death[m].sum() / entry, f_age[m].sum() / entry,
                     f_repro[m].sum() / entry, f_max[m].sum() / entry))
    return rows, x, states, (f_death, f_age, f_repro, f_max), U


def main():
    rows, x, states, flows, U = exit_table()
    print(f"unit={P.TIME_UNIT}   (shares are per cow ENTERING the lactation; rates per cow-year)")
    print(f"{'lac':>3} {'enter/cyc':>9} {'cowyr':>6} | {'death%':>6} {'(Miller)':>8} | {'cull_other%':>11} {'cull_milkfloor/repro%':>21} {'maxlac%':>7} | "
          f"{'death %/yr':>10} {'(Pinedo)':>8} | {'cull %/yr':>9} {'(Pinedo)':>8}")
    for (L, entry, cy, d, a, rp, mx) in rows:
        if entry < 1e-12:
            print(f"{L:>3} {entry:9.3f}   (not reached)")
            continue
        dr = d * entry / cy * 100
        cr = (a + rp) * entry / cy * 100
        print(f"{L:>3} {entry:9.3f} {cy:6.2f} | {d*100:6.2f} {MILLER_DEATH[L-1]:8.2f} | {a*100:11.2f} {rp*100:21.2f} {mx*100:7.2f} | "
              f"{dr:10.2f} {PINEDO_DEATH[L-1]:8.1f} | {cr:9.2f} {PINEDO_CULL[L-1]:8.1f}")
    tot_cy = sum(r[2] for r in rows)
    fd, fa, fr, fm = (f.sum() for f in flows)
    print(f"\nwhole cycle: cow-years={tot_cy:.2f}  death={fd:.3f} other-cull={fa:.3f} milk-floor/repro cull={fr:.3f} max-lactation={fm:.3f}")
    print(f"annualized over the herd: death {fd/tot_cy*100:.1f} %/yr   live cull {(fa+fr+fm)/tot_cy*100:.1f} %/yr "
          f"(Pinedo 2010 all herds: 6.6 / 25.1 ; Miller 2008: 5.7 death/cow-year)")
    pregm = (states['dp'].values > 0)
    d_open = flows[0][~pregm].sum() / x[~pregm].sum() / U * 365 * 100
    d_preg = flows[0][pregm].sum() / x[pregm].sum() / U * 365 * 100
    c_open = (flows[1] + flows[2])[~pregm].sum() / x[~pregm].sum() / U * 365 * 100
    c_preg = (flows[1] + flows[2])[pregm].sum() / x[pregm].sum() / U * 365 * 100
    print(f"open vs pregnant (annualized): death {d_open:.1f} vs {d_preg:.1f} %/yr (Pinedo 11.1 vs 3.1); "
          f"cull {c_open:.1f} vs {c_preg:.1f} %/yr (Pinedo 40.4 vs 7.2)")


if __name__ == "__main__":
    main()

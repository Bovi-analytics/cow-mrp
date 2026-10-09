"""
audit_repro.py
==============
Emergent reproductive outputs by lactation (audit P3 / P5): the model's inputs are conception, pregnancy-loss and
calving probabilities; "pregnant by DIM d" is an OUTPUT. This prints, for lactations 1, 2, 3, 5, 8 (cows that start the
lactation at DIM 0):
  - cumulative FIRST conception by DIM d, among all cows entering the lactation (cows that die/are culled first never conceive)
  - P(pregnant | still in the herd, in lactation, at DIM d)   (state occupancy)
  - expected calving interval of cows that calve again, days open (calving to the conception that reaches calving)
  - pregnancy-loss share of conceptions
Reference points (never tuning targets):
  Cook & Green 2016 (UK, 33 herds picked for POOR reproduction): 30 % conceived by 100 DIM, 46 % by 150 DIM, 65 % ever;
      overall conception rate per insemination 27.5 %; odds of pregnancy by 150 DIM: parity 2 = 0.68, >=3 = 0.47 (vs parity 1)
  CDCB aggregate used in A-033: days open ~137 d.

    python verify/audit_repro.py [--unit week]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "--unit" in sys.argv:
    os.environ["COW_TIME_UNIT"] = sys.argv[sys.argv.index("--unit") + 1]
elif "COW_TIME_UNIT" not in os.environ:
    os.environ["COW_TIME_UNIT"] = "week"

import numpy as np
import scipy.sparse as sp
import cow.parameters as P
from mrp import core as mc
from cow.states import build_cow_states
from cow.probabilities import compute_probabilities
from cow.transitions import compute_transitions, get_replacement_state_id


def compute(lactations=(1, 2, 3, 5, 8), checkpoints=(100, 150, 200, 300)):
    """{lactation: dict(cum_conc{d}, preg{d}, days_open, calving_interval, loss_per_conception)} for the active model."""
    states = build_cow_states()
    U = P.UNIT_DAYS
    probs = compute_probabilities(states)
    r, c, v = compute_transitions(states, probs)
    M = mc.assemble_matrix(states, r, c, v).tocsr()
    rid = get_replacement_state_id(states)
    Mlil = M.tolil()
    Mlil[:, rid] = 0                      # replacement is absorbing: a cow replaced in lactation 1 must not re-enter it
    M = Mlil.tocsr()
    lac, dim, dp = (states[k].values for k in ("lactation", "dim", "dp"))
    survive = (1 - probs["p_dead"]) * (1 - probs["p_cull"])
    checkpoints = list(checkpoints)
    first, result = {}, {}
    for L in lactations:
        idx_open = {k: i for i, k in zip(np.where((lac == L) & (dp == 0))[0], dim[(lac == L) & (dp == 0)])}
        open_mass = 1.0
        cum, ever = {}, 0.0
        for k in range(0, P.MAX_DIM + 1):
            if k not in idx_open:
                break
            i = idx_open[k]
            conc = open_mass * survive[i] * probs["p_preg"][i]
            ever += conc
            open_mass = open_mass * survive[i] * (1 - probs["p_preg"][i])
            cum[(k + 1) * U] = ever
        cum_at = {d: max([val for day, val in cum.items() if day <= d] or [0.0]) for d in checkpoints}
        # occupancy by DIM inside the lactation
        sel = np.where(lac == L)[0]
        ML = M[sel][:, sel].tocsr()
        pos = {s: j for j, s in enumerate(sel)}
        dist = np.zeros(len(sel))
        dist[pos[np.where((lac == L) & (dim == 0) & (dp == 0))[0][0]]] = 1.0
        occ = {}
        for step in range(0, P.MAX_DIM + 1):
            alive = dist.sum()
            if alive < 1e-9:
                break
            occ[(step + 1) * U] = (dist[dp[sel] > 0].sum() / alive) if alive > 0 else 0.0
            dist = ML.T @ dist
        occ_at = {d: min(occ.items(), key=lambda kv: abs(kv[0] - d))[1] if occ else 0.0 for d in checkpoints}
        # days open and calving interval for cows that calve: chain restricted to the lactation, absorbing 'calve'
        nxt = np.where(lac == L + 1)[0]
        calv_rate = np.asarray(M[sel][:, nxt].sum(axis=1)).ravel() if len(nxt) else np.zeros(len(sel))
        I = sp.eye(len(sel), format="csc")
        from scipy.sparse.linalg import spsolve
        e0 = np.zeros(len(sel)); e0[pos[np.where((lac == L) & (dim == 0) & (dp == 0))[0][0]]] = 1.0
        visits = spsolve((I - ML.tocsc()).T.tocsc(), e0)
        calves = visits @ calv_rate
        interval = U * ((visits * 1.0) @ (spsolve((I - ML.tocsc()), calv_rate))) / calves if calves > 1e-9 else float('nan')
        # days open: conception step of the pregnancy that reaches calving = dim - dp + 1 at calving states
        conc_step = (dim[sel] - dp[sel] + 1) * U
        days_open = float((visits * calv_rate * np.where(dp[sel] > 0, conc_step, 0)).sum() / max((visits * calv_rate).sum(), 1e-12))
        # losses per conception: expected abortions / expected conceptions in this lactation
        is_open = dp[sel] == 0
        conceptions = (visits * survive[sel] * probs["p_preg"][sel] * is_open).sum() + (visits * survive[sel] * probs["p_abort"][sel] * probs["p_resume"][sel] * (dp[sel] > 0)).sum()
        losses = (visits * survive[sel] * probs["p_abort"][sel] * (dp[sel] > 0)).sum()
        first[L] = cum_at
        result[L] = {"cum_conc": cum_at, "preg": occ_at, "days_open": days_open, "calving_interval": interval, "loss_per_conception": losses / conceptions}
    return result


def main():
    res = compute()
    checkpoints = [100, 150, 200, 300]
    print(f"unit={P.TIME_UNIT}  VWP={P.VOLUNTARY_WAITING_PERIOD_DAYS} d  last breeding DIM={P.LAST_BREEDING_DIM_DAYS} d")
    print(f"{'lac':>3} | " + " ".join(f"{'conc<=' + str(d):>9}" for d in checkpoints) + " | " +
          " ".join(f"{'P(preg)@' + str(d):>10}" for d in checkpoints) + " | days-open  calv-int  loss/concep")
    for L, r in res.items():
        print(f"{L:>3} | " + " ".join(f"{r['cum_conc'][d]*100:8.1f}%" for d in checkpoints) + " | " +
              " ".join(f"{r['preg'][d]*100:9.1f}%" for d in checkpoints) + f" | {r['days_open']:8.0f}  {r['calving_interval']:8.0f}  {r['loss_per_conception']*100:9.1f}%")
    first = {L: r["cum_conc"] for L, r in res.items()}
    o1, o2, o3 = (lambda p: p / (1 - p))(first[1][150]), (lambda p: p / (1 - p))(first[2][150]), (lambda p: p / (1 - p))(first[3][150])
    print(f"\nodds of conceiving by 150 DIM relative to lactation 1: lac2 {o2/o1:.2f}, lac3 {o3/o1:.2f}   (Cook & Green: 0.68, 0.47; survivors-only differences aside)")


if __name__ == "__main__":
    main()

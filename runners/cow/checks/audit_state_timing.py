"""
audit_state_timing.py
=====================
Read-only audit of what `dim` and `dp` mean under the model's own timing convention, and of the boundary cases
(waiting period, first pregnant state, pregnancy loss, dry-off, calving, next lactation). Prints the numbers quoted in
the maintainers' model audit. Nothing is written. Resolution: COW_TIME_UNIT=month|week|day.

Convention (cow/lookup_tables.py, cow/tables.py `_entry_ages`): the age at the instant of step k is k*U + u and the
gestation age at the instant of dp = j is (j-1)*U + v, u and v ~ Uniform(0, U) a priori; a cow in (k, j) conceived at
a = (e+1)*U + u - v, e = k - j. The model never tracks u or v.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import numpy as np


def main():
    import pandas as pd
    import cowmrp.domains.cow.parameters as P
    from cowmrp.domains.cow.parameters import UNIT_DAYS as U, VOLUNTARY_WAITING_PERIOD_DAYS as VWP, LAST_BREEDING_DIM_DAYS as LAST
    from cowmrp.domains.cow.states import build_cow_states
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.probabilities.calving import calving_step_mass
    from cowmrp.domains.cow.tables import get_tables
    from cowmrp.domains.cow.lookup_tables import N_DIM_MONTHS, get_full_table, table_to_array

    t = get_tables()
    S = build_cow_states()
    pr = compute_probabilities(S)
    lac, dim, dp = S['lactation'].values, S['dim'].values, S['dp'].values
    print(f"=== resolution {P.TIME_UNIT}: U = {U} d, VWP = {VWP} d (first conception day {VWP + 1}), last breeding day {LAST}")

    print("\n[1] meaning of the variables")
    print(f"  dim = k : nominal step index. Age at the instant = k*U + u, u ~ U(0,{U}) -> bin {U}k+1 .. {U}(k+1) d (label); mean age during the step = (k+1)*U.")
    print(f"  dp  = j : nominal gestation step. Gestation age at the instant = (j-1)*U + v, v ~ U(0,{U}); mean gestation age during the step = j*U.")
    print(f"  conception: during step e = dim - dp, at time s = U - v into it; recorded at the NEXT instant (dp = 1).")

    print("\n[2] earliest conception and first pregnant state")
    Pm = table_to_array(get_full_table('conception_full'), N_DIM_MONTHS)[0]
    for L in (1, 2, 3):
        ix = [S[(S.lactation == L) & (S.dim == k) & (S.dp == 0)].index[0] for k in range(int(t['entry_steps'].min()) + 3)]
        first = next(k for k, i in enumerate(ix) if pr['p_preg'][i] > 1e-12)
        print(f"  L={L}: first step with p_preg > 0: dim {first}; p_preg of steps {first}..{first + 2}: "
              f"{[round(float(pr['p_preg'][ix[k]]), 4) for k in range(first, first + 3)]}")
    e0 = int(t['entry_steps'].min())
    k0 = e0 + 1
    print(f"  first conception day = {VWP + 1}; first conception step e0 = {e0}; first pregnant state = (dim {k0}, dp 1)")
    # joint (u, s) density of conceptions in step e0, lactation 1 (competing risks ignored): weight = S(a)*lambda(a), a = e0*U + u + s
    step = 0.05 if U > 1 else 0.01
    g = np.arange(step / 2, U, step)
    uu, ss = np.meshgrid(g, g, indexing='ij')
    a = e0 * U + uu + ss
    days = np.arange(0, (t['dim_max'] + 3) * U + 2)
    p_m = Pm[0][np.minimum(days // 30, Pm.shape[1] - 1)]
    lam = np.where((days > VWP) & (days < LAST), -np.log1p(-p_m) / 30.0, 0.0)           # per-day continuous hazard
    cum = np.concatenate(([0.0], np.cumsum(lam)))
    H = np.interp(a, np.arange(len(cum)), cum)
    lam_a = lam[np.minimum(a.astype(int), len(lam) - 1)]
    w = np.exp(-H) * lam_a
    A_inst = (e0 + 1) * U + uu          # age at the instant
    v = U - ss                          # gestation age at the instant
    W = w / w.sum()
    pos = w > 0
    for name, arr, a_uniform in (("age at the instant", A_inst, k0 * U + U / 2), ("gestation at the instant", v, U / 2)):
        lo, hi, mean = arr[pos].min(), arr[pos].max(), float((arr * W).sum())
        print(f"  (dim {k0}, dp 1) {name}: between {lo:.2f} and {hi:.2f} d, conditional mean {mean:.2f} d "
              f"(the generic convention assumes uniform: mean {a_uniform:.1f} d)")
    print(f"  conception age: > {VWP + 1} d (hazard 0 through day {VWP}); at the instant she is {VWP + 1}..{(e0 + 2) * U} d old with "
          f"0..{(e0 + 2) * U - (VWP + 1)} d of gestation")
    print(f"  younger than {VWP + 1} d at the instant? No: the instant follows the conception, which is after day {VWP}.")
    i_open = S[(S.lactation == 1) & (S.dim == e0) & (S.dp == 0)].index[0]
    print(f"  probability of entering it from the open state (dim {e0}), lactation 1: {float((1 - pr['p_dead'][i_open]) * (1 - pr['p_cull'][i_open]) * pr['p_preg'][i_open]):.5f}")

    print("\n[3] every lactation: the waiting period restarts at calving")
    for L in (1, 2, 3, 10):
        row = [float(pr['p_preg'][S[(S.lactation == L) & (S.dim == k) & (S.dp == 0)].index[0]]) for k in range(e0, e0 + 3)]
        n_diag = int(((lac == L) & (dp > 0) & (dim - dp == e0)).sum())
        print(f"  L={L:2d}: p_preg at dim {e0}..{e0 + 2} = {np.round(row, 4)}; pregnant states with e = {e0}: {n_diag}")
    from cowmrp.domains.cow.transitions import compute_transitions
    r, c, vv = compute_transitions(S, pr)
    calv = (dp[r] > 0) & (lac[c] == lac[r] + 1)
    print(f"  calving transitions: {int(calv.sum())}; all land on (L+1, dim 0, dp 0): "
          f"{bool(((dim[c][calv] == 0) & (dp[c][calv] == 0)).all())}")

    print("\n[4] calving and dry-off")
    mass = calving_step_mass(0.0)
    print(f"  calving step mass (gestation {P.GESTATION_LENGTH_DAYS} d): { {k: round(v_, 4) for k, v_ in sorted(mass.items())} }")
    print(f"  dry-off: DRY_OFF_DP_DAYS = {P.DRY_OFF_DP_DAYS} d -> DRY_OFF_DP = {P.DRY_OFF_DP} (step with mean gestation age j*U = {P.DRY_OFF_DP * U} d)")

    print("\n[5] pregnancy loss")
    sub = S[(S.lactation == 1) & (S.dim - S.dp == e0) & (S.dp > 0)].sort_values('dp')
    pa = [(int(d), round(float(pr['p_abort'][i]), 4)) for i, d in zip(sub.index[:5], sub['dp'].values[:5])]
    surv = np.cumprod([1 - pr['p_abort'][i] for i in sub.index])
    j90 = max(1, int(round(90 / U)))
    print(f"  p_abort by dp (first five): {pa}")
    print(f"  cumulative loss through dp {j90} (about 90 d of gestation): {1 - float(surv[j90 - 1]):.3f}; "
          f"published windows (Albaaj 2022: 27/13/7/2 %) give {1 - 0.73 * 0.87 * 0.93 * 0.98:.3f}")
    print("  gestation day 0 is the conception instant of this model; the loss windows start at day 19. Whether the conception table "
          "is a service-level or a confirmed-pregnancy rate is not decided.")

    print("\n[6] independent re-derivation of p_preg from the hazard (exponential in each month; ratio of u-averaged survivals, because the open cows at an instant are weighted by survival to it)")
    worst = 0.0
    uu1 = (np.arange(400) + 0.5) / 400 * U
    for li in range(P.MAX_LACTATION):
        pm = Pm[li][np.minimum(days // 30, Pm.shape[1] - 1)]
        lam_l = np.where((days > VWP) & (days < LAST), -np.log1p(-pm) / 30.0, 0.0)
        cum_l = np.concatenate(([0.0], np.cumsum(lam_l)))
        Hf = lambda x: np.interp(x, np.arange(len(cum_l)), cum_l)
        for k in range(t['dim_max'] + 1):
            mine = 1 - float(np.mean(np.exp(-Hf(k * U + uu1 + U))) / np.mean(np.exp(-Hf(k * U + uu1))))   # open at the instant: u weighted by S(start)
            i = S[(S.lactation == li + 1) & (S.dim == k) & (S.dp == 0)].index
            if len(i):
                worst = max(worst, abs(mine - float(pr['p_preg'][i[0]])))
    print(f"  max |p_preg(model) - independent| over all lactations and steps: {worst:.2e}")


if __name__ == "__main__":
    main()

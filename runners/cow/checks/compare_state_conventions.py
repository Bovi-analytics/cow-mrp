"""
compare_state_conventions.py
============================
Read-only experiment: how much does the timing of conception change if eligibility and exposure are taken from the
NOMINAL bin [k*U, (k+1)*U) of each step instead of the offset-averaged window of the current convention
(cow/lookup_tables.py)? The nominal variant is what an "exact start-of-bin" design would do for conception: the
waiting period is applied to the bin itself, so a cow that conceives on days 51-60 is first recorded as pregnant at
the boundary that follows MIM 2 (and the diagonal state (dim=1, dp=1) disappears). Everything else (calving,
mortality, culling, milk curve, rewards) is left on the current convention, so the difference isolates the
conception-timing rule. Nothing is written; the variant exists only inside this process.

  python runners/cow/checks/compare_state_conventions.py            # runs the whole grid in subprocesses and prints the table
  python runners/cow/checks/compare_state_conventions.py UNIT VARIANT SCENARIO     # one run, prints JSON
      UNIT     month | week | day
      VARIANT  current | nominal
      SCENARIO base | vwp70 | conc80      (management / parameter scenarios used to compare rankings)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import json
import subprocess
import sys


def run_one(unit, variant, scen):
    _os.environ['COW_TIME_UNIT'] = unit
    import cowmrp.domains.cow.parameters as P
    if scen == 'vwp70':
        P.VOLUNTARY_WAITING_PERIOD_DAYS = 70
    import numpy as np
    import scipy.sparse as sp
    import cowmrp.domains.cow.probabilities.conception as C
    U, VWP, LAST = P.UNIT_DAYS, P.VOLUNTARY_WAITING_PERIOD_DAYS, P.LAST_BREEDING_DIM_DAYS
    orig_arrays = C._conception_step_arrays

    if variant == 'nominal':
        from cowmrp.domains.cow.lookup_tables import N_DIM_MONTHS, get_full_table, table_to_array, survival_from_monthly

        def nominal_arrays():
            Pm = table_to_array(get_full_table('conception_full'), N_DIM_MONTHS)[0]
            k = np.arange(P.MAX_DIM + 1)
            rows = []
            for i in range(Pm.shape[0]):
                S = survival_from_monthly(Pm[i], zero_until_day=VWP, zero_after_day=LAST)
                rows.append(1 - S((k + 1) * U) / S(k * U))
            return np.stack(rows)
        C._conception_step_arrays = nominal_arrays
    if scen == 'conc80':
        f = C._conception_step_arrays
        C._conception_step_arrays = lambda: f() * 0.8

    import cowmrp.domains.cow.state_constraints as sc
    if variant == 'nominal':    # the rule that matches nominal exposure: latest age in the bin is (e+1)*U
        keep = [c for c in sc.CONSTRAINTS if c.name != 'conception_after_vwp']
        keep.append(sc.StateConstraint('conception_after_vwp_nominal', 'imposed', 'nominal bin',
                                       lambda dim, dp, p: ~(dp > 0) | ((dim - dp + 1) * p.unit_days > p.vwp_days + 1)))
        sc.CONSTRAINTS = tuple(keep)

    from cowmrp.domains.cow.tables import get_tables, state_milk_kg
    t = get_tables()
    if variant == 'nominal':    # breeding-window share of each step from the nominal bin
        k = np.arange(P.MAX_DIM + 1)[:, None]
        days = k * U + np.arange(U)[None, :] + 0.5
        t['breed_frac'][:] = ((days > VWP) & (days <= LAST)).mean(axis=1)
        t['after_vwp_frac'][:] = (days > VWP).mean(axis=1)
    from cowmrp.domains.cow.states import build_cow_states
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.rewards import compute_rewards
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id, find_state_ids
    from cowmrp.mrp.core import evaluate_infinite_horizon, annual_to_per_step_gamma

    S = build_cow_states()
    pr = compute_probabilities(S)
    n = len(S)
    r, c, v = compute_transitions(S, pr)
    T = sp.csr_matrix((v, (r, c)), shape=(n, n))
    R = compute_rewards(S, pr)
    tot = np.asarray(R['total'])
    V = evaluate_infinite_horizon(T, tot, annual_to_per_step_gamma(0.95, U))
    out = dict(unit=unit, variant=variant, scen=scen, states=n, V_mean=float(V.mean()),
               V_repl=float(V[get_replacement_state_id(S)]))

    def vs(L, d, p):
        return float(V[int(find_state_ids(S, np.array([L]), np.array([d]), np.array([p]))[0])])
    out['PV30'] = {}
    for dd in (100, 150, 200, 250):
        k, j = P.dim_step_of_days(dd), P.dp_step_of_days(30)
        try:
            out['PV30'][dd] = vs(1, k, j) - vs(1, k, 0)
        except Exception:
            out['PV30'][dd] = None
    if scen == 'base':
        dim, dp, lac = S['dim'].values, S['dp'].values, S['lactation'].values
        pi = np.zeros(n)
        pi[get_replacement_state_id(S)] = 1
        for _ in range(80000):
            nx = T.T @ pi
            if abs(nx - pi).sum() < 1e-12:
                break
            pi = nx
        yr = 365 / U
        out['comp'] = {k_: float((pi * np.asarray(R[k_])).sum() * yr) for k_ in
                       ('milk_income', 'feed_cost', 'reproduction_cost', 'pregnancy_reward', 'cull_reward', 'total')}
        sv = (1 - pr['p_dead']) * (1 - pr['p_cull'])
        calv = float((pi * sv * (1 - pr['p_abort']) * pr['p_calve']).sum() * yr)
        st = np.where(dp == 0, 'open', np.where(dp < P.DRY_OFF_DP, 'preg', 'dry'))
        out['calv_yr'] = calv
        out['days_per_calving'] = {k_: float(pi[st == k_].sum() * 365 / calv) for k_ in ('open', 'preg', 'dry')}
        out['milk_kg_yr'] = float((pi * state_milk_kg(dim, lac, dp) * U).sum() * yr)
    return out


def run_grid(units=('month', 'week')):
    res = {}
    for unit in units:
        for variant in ('current', 'nominal'):
            for scen in ('base', 'vwp70', 'conc80'):
                p = subprocess.run([sys.executable, __file__, unit, variant, scen], capture_output=True, text=True)
                res[(unit, variant, scen)] = json.loads(p.stdout.strip().splitlines()[-1])
    for unit in units:
        print(f"\n== {unit}")
        for variant in ('current', 'nominal'):
            b = res[(unit, variant, 'base')]
            print(f" {variant:8s} states {b['states']:>6}  V_mean {b['V_mean']:9.1f}  calvings/yr {b['calv_yr']:.4f}  "
                  f"open/preg/dry days per calving {b['days_per_calving']['open']:.1f}/{b['days_per_calving']['preg']:.1f}/{b['days_per_calving']['dry']:.1f}  "
                  f"net reward/yr {b['comp']['total']:.1f}  milk kg/yr {b['milk_kg_yr']:.0f}  PV30@150 {b['PV30']['150']:.1f}")
            for scen in ('vwp70', 'conc80'):
                x = res[(unit, variant, scen)]
                print(f"          {scen}: dV_mean {x['V_mean'] - b['V_mean']:8.1f} ({100 * (x['V_mean'] / b['V_mean'] - 1):6.3f}%)")
    return res


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(run_one(*sys.argv[1:4])))
    else:
        run_grid()

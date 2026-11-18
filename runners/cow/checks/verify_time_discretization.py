"""
verify_time_discretization.py
=============================
Time-discretization consistency of the lookup-table layer. Runs the same biology at
MONTH, WEEK and DAY resolution (one subprocess each, COW_TIME_UNIT set) and checks that
cumulative event probabilities over the SAME real period agree across resolutions:

  - mortality, age/stage culling: the product of per-step probabilities over 210 real days from
    DIM ~90 must equal the CONTINUOUS-TIME cumulative probability of the same monthly table over the
    window [k0*U + U/2, k0*U + 210 + U/2] (relative error per resolution <= TOL). The half-step shift
    is the shared entry convention: a cow in state dim = k has true age k*U + u, u ~ Uniform(0, U),
    so on average U/2 older than k*U -- at MONTH a cow in state dim 3 is ~15 days older than a DAY
    cow in state dim 90. Comparing raw step products across resolutions would compare different age
    windows, so each resolution is checked against the continuous reference for ITS window.
  - conception: cumulative probability by DIM 210 (the waiting period is applied by day)
  - abortion: cumulative pregnancy loss over the first 210 days of gestation
  - calving: probability mass is complete (sums to 1) and E[k*U] equals GESTATION_LENGTH_DAYS
  - milk: step-mean x steps over 210 days (table consistency of the Wood curve)

Tolerance TOL is relative; the entry-averaging rule is designed to make these agree to well
under 1%, the check allows 2%. Exits 1 on any failure.   Run: python verify_time_discretization.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import json
import os
import subprocess
import sys

TOL = 0.02
PERIOD_DAYS = 210


def worker():
    import numpy as np
    import pandas as pd
    from cowmrp.domains.cow.parameters import UNIT_DAYS, GESTATION_LENGTH_DAYS
    import cowmrp.domains.cow.probabilities.mortality as M
    import cowmrp.domains.cow.probabilities.culling as C
    import cowmrp.domains.cow.probabilities.conception as K
    import cowmrp.domains.cow.probabilities.abortion as A
    import cowmrp.domains.cow.probabilities.calving as V
    from cowmrp.domains.cow.tables import state_milk_kg

    n = PERIOD_DAYS // UNIT_DAYS
    k = np.arange(n)
    k0 = int(round(90 / UNIT_DAYS))                      # window start: DIM ~90 days at every resolution
    frame = lambda dp: pd.DataFrame({'lactation': 2, 'dim': k0 + k, 'dp': dp})
    cum = lambda p: float(1 - np.prod(1 - np.asarray(p)))
    out = {}
    from cowmrp.domains.cow.lookup_tables import get_full_table, table_to_array, survival_from_monthly, N_DIM_MONTHS, STATUSES
    a_day = k0 * UNIT_DAYS + UNIT_DAYS / 2.0

    def continuous_cum(table_name):
        P = table_to_array(get_full_table(table_name), N_DIM_MONTHS, STATUSES)[0, 1]   # Open, lactation 2
        S = survival_from_monthly(P)
        return float(1 - S(a_day + PERIOD_DAYS) / S(a_day))

    for key, fn, table in (('mortality', M.p_mortality_full_table, 'mortality_full'),
                           ('culling_age_stage', C.p_culling_age_stage_full_table, 'culling_age_stage_full')):
        steps = cum(fn(frame(0), {}))
        ref = continuous_cum(table)
        out[f'{key}_step_vs_continuous_relerr'] = abs(steps - ref) / ref
        out[f'{key}_210d_from_d90'] = steps
    f1 = pd.DataFrame({'lactation': 1, 'dim': k, 'dp': 0})
    out['conception_210d_lac1'] = cum(K.p_conception_full_table(f1, {}))
    g = pd.DataFrame({'lactation': 2, 'dim': 10, 'dp': np.arange(1, n + 1)})
    out['abortion_210d_gestation'] = cum(A.p_abortion_full_table(g, {}))
    mass = V.calving_step_mass()
    out['calving_mass_sum'] = float(sum(mass.values()))
    # E[k*U] over the calving-step distribution: the model leaves the pregnant states at the END of the
    # calving step, whose offset from the true calving moment is uniform(0, U) -- so E[k*U] must equal the
    # gestation length at every resolution.
    out['mean_calving_step_days_minus_G'] = float(sum(m * kk * UNIT_DAYS for kk, m in mass.items()) - GESTATION_LENGTH_DAYS)
    from cowmrp.domains.cow.curves import curve_kg
    milk = state_milk_kg(k, np.full(n, 2), np.zeros(n, dtype=int)) * UNIT_DAYS
    ref = curve_kg(np.arange(PERIOD_DAYS) + 0.5 + UNIT_DAYS / 2.0, 2).sum()   # continuous curve, window shifted U/2
    out['milk_step_vs_continuous_relerr'] = float(abs(milk.sum() - ref) / ref)
    out['milk_kg_210d_lac2'] = float(milk.sum())
    # lifetime / event-level consistency of the WHOLE chain (the quantities a herd manager reads)
    from runners.cow.checks.audit_metrics import collect
    m = collect(print_result=False)
    for k in ('cycle_days', 'calvings_per_cycle', 'lac2_P_calve_again', 'lac2_calving_interval_days_if_calved'):
        out['lifetime_' + k] = float(m[k])
    print("RESULT" + json.dumps(out))


def main():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results = {}
    for unit in ("month", "week", "day"):
        env = dict(os.environ, COW_TIME_UNIT=unit)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, cwd=here,
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode != 0 or not line:
            print(f"[FAIL] worker at {unit} failed:\n{p.stderr[-1500:]}")
            sys.exit(1)
        results[unit] = json.loads(line[0][6:])
    keys = list(results["month"])
    failed = []
    print(f"{'metric':32s} {'month':>11s} {'week':>11s} {'day':>11s}   max rel spread")
    for key in keys:
        vals = [results[u][key] for u in ("month", "week", "day")]
        spread = (max(vals) - min(vals)) / max(abs(max(vals)), 1e-12)
        flag = "OK" if spread <= TOL else "FAIL"
        if key.startswith("lifetime_"):
            tol = {'lifetime_calvings_per_cycle': 0.03, 'lifetime_cycle_days': 0.03}.get(key, 0.02 if 'P_calve' in key else 0.01)
            flag = "OK" if spread <= tol else "FAIL"
            print(f"[{flag}] {key:28s} {vals[0]:11.3f} {vals[1]:11.3f} {vals[2]:11.3f}   {spread:.2%} (limit {tol:.0%})")
            if flag == "FAIL":
                failed.append(key)
            continue
        if key.endswith("_vs_continuous_relerr"):
            flag = "OK" if all(v <= TOL for v in vals) else "FAIL"
            print(f"[{flag}] {key:28s} {vals[0]:11.4%} {vals[1]:11.4%} {vals[2]:11.4%}   (limit {TOL:.0%})")
            if flag == "FAIL":
                failed.append(key)
            continue
        if key == "milk_kg_210d_lac2":
            print(f"[info] {key:28s} {vals[0]:11.4f} {vals[1]:11.4f} {vals[2]:11.4f}   (different age windows)")
            continue
        if key.endswith("_210d_from_d90"):
            print(f"[info] {key:28s} {vals[0]:11.4f} {vals[1]:11.4f} {vals[2]:11.4f}   "
                  f"(differs by design: different age windows)")
            continue
        if key == "calving_mass_sum":
            flag = "OK" if all(abs(v - 1.0) < 1e-9 for v in vals) else "FAIL"
        if key == "mean_calving_step_days_minus_G":
            flag = "OK" if all(abs(v) <= 1.0 for v in vals) else "FAIL"   # within one day of GESTATION_LENGTH_DAYS
        print(f"[{flag}] {key:28s} {vals[0]:11.4f} {vals[1]:11.4f} {vals[2]:11.4f}   {spread:.4%}")
        if flag == "FAIL":
            failed.append(key)
    if failed:
        print(f"\nTIME_DISCRETIZATION_FAILED: {failed}")
        sys.exit(1)
    print("\nTIME_DISCRETIZATION_OK")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()

"""
audit_methods.py
================
Event-level comparison of the alternative implementations of each probability (table vs formula), as cumulative
event probabilities over fixed real-age windows -- NOT through V. When two methods disagree this shows WHERE
(which ages) and by HOW MUCH, so the choice between them is made on biology, not on which gives the preferred V.

    python verify/audit_methods.py            # runs at DAY resolution (steps of 1 day: negligible discretization)

Writes outputs/audit/method_comparison.csv and prints the table.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # project root (mrp/, cow/)
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker():
    import numpy as np
    import pandas as pd
    import cow.probabilities.mortality as M
    import cow.probabilities.culling as C
    import cow.probabilities.conception as K
    import cow.probabilities.abortion as A
    from cow.parameters import UNIT_DAYS, DRY_OFF_DP
    assert UNIT_DAYS == 1
    rows = []

    def cum(p, a, b):
        return float(1 - np.prod(1 - np.asarray(p)[a:b]))

    ks = np.arange(0, 700)
    for lac in (1, 2, 3, 6):
        f = pd.DataFrame({'lactation': lac, 'dim': ks, 'dp': 0})
        for name, tab, form in (('mortality (open)', M.p_mortality_full_table, M.p_mortality_anchor_calibrated),
                                ('culling age/stage (open)', C.p_culling_age_stage_full_table, C.p_cull_age_stage_anchor_calibrated)):
            pt, pf = tab(f, {}), form(f, {})
            for a, b in ((0, 60), (60, 150), (150, 300), (300, 600)):
                rows.append({'quantity': name, 'lactation': lac, 'window': f'DIM {a}-{b} d',
                             'table': cum(pt, a, b), 'formula(bucket)': cum(pf, a, b)})
        pt = K.p_conception_full_table(f, {})
        pf = K._conception_survival_function(lac == 1)      # legacy bucket survival (reference)
        S = pf(np.arange(0, 701)); legacy = 1 - S[1:] / S[:-1]
        for a, b in ((0, 100), (0, 200), (0, 300)):
            rows.append({'quantity': 'conception (cumulative if never lost)', 'lactation': lac, 'window': f'DIM {a}-{b} d',
                         'table': cum(pt, a, b), 'formula(bucket)': cum(legacy, a, b)})
    g = pd.DataFrame({'lactation': 2, 'dim': 0, 'dp': np.arange(0, 290)})
    pt, pf = A.p_abortion_full_table(g, {}), A.p_abortion_albaaj2022(g, {})
    for a, b in ((1, 30), (1, 60), (1, 90), (1, 150), (1, 277)):
        rows.append({'quantity': 'pregnancy loss (cumulative from conception)', 'lactation': 'all (pooled)',
                     'window': f'gestation {a}-{b} d', 'table': cum(pt, a, b), 'formula(bucket)': cum(pf, a, b)})
    df = pd.DataFrame(rows)
    df['abs_diff'] = df['table'] - df['formula(bucket)']
    print("RESULT" + df.to_json(orient='records'))


def main():
    env = dict(os.environ, COW_TIME_UNIT='day')
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker'], env=env, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith('RESULT')]
    if p.returncode or not line:
        print(p.stderr[-2000:]); sys.exit(1)
    import pandas as pd
    df = pd.read_json(line[0][6:])
    out = os.path.join(ROOT, 'outputs', 'audit')
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'method_comparison.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
    print("\nColumns: 'table' = full lactation x month table (active for mortality/culling/conception; abortion default is the"
          "\nAlbaaj formula); 'formula(bucket)' = the previous implementation. Differences come from month alignment of the"
          "\nDIM buckets (60/150/300 d), the half-step entry offset, and the 350-d conception edge -- see MODEL_CHANGELOG.")


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()

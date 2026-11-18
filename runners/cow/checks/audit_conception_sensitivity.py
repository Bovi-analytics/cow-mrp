"""
audit_conception_sensitivity.py
===============================
How much does the one remaining conception PLACEHOLDER (the lactation-1 monthly level and DIM shape) matter?
Scenarios perturb ONLY the lactation-1 reference values of `cow.probabilities.conception.conception_table` (monthly probability per
eligible open cow by DIM bucket); lactation >= 2 follow through the unchanged parity odds ratios. Nothing is tuned and nothing is
written to disk in the model: each scenario runs in its own process.

Scenarios (lactation-1 monthly probability at DIM 51-100 / 101-200 / 201-350 / 351+; base 0.45 / 0.45 / 0.45 / 0.35):
  base             as is
  level_low        odds x 0.75 in every bucket   (about 0.38 / 0.38 / 0.38 / 0.28)
  level_high       odds x 1.33 in every bucket   (about 0.52 / 0.52 / 0.52 / 0.42)
  shape_declining  0.52 / 0.45 / 0.36 / 0.28     (fertility falling with DIM)
  shape_peak       0.38 / 0.50 / 0.42 / 0.30     (peak around DIM 100-200, as Remnant 2018 suggests)
  shape_flat       0.45 / 0.45 / 0.45 / 0.45     (no late drop)

    python runners/cow/checks/audit_conception_sensitivity.py [--unit week]
"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, HERE)
from runners._shared import common

SCEN = {
    "base": None,
    "level_low": ("odds", 0.75),
    "level_high": ("odds", 1 / 0.75),
    "shape_declining": ("set", [0.52, 0.45, 0.36, 0.28]),
    "shape_peak": ("set", [0.38, 0.50, 0.42, 0.30]),
    "shape_flat": ("set", [0.45, 0.45, 0.45, 0.45]),
    # pregnancy-loss check (not conception): scale every loss window AFTER day 30 (Albaaj 30-90 d, Lee & Kim 91-270 d)
    "loss_after30_x0.5": ("loss", 0.5),
    "loss_after30_x0.75": ("loss", 0.75),
}
BUCKETS = ["51-100", "101-200", "201-350", "351-449"]


def apply(name):
    s = SCEN[name]
    if s is None:
        return
    if s[0] == "loss":         # must run before the abortion module builds its survival tables at import
        import cowmrp.domains.cow.parameters as P
        a = P.ABORTION_ALBAAJ2022
        for w in a["published"]["windows"][1:]:
            w["loss_rate"] *= s[1]
        for w in a["outside_published_range"]["late_windows"]:
            w["loss_rate"] *= s[1]
        return
    import cowmrp.domains.cow.probabilities.conception as K
    df = K.conception_table
    m1 = df["lactation_bucket"] == "1"
    for b_i, b in enumerate(BUCKETS):
        sel = m1 & (df["dim_bucket"] == b)
        if s[0] == "odds":
            p = df.loc[sel, "p"].astype(float)
            o = p / (1 - p) * s[1]
            df.loc[sel, "p"] = o / (1 + o)
        else:
            df.loc[sel, "p"] = s[1][b_i]


def one(name):
    apply(name)
    import numpy as np
    import cowmrp.domains.cow.parameters as P
    from cowmrp.domains.cow.lookup_tables import get_full_table
    import runners.cow.checks.audit_metrics as AM
    import runners.cow.checks.audit_repro as AR
    d = AM.collect(print_result=False)
    r = AR.compute(lactations=(1, 2, 3), checkpoints=(100, 150, 200, 300))
    tab = get_full_table("conception_full")
    monthly = {L: [float(tab[(tab.lactation == L) & (tab.month == m)]["p"].iloc[0]) for m in (2, 5, 8)] for L in (1, 2, 3)}   # DIM 75 / 165 / 255
    out = {"V": d["V_replacement"], "preg_new": d["pregnancy_value_dp1_lac2_d150"], "preg_matched": d["pregnancy_value_matched_lac2_dim165_gest105"],
           "cycle_days": d["cycle_days"], "calvings": d["calvings_per_cycle"], "open_days_cycle": d["open_days"],
           "exit_cull_repro": d["exit_cull_repro"], "exit_death": d["exit_death"], "live_cull": d["live_cull_pct_per_year"], "death": d["death_pct_per_year"],
           "monthly_p": monthly, "days_open": {L: r[L]["days_open"] for L in r}, "preg150": {L: r[L]["preg"][150] for L in r},
           "preg300": {L: r[L]["preg"][300] for L in r}}
    print("RESULT" + json.dumps(out))


def main():
    unit = sys.argv[sys.argv.index("--unit") + 1] if "--unit" in sys.argv else "week"
    res = {}
    for name in SCEN:
        env = dict(os.environ, COW_TIME_UNIT=unit)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name], env=env, cwd=HERE, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            print(p.stderr[-1500:]); sys.exit(1)
        res[name] = json.loads(line[0][6:])
    b = res["base"]
    print(f"unit={unit}\n")
    print("conception probability by DIM and lactation (native monthly probability per eligible open cow; DIM ~75 / 165 / 255):")
    for name, r in res.items():
        print(f"  {name:16s} " + "   ".join(f"lac{L}: " + "/".join(f"{x:.2f}" for x in r["monthly_p"][str(L)]) for L in (1, 2, 3)))
    print(f"\n{'scenario':16s} {'days open L1/L2/L3':>20s} {'P(preg)@150 L1/L2/L3':>24s} {'P(preg)@300 L1':>15s} {'V(repl)':>9s} {'dV':>7s} {'new preg value':>15s} {'dPV':>7s} {'cycle d':>8s} {'milk-floor share':>17s}")
    for name, r in res.items():
        print(f"{name:16s} {'/'.join(f'{r['days_open'][str(L)]:.0f}' for L in (1, 2, 3)):>20s} {'/'.join(f'{r['preg150'][str(L)]*100:.0f}%' for L in (1, 2, 3)):>24s} "
              f"{r['preg300']['1']*100:14.0f}% {r['V']:9.0f} {(r['V']/b['V']-1)*100:+6.1f}% {r['preg_new']:15.1f} {(r['preg_new']/b['preg_new']-1)*100:+6.1f}% {r['cycle_days']:8.0f} {r['exit_cull_repro']*100:16.1f}%")
    json.dump(res, open(common.module_outputs("cow", "audit", "conception_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    if "--one" in sys.argv:
        one(sys.argv[sys.argv.index("--one") + 1])
    else:
        main()

"""
audit_later_lactation_sensitivity.py
====================================
PROPOSED corrections for lactations 4-10, evaluated as sensitivities. NOTHING here changes a model default or a file under cow/.

Two findings from re-reading the existing PDFs (docs/COW_MODEL_AUDIT.md section 29):

 M  Mortality, lactations 4-10. The model spreads Pinedo's single '>=4' hazard over lactations with Miller (2008) ratios, but normalises the
    ratios with an EQUAL-weight mean over lactations 4-10 (cow/evidence/miller2008.py). The '>=4' group is not equal-weighted: most of its
    cow-time is in lactations 4-5. Equal weights therefore push lactation 4 BELOW the '>=4' value and below lactation 3 (against both
    sources). Proposal: weight the normalisation by the share of cows entering each lactation (derived from Hadley 2006 Table 5, independent
    of this model).
 C  Other-reason culling, lactations 5-10. The model repeats Pinedo's '>=4' hazard flat. Hadley 2006 Table 5 (all culls per lactation
    42.9, 47.3, 51.1, 54.4, 57.2, 60.2, 61.8 % for lactations 4..10+) and Rilanto 2020 Table 4 (hazard ratios vs parity 2: 1.81, 2.22, 2.49,
    3.13 for parity 4, 5, 6, >=7) both show culling still rising. Proposal: multiply the lactation-L (5..10) hazard by the source's ratio
    to lactation 4 (hazard scale).

Scenarios (WEEK):  base | M (weighted Miller normalisation) | C_hadley | C_rilanto | M + C_hadley
    python verify/audit_later_lactation_sensitivity.py
"""
import os, sys, json, subprocess, math
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

HADLEY_ALL_CULL = [17.9, 31.3, 37.3, 42.9, 47.3, 51.1, 54.4, 57.2, 60.2, 61.8]       # % per lactation, lactations 1..10+ (Table 5)
RILANTO_HR = {2: 1.00, 3: 1.38, 4: 1.81, 5: 2.22, 6: 2.49, 7: 3.13, 8: 3.13, 9: 3.13, 10: 3.13}   # vs parity 2 (Table 4); '>=7' repeated


def hadley_hazard_ratio_to_L4():
    h = {L: -math.log(1 - HADLEY_ALL_CULL[L - 1] / 100.0) for L in range(1, 11)}
    return {L: h[L] / h[4] for L in range(5, 11)}


def rilanto_ratio_to_L4():
    return {L: RILANTO_HR[L] / RILANTO_HR[4] for L in range(5, 11)}


def hadley_entering_shares():
    """share of replacement cows entering lactation L (1..10), from Hadley's all-cull percentages (independent of the model)"""
    s, out = 1.0, {}
    for L in range(1, 11):
        out[L] = s
        s *= 1 - HADLEY_ALL_CULL[L - 1] / 100.0
    return out


def weighted_miller_ratio():
    from cow.evidence.miller2008 import DEATH_PER_LACTATION_PCT as D
    w = hadley_entering_shares()
    val = {L: D[min(L, 8)] for L in range(4, 11)}
    mean = sum(w[L] * val[L] for L in range(4, 11)) / sum(w[L] for L in range(4, 11))
    return {L: val[L] / mean for L in range(4, 11)}


def apply(name):
    import cow.probabilities, cow.tables                                        # noqa: F401
    from cow.lookup_tables import FULL_TABLE_REGISTRY
    from cow.probabilities.hazard_tables import pinedo_status_table
    from cow.evidence.pinedo2010 import OTHER_CULL_CODES
    from cow.evidence.miller2008 import death_relative_to_pinedo_parity4plus
    mort = weighted_miller_ratio() if name in ("M", "M_C_hadley") else death_relative_to_pinedo_parity4plus()
    cull = {"C_hadley": hadley_hazard_ratio_to_L4(), "M_C_hadley": hadley_hazard_ratio_to_L4(), "C_rilanto": rilanto_ratio_to_L4()}.get(name)
    FULL_TABLE_REGISTRY["mortality_full"]["build"] = lambda: pinedo_status_table(("died",), lactation_factor=mort)
    if cull:
        FULL_TABLE_REGISTRY["culling_age_stage_full"]["build"] = lambda: pinedo_status_table(OTHER_CULL_CODES, lactation_factor=cull)


def one(name):
    apply(name)
    import verify.audit_metrics as AM
    import verify.audit_exits as AE
    d = AM.collect(print_result=False)
    rows, x, states, flows, U = AE.exit_table()
    per = {r[0]: {"entry": r[1], "death": r[3], "other": r[4], "floor": r[5]} for r in rows}
    print("RESULT" + json.dumps({"V": d["V_replacement"], "cycle": d["cycle_days"], "calvings": d["calvings_per_cycle"], "preg_new": d["pregnancy_value_dp1_lac2_d150"],
                                 "death_yr": d["death_pct_per_year"], "cull_yr": d["live_cull_pct_per_year"], "per": per}))


def main():
    names = ["base", "M", "C_hadley", "C_rilanto", "M_C_hadley"]
    res = {}
    for n in names:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", n], env=dict(os.environ, COW_TIME_UNIT="week"), cwd=HERE, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            print(p.stderr[-2000:]); sys.exit(1)
        res[n] = json.loads(line[0][6:])
    mr, hr, rr = weighted_miller_ratio(), hadley_hazard_ratio_to_L4(), rilanto_ratio_to_L4()
    from cow.evidence.miller2008 import death_relative_to_pinedo_parity4plus as eq
    eqr = eq()
    print("Miller normalisation (multiplier on the '>=4' hazard)   lactation: " + " ".join(f"{L:>6d}" for L in range(4, 11)))
    print("  equal weights (current)               " + " ".join(f"{eqr[L]:6.2f}" for L in range(4, 11)))
    print("  weighted by share of cows entering    " + " ".join(f"{mr[L]:6.2f}" for L in range(4, 11)))
    print("Other-culling multiplier vs lactation 4 (hazard scale)  lactation: " + " ".join(f"{L:>6d}" for L in range(5, 11)))
    print("  Hadley 2006 Table 5 (all culls)       " + " ".join(f"{hr[L]:6.2f}" for L in range(5, 11)))
    print("  Rilanto 2020 Table 4 (HR vs parity 4) " + " ".join(f"{rr[L]:6.2f}" for L in range(5, 11)))
    b = res["base"]
    print(f"\n{'scenario':12s} {'V(repl)':>9s} {'dV':>7s} {'cycle d':>8s} {'calv':>6s} {'new preg value':>15s} {'death %/yr':>11s} {'live cull %/yr':>15s}")
    for n, r in res.items():
        print(f"{n:12s} {r['V']:9.0f} {(r['V']/b['V']-1)*100:+6.1f}% {r['cycle']:8.0f} {r['calvings']:6.2f} {r['preg_new']:15.1f} {r['death_yr']:11.2f} {r['cull_yr']:15.2f}")
    print("\ndeath per lactation, % of cows entering (Miller 2008: 2.05 2.66 3.72 4.38 4.83 5.78 5.92 6.40 | Hadley 2006: 1.9 3.4 4.5 5.4 6.0 6.3 6.6 6.6 6.9 7.5)")
    for n, r in res.items():
        print(f"  {n:11s} " + " ".join(f"{r['per'][str(L)]['death'] * 100:5.2f}" for L in range(1, 11)))
    print("non-death culling per lactation (other + milk floor), % of entering cows (Hadley 'all culls': " + " ".join(f"{v:.1f}" for v in HADLEY_ALL_CULL) + ")")
    for n, r in res.items():
        print(f"  {n:11s} " + " ".join(f"{(r['per'][str(L)]['other'] + r['per'][str(L)]['floor']) * 100:5.1f}" for L in range(1, 11)))
    json.dump(res, open(os.path.join(HERE, "outputs", "audit", "later_lactation_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    if "--one" in sys.argv:
        one(sys.argv[sys.argv.index("--one") + 1])
    else:
        main()

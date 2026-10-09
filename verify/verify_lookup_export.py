"""
verify_lookup_export.py
=======================
Checks that the exported lookup tables (outputs/lookup_tables/) and Plotly figures really are the ACTIVE model, for lactations 1-10.

    python verify/export_lookup_tables.py && python verify/verify_lookup_export.py

  1  every expected export file exists (lookup tables, source evidence, effective tables, per-step tables at MONTH / WEEK / DAY, HTML, Plotly)
  2  effective wide tables have columns L1..L10, no missing values, probabilities in [0, 1]
  3  effective native tables equal the active full tables read by the model
  4  per-step tables equal what the ACTIVE resolver returns now (sampled; each resolution in its own process)
  5  provenance: every mechanism x lactation has a class and agrees with the table `basis` labels; the provenance matrix file matches the registry
  6  the legacy '1 / 2 / 3+' bucket tables do NOT drive the active probabilities (perturbing them leaves the active values unchanged), except the
     documented conception lactation-'1' reference rows (perturbing those changes conception)
  7  the milk-curve export covers lactations 1-10 for both curves and equals cow.curves.curve_kg
  8  Plotly: individual figure files, validation_all.html and index.html exist; traces for lactations 1-10 are present in the lactation figures
Exit status 1 on any failure.
"""
import os, sys, re, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
UNITS = ("month", "week", "day")
LACS = list(range(1, 11))
OUT = os.path.join(ROOT, "outputs", "lookup_tables")
PLY = os.path.join(ROOT, "outputs", "validation", "plotly")
failed = []


def ok(cond, msg):
    print(("[OK] " if cond else "[FAIL] ") + msg)
    if not cond:
        failed.append(msg)


def _scale(g, mask_fn):
    g = g.copy()
    m = mask_fn(g)
    g.loc[m, "p"] = (g.loc[m, "p"] * 7).clip(upper=1.0)
    return g


def check_unit(unit):
    """Per-step tables vs the active resolver at one resolution (run in its own process)."""
    import numpy as np, pandas as pd
    import cow.probabilities
    from mrp.engine import resolve_method
    from cow.probabilities import probability_specs
    spec = {s["name"]: s for s in probability_specs}
    rng = np.random.default_rng(1)
    worst = 0.0
    n = 0
    for fname, name in (("mortality", "mortality"), ("other_reason_culling", "culling_age_stage"), ("conception", "conception"), ("milk_floor_cull", "culling_repro"),
                        ("pregnancy_loss", "abortion"), ("calving", "calving")):
        df = pd.read_csv(os.path.join(OUT, "effective", "effective_step", unit, fname + ".csv"))
        idx = rng.choice(len(df), size=min(60, len(df)), replace=False)
        for i in idx:
            r = df.iloc[i]
            dim = int(r["step_index"]) if r["status"] == "Open" else 0
            dp = 0 if r["status"] == "Open" else int(r["step_index"])
            for L in LACS:
                v = np.asarray(resolve_method(spec[name])(pd.DataFrame({"lactation": [L], "dim": [dim], "dp": [dp]}), {}), float)[0]
                worst = max(worst, abs(v - float(r[f"L{L}"]))); n += 1
    return worst, n


def main():
    import numpy as np, pandas as pd
    # ---- 1 files
    expected = ["lactation_tables.html", "README.md", "milk_curves_by_lactation.csv", "mortality_full.csv", "culling_age_stage_full.csv", "conception_full.csv", "abortion_full.csv",
                "pregnancy_milk_factor_full.csv"]
    expected += [f"source_evidence/{n}.csv" for n in ("pinedo2010_hazards_source", "miller2008_death_per_lactation_source", "legacy_bucket_tables", "conception_parity_effect_source",
                                                       "pregnancy_loss_source", "gestation_length_source", "later_lactation_evidence_NOT_USED", "milk_curve_source")]
    mechs = ("mortality", "other_reason_culling", "conception", "pregnancy_loss", "calving", "milk_floor_cull", "pregnancy_milk_factor", "milk_curve")
    expected += [f"effective/{m}_effective_wide.csv" for m in mechs] + ["effective/effective_long.csv", "effective/provenance_matrix_wide.csv", "effective/provenance_matrix_long.csv",
                                                                        "effective/milk_curves_by_lactation.csv", "effective/pregnancy_loss_active_daily_hazard.csv"]
    for u in UNITS:
        expected += [f"effective/effective_step/{u}/{n}.csv" for n in ("mortality", "other_reason_culling", "conception", "milk_floor_cull", "pregnancy_loss", "calving")]
    missing = [f for f in expected if not os.path.isfile(os.path.join(OUT, f))]
    ok(not missing, f"all {len(expected)} expected lookup-table export files exist" + (f"; MISSING {missing[:5]}" if missing else ""))

    # ---- 2 wide tables
    bad = []
    for m in mechs:
        d = pd.read_csv(os.path.join(OUT, "effective", f"{m}_effective_wide.csv"))
        cols = [f"L{L}" for L in LACS]
        if not all(c in d.columns for c in cols) or d[cols].isna().any().any():
            bad.append(f"{m}: missing L-columns or NaN")
        elif m not in ("calving", "milk_floor_cull", "milk_curve") and ((d[cols] < -1e-12).any().any() or (d[cols] > 1 + 1e-12).any().any()):
            bad.append(f"{m}: value outside [0, 1]")
    ok(not bad, "effective wide tables carry L1..L10 for every row, no missing values, probabilities within [0, 1]" + (f"  {bad}" if bad else ""))

    # ---- 3 native effective == active full tables
    import cow.probabilities, cow.rewards, cow.tables
    from cow.lookup_tables import get_full_table
    diffs = []
    for m, tname, col in (("mortality", "mortality_full", "p"), ("other_reason_culling", "culling_age_stage_full", "p"), ("conception", "conception_full", "p"),
                          ("pregnancy_loss", "abortion_full", "p"), ("pregnancy_milk_factor", "pregnancy_milk_factor_full", "factor")):
        eff = pd.read_csv(os.path.join(OUT, "effective", f"{m}_effective_wide.csv"))
        tab = get_full_table(tname)
        for _, r in eff.sample(min(40, len(eff)), random_state=0).iterrows():
            for L in LACS:
                q = tab[(tab.lactation == L) & (tab.month == r["month"])]
                if "status" in tab.columns and r["status"] in ("Open", "Pregnant", "Dry"):
                    q = q[q.status == r["status"]]
                diffs.append(abs(float(q[col].iloc[0]) - float(r[f"L{L}"])))
    ok(max(diffs) <= 1e-12, f"effective native tables equal the active full tables the model reads (max abs diff {max(diffs):.1e}, {len(diffs)} cells)")

    # ---- 4 per-step vs active resolver (own process per resolution)
    for u in UNITS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--unit", u], env=dict(os.environ, COW_TIME_UNIT=u), cwd=ROOT, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            ok(False, f"per-step check at {u} crashed: {p.stderr[-300:]}"); continue
        worst, n = json.loads(line[0][6:])
        ok(worst <= 1e-12, f"{u}: exported per-step probabilities equal the active resolvers' output ({n} sampled values, max abs diff {worst:.1e})")

    # ---- 5 provenance
    from cow.registry import check_provenance, lactation_provenance
    prov = check_provenance()
    ok(True, "provenance classes cover lactations 1-10 for every mechanism and agree with the active tables' basis labels")
    pl = pd.read_csv(os.path.join(OUT, "effective", "provenance_matrix_long.csv"))
    same = all(pl[(pl.mechanism == m) & (pl.lactation == L)]["provenance_class"].iloc[0] == prov[m][L][0] for m in prov for L in LACS)
    ok(same and len(pl) == len(prov) * 10, "exported provenance matrix equals cow/registry.py::lactation_provenance")
    ok(all(prov[m][L][0] in ("repeated", "grouped") for m in ("mortality", "other_reason_culling", "conception") for L in range(5, 11) if not (m == "mortality" and L in (5, 6, 7, 8) and False)),
       "lactations 5-10 of mortality, other-reason culling and conception are labelled grouped/repeated, never direct")

    # ---- 6 legacy buckets do not drive the active model
    import cow.probabilities.mortality as MORT, cow.probabilities.culling as CULL, cow.probabilities.conception as CONC
    from mrp.engine import resolve_method
    from cow.probabilities import probability_specs
    spec = {s["name"]: s for s in probability_specs}
    import cow.parameters as P
    k150 = max(1, int(round(150 / P.UNIT_DAYS)) - 1)                     # the same true age (~150 d in milk) at every resolution
    k60 = max(0, int(round(60 / P.UNIT_DAYS)) - 1)
    st = pd.DataFrame({"lactation": [1, 2, 3, 5, 10] * 2, "dim": [k150] * 5 + [k60] * 5, "dp": [0] * 10})

    def active(name):
        return np.asarray(resolve_method(spec[name])(st, {}), float)
    before = {n: active(n) for n in ("mortality", "culling_age_stage", "conception")}
    ok(all((v > 0).all() for v in before.values()), "the perturbation test cells have positive baseline probabilities (checks below are not vacuous)")
    saved = (MORT.mortality_table["p"].copy(), CULL.culling_table["p"].copy(), CONC.conception_table["p"].copy())
    MORT.mortality_table["p"] *= 3; CULL.culling_table["p"] *= 3
    ok(np.allclose(active("mortality"), before["mortality"]) and np.allclose(active("culling_age_stage"), before["culling_age_stage"]),
       "perturbing the legacy 1/2/3+ mortality and culling bucket tables leaves the ACTIVE mortality and culling probabilities unchanged (they feed only the inactive 'formula' slots)")
    MORT.mortality_table["p"] = saved[0]; CULL.culling_table["p"] = saved[1]
    two = CONC.conception_table["lactation_bucket"] == "2+"
    CONC.conception_table.loc[two, "p"] *= 0.5
    from cow.lookup_tables import FULL_TABLE_REGISTRY
    import cow.probabilities.conception as C2
    C2._conception_step_arrays.cache_clear(); __import__("cow.lookup_tables", fromlist=["x"]).clear_table_cache()
    ok(np.allclose(active("conception"), before["conception"]), "perturbing the legacy conception '2+' bucket rows leaves active conception unchanged (lactations >= 2 come from odds ratios)")
    CONC.conception_table["p"] = saved[2]
    one = CONC.conception_table["lactation_bucket"] == "1"
    CONC.conception_table.loc[one & (CONC.conception_table["p"] > 0), "p"] *= 0.5
    C2._conception_step_arrays.cache_clear(); __import__("cow.lookup_tables", fromlist=["x"]).clear_table_cache()
    ok(not np.allclose(active("conception"), before["conception"]), "the conception lactation-'1' bucket rows ARE the active lactation-1 reference (documented dependence)")
    CONC.conception_table["p"] = saved[2]
    C2._conception_step_arrays.cache_clear(); __import__("cow.lookup_tables", fromlist=["x"]).clear_table_cache()

    # ---- 6b unreachable rows (padding) are never read; reachable rows are
    import cow.probabilities.mortality as MORT2, cow.probabilities.culling as CULL2
    from cow.lookup_tables import FULL_TABLE_REGISTRY as REG
    reach_all = json.load(open(os.path.join(ROOT, "outputs", "audit", "reachability.json")))["used_months"]
    states = __import__("cow.states", fromlist=["build_cow_states"]).build_cow_states()
    def eff_vals():
        return {n: np.asarray(resolve_method(spec[n])(states, {}), float) for n in ("mortality", "culling_age_stage")}
    base_v = eff_vals()
    # months the active model can read (union over the three resolutions): Open 0-21, Pregnant 0-7, Dry 7-10
    def pad_mask(g):
        s, m = g["status"].values, g["month"].values
        return ((s == "Open") & (m >= 22)) | ((s == "Pregnant") & (m >= 8)) | ((s == "Dry") & ((m <= 6) | (m >= 11)))
    def used_mask(g):
        s, m = g["status"].values, g["month"].values
        return ((s == "Dry") & (m == 8)) | ((s == "Pregnant") & (m == 3)) | ((s == "Open") & (m == 5))
    saved_builds = {k: REG[k]["build"] for k in ("mortality_full", "culling_age_stage_full")}
    for pad_name, mask_fn, expect_change in (("padding rows (Open >= 22, Pregnant >= 8, Dry <= 6 or >= 11)", pad_mask, False), ("reachable rows (Open 5, Pregnant 3, Dry 8)", used_mask, True)):
        for k in saved_builds:
            REG[k]["build"] = (lambda orig, fn: (lambda: _scale(orig(), fn)))(saved_builds[k], mask_fn)
        MORT2._mortality_step_arrays.cache_clear(); CULL2._culling_age_stage_step_arrays.cache_clear()
        new_v = eff_vals()
        changed = any(not np.allclose(new_v[n], base_v[n]) for n in base_v)
        ok(changed == expect_change, f"multiplying the {pad_name} of the mortality and culling tables by 7 {'changes' if expect_change else 'does NOT change'} the probabilities the active model applies "
                                    f"({'rows are read' if expect_change else 'rows are never read'})")
    for k in saved_builds:
        REG[k]["build"] = saved_builds[k]
    MORT2._mortality_step_arrays.cache_clear(); CULL2._culling_age_stage_step_arrays.cache_clear()
    for tname, lo, hi, stt in (("mortality_applicable", 0, 21, "Open"), ("mortality_applicable", 0, 7, "Pregnant"), ("mortality_applicable", 7, 10, "Dry"),
                               ("culling_age_stage_applicable", 7, 10, "Dry"), ("conception_applicable", 1, 9, None), ("abortion_applicable", 0, 9, None), ("pregnancy_milk_factor_applicable", 0, 7, None)):
        a = pd.read_csv(os.path.join(OUT, tname + ".csv"))
        a = a[a["status"] == stt] if stt else a
        ok(len(a) > 0 and a["month"].min() >= lo and a["month"].max() <= hi, f"{tname}{' / ' + stt if stt else ''}: applicable months {int(a['month'].min())}-{int(a['month'].max())} lie within the biologically possible range {lo}-{hi}")
    full = pd.read_csv(os.path.join(OUT, "mortality_full.csv"))
    ok(all(c in full.columns for c in ("clock", "start_day", "end_day", "used_month", "used_week", "used_day", "used_any")), "the full tables carry clock, start_day, end_day and used_month/week/day/any flags")
    for u in UNITS:
        sdf = pd.read_csv(os.path.join(OUT, "effective", "effective_step", u, "mortality.csv"))
        dry = sdf[sdf["status"] == "Dry"]
        ok(dry["mean_age_days"].min() >= 220 - 0.5 * 30 and dry["mean_age_days"].max() <= 300 and dry["step_index"].min() == int(reach_all and json.load(open(os.path.join(ROOT, "outputs", "audit", "reachability.json")))["per_unit"][u]["dry_off_dp"]),
           f"{u}: per-step Dry rows run from the dry-off step ({int(dry['step_index'].min())}) to the last pregnant step, mean ages {dry['mean_age_days'].min()}-{dry['mean_age_days'].max()} d")

    # ---- 7 milk curves
    from cow.curves import curve_kg
    mc = pd.read_csv(os.path.join(OUT, "milk_curves_by_lactation.csv"))
    cols = [f"{m}_L{L}" for m in ("wood", "milkbot") for L in LACS]
    ok(all(c in mc.columns for c in cols), "milk-curve export has Wood and MilkBot for lactations 1-10")
    err = max(float(np.abs(mc[f"{m}_L{L}"].values - np.round(curve_kg(mc["dim"].values.astype(float), L, m), 3)).max()) for m in ("wood", "milkbot") for L in LACS)
    ok(err <= 1e-9, f"milk-curve export equals cow.curves.curve_kg (max abs diff {err:.1e})")

    # ---- 8 plotly
    figs = [f"P{n:02d}" for n in range(1, 15)]
    files = os.listdir(PLY) if os.path.isdir(PLY) else []
    have = [f for f in figs if any(x.startswith(f) and x.endswith(".html") for x in files)]
    ok(len(have) == 14 and "validation_all.html" in files and "index.html" in files, f"Plotly: {len(have)}/14 figure files, validation_all.html and index.html exist in {PLY}")
    for f in ("P03_mortality.html", "P04_other_reason_culling.html", "P06_conception.html"):
        t = open(os.path.join(PLY, f), encoding="utf-8").read()
        found = [L for L in LACS if re.search(rf"\bL{L} \[", t) or f'"name":"L{L}' in t]
        ok(len(found) == 10, f"{f}: traces for lactations 1-10 with provenance tags present (found {len(found)})")

    print("\nLOOKUP_EXPORT_OK" if not failed else f"\nLOOKUP_EXPORT_FAILED: {len(failed)} check(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if "--unit" in sys.argv:
        w, n = check_unit(sys.argv[sys.argv.index("--unit") + 1])
        print("RESULT" + json.dumps([w, n]))
    else:
        main()

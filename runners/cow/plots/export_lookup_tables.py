"""
export_lookup_tables.py
=======================
Make every probability assumption of the model visible and traceable, for EVERY lactation 1..10.

    python runners/cow/plots/export_lookup_tables.py              # writes runners/cow/outputs/lookup_tables/ (run again after any parameter or table change)

Two kinds of table are written, deliberately kept apart:

  A. SOURCE / EVIDENCE tables  (runners/cow/outputs/lookup_tables/source_evidence/)
     The categories and values AS THE SOURCE REPORTS THEM: Pinedo 2010 parity 1/2/3/>=4, Miller 2008 per lactation, the conception
     lactation-1 reference, parity odds ratios, pregnancy-loss windows, gestation means, milk-curve parameters ... plus the later-lactation
     evidence in the reviewed papers that the model does NOT use (Hadley 2006, Rilanto 2020, Mwangi 2025 parity means).
     Legacy uncited bucket tables ('1 / 2 / 3+') are exported too, flagged INACTIVE.

  B. EFFECTIVE tables  (runners/cow/outputs/lookup_tables/effective/)
     The values the ACTIVE model uses for each individual lactation L1..L10, for every DIM / DP bucket and status:
       <mechanism>_effective_wide.csv      native monthly values (or days), columns L1..L10 and the row-level `basis_L*` labels
       effective_long.csv                  every mechanism in one tidy file with provenance class and source category per value
       provenance_matrix_wide/long.csv     mechanism x lactation: direct / adapted / grouped / repeated / placeholder / policy
       effective_step/<unit>/<mechanism>.csv   the per-step probabilities the model actually applies at MONTH / WEEK / DAY. These are
                                           evaluated by calling the ACTIVE resolver of each spec (cowmrp.mrp.engine.resolve_method), not
                                           re-implemented here.
  lactation_tables.html     everything on one page, colour-coded by provenance class (ten populated rows are NOT ten estimates)
  <table>_full.csv          the raw full tables exactly as the model reads them (cow/data overrides applied)

Provenance classes: cow/registry.py::lactation_provenance (single source, also used by the Plotly figures).

WHAT 'month' MEANS (every table): the 30-day bucket of the table's own clock. Open rows: months since CALVING (days in milk). Pregnant and Dry rows:
months of PREGNANCY (days pregnant / 30, an absolute pregnancy-day bucket; NOT time since dry-off). The full tables carry the same 37 months for every
status so that they can be edited uniformly; most of those rows can never be read. Every exported row therefore carries `start_day`, `end_day`, `clock`
and `used_month / used_week / used_day / used_any` (traced from the executed states by runners/cow/checks/verify_biological_consistency.py,
runners/cow/outputs/audit/reachability.json). `*_applicable*.csv` files keep only the rows the active model can read; the full files keep everything for auditing.
"""
import os, sys, json, html, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from runners._shared import common
OUT = common.module_outputs("cow", "lookup_tables")
SRC = os.path.join(OUT, "source_evidence")
EFF = os.path.join(OUT, "effective")
STEP = os.path.join(EFF, "effective_step")
UNITS = ("month", "week", "day")
LACS = list(range(1, 11))
COLOR = {"direct": "#c8e6c9", "adapted": "#dcedc8", "grouped": "#fff9c4", "repeated": "#ffe0b2", "placeholder": "#ffcdd2", "policy": "#d1c4e9", "sensitivity": "#e0e0e0"}


# ------------------------------------------------------------------------------------------ per-unit step export
def export_steps(unit):
    import numpy as np, pandas as pd
    import cowmrp.domains.cow.parameters as P
    from cowmrp.mrp.engine import resolve_method
    from cowmrp.domains.cow.probabilities import probability_specs
    from cowmrp.domains.cow.tables import get_tables
    spec = {s["name"]: s for s in probability_specs}
    U, t = P.UNIT_DAYS, get_tables()
    d = os.path.join(STEP, unit)
    os.makedirs(d, exist_ok=True)
    kmax = min(P.MAX_DIM, int(np.ceil(700 / U)))
    dpmax = int(t["dp_max"])

    def call(name, lac, dim, dp):
        df = pd.DataFrame({"lactation": lac, "dim": dim, "dp": dp})
        return np.asarray(resolve_method(spec[name])(df, {}), dtype=float)

    def wide(name, rows):
        """rows: list of (status, clock, step_index, mean_age_days, dim, dp) -> DataFrame with L1..L10 from the active resolver"""
        base = pd.DataFrame(rows, columns=["status", "clock", "step_index", "mean_age_days", "dim", "dp"])
        out = base[["status", "clock", "step_index", "mean_age_days"]].copy()
        out["reachable_step"] = np.where(base["status"] == "Open", base["step_index"] <= int(t["dim_max"]), True)     # DIM steps beyond dim_max never occur
        for L in LACS:
            out[f"L{L}"] = call(name, np.full(len(base), L), base["dim"].values, base["dp"].values)
        return out

    open_rows = [("Open", "days in milk", k, (k + 1) * U, k, 0) for k in range(kmax + 1)]
    preg_rows = [("Pregnant", "days pregnant", j, j * U, 0, j) for j in range(1, P.DRY_OFF_DP)]
    dry_rows = [("Dry", "days pregnant", j, j * U, 0, j) for j in range(P.DRY_OFF_DP, dpmax + 1)]
    for mech, name in (("mortality", "mortality"), ("other_reason_culling", "culling_age_stage")):
        wide(name, open_rows + preg_rows + dry_rows).to_csv(os.path.join(d, f"{mech}.csv"), index=False)
    wide("conception", [r for r in open_rows]).to_csv(os.path.join(d, "conception.csv"), index=False)
    wide("culling_repro", open_rows).rename(columns={}).to_csv(os.path.join(d, "milk_floor_cull.csv"), index=False)
    wide("abortion", [("Pregnant", "days pregnant", j, j * U, 0, j) for j in range(1, dpmax + 1)]).to_csv(os.path.join(d, "pregnancy_loss.csv"), index=False)
    wide("calving", [("Pregnant", "days pregnant", j, j * U, 0, j) for j in range(1, dpmax + 1)]).to_csv(os.path.join(d, "calving.csv"), index=False)
    print(f"[{unit}] step tables written ({kmax + 1} DIM steps, {dpmax} DP steps) to {d}")


# ------------------------------------------------------------------------------------------ native / source tables
def main():
    os.environ.setdefault("COW_TIME_UNIT", "week")
    import numpy as np
    import pandas as pd
    import cowmrp.domains.cow.parameters as P
    import cowmrp.domains.cow.probabilities, cowmrp.domains.cow.rewards, cowmrp.domains.cow.tables                       # register every table
    from cowmrp.domains.cow.lookup_tables import FULL_TABLE_REGISTRY, get_full_table, table_to_array, N_DIM_MONTHS, N_GEST_MONTHS, STATUSES
    from cowmrp.domains.cow.registry import check_provenance, lactation_provenance
    from cowmrp.domains.cow.evidence import pinedo2010 as PIN, miller2008 as MIL
    from cowmrp.domains.cow.curves import curve_kg
    import cowmrp.domains.cow.probabilities.mortality as MORT
    import cowmrp.domains.cow.probabilities.culling as CULL
    import cowmrp.domains.cow.probabilities.conception as CONC
    import cowmrp.domains.cow.probabilities.abortion as ABO

    prov = check_provenance()                                                 # raises if provenance and active tables disagree
    reach_path = common.module_outputs("cow", "audit", "reachability.json")
    if not os.path.isfile(reach_path):
        subprocess.run([sys.executable, os.path.join(ROOT, "runners", "cow", "checks", "verify_biological_consistency.py")], cwd=ROOT, capture_output=True, text=True)
    reach = json.load(open(reach_path))["used_months"]                       # {mechanism: {status: {unit: [months]}}}
    MECH_OF_TABLE = {"mortality_full": ("mortality", None), "culling_age_stage_full": ("other_reason_culling", None), "conception_full": ("conception", "Open"),
                     "abortion_full": ("pregnancy_loss", "Pregnant"), "pregnancy_milk_factor_full": ("pregnancy_milk_factor", "Pregnant"), "milk_income_full": (None, None)}

    def annotate(df, tname):
        """add clock, start_day, end_day and used_* flags (month = 30-day bucket of the row's own clock)"""
        d = df.copy()
        mech, fixed = MECH_OF_TABLE.get(tname, (None, None))
        stat = d["status"] if "status" in d.columns else pd.Series(fixed or "all", index=d.index)
        d["clock"] = np.where(stat == "Open", "days in milk", np.where(stat.isin(["Pregnant", "Dry"]), "days pregnant", "days in milk" if tname == "conception_full" else "days pregnant"))
        d["start_day"] = d["month"] * 30
        d["end_day"] = d["month"] * 30 + 30
        for u in UNITS:
            if mech is None:
                d[f"used_{u}"] = False
            else:
                d[f"used_{u}"] = [int(m) in set(reach[mech][s][u]) for m, s in zip(d["month"], (stat if fixed is None else pd.Series(fixed, index=d.index)))]
        d["used_any"] = d[[f"used_{u}" for u in UNITS]].any(axis=1)
        d["use_note"] = np.where(d["used_any"], "read by the active model", "NOT READ by the active model (unreachable status/time combination or table padding)" if mech else "inactive template: the active milk income is the Wood/MilkBot formula")
        return d
    for d in (OUT, SRC, EFF, STEP):
        os.makedirs(d, exist_ok=True)
    for f in os.listdir(OUT):                                                 # remove stale top-level files from earlier exports
        if f.endswith((".csv", ".html", ".md")) and os.path.isfile(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))

    # ---- raw full tables
    for name in sorted(FULL_TABLE_REGISTRY):
        df = annotate(get_full_table(name), name).sort_values([c for c in ("status", "lactation", "month") if c in get_full_table(name).columns])
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)                                   # full table + flags (technical audit)
        df[df["used_any"]].to_csv(os.path.join(OUT, f"{name.replace('_full', '')}_applicable.csv"), index=False)   # only rows the model can read

    # ================================================== A. SOURCE / EVIDENCE
    src_tables = {}

    def put(name, df, note):
        df.to_csv(os.path.join(SRC, name + ".csv"), index=False)
        src_tables[name] = (df, note)

    rows = []
    for code, by_par in PIN.OPEN.items():
        for par, vals in by_par.items():
            for m, v in enumerate(vals):
                rows.append({"code": code, "parity_group": f"{par}" if par < 4 else ">=4", "clock": "days in milk", "month": m, "start_day": m * 30, "end_day": m * 30 + 30, "risk_per_100000_cow_days": v})
    for code, by_par in PIN.PREGNANT.items():
        for par, vals in by_par.items():
            for m, v in enumerate(vals):
                rows.append({"code": code, "parity_group": f"{par}" if par < 4 else ">=4", "clock": "days pregnant", "month": m, "start_day": m * 30, "end_day": m * 30 + 30, "risk_per_100000_cow_days": v})
    put("pinedo2010_hazards_source", pd.DataFrame(rows),
        "Pinedo, De Vries & Webb 2010 JDS 93:2250 Figs 4-7 (open cows by days after calving, pregnant cows by days pregnant). Parity groups 1, 2, 3, >=4 ONLY. Values read from the graphs at 30-day bins (+/-10-15 %). 'low production' and 'reproduction' codes are not used (the milk-floor rule represents them).")
    put("miller2008_death_per_lactation_source", pd.DataFrame([{"parity": (str(k) if k < 8 else ">=8"), "death_percent_per_lactation": v} for k, v in MIL.DEATH_PER_LACTATION_PCT.items()]),
        "Miller et al. 2008 JDS 91:3710 Table 2: lactation-specific for 1-7, one category '>=8'. A lower bound (deaths before the first test day are missed). Used only as RATIOS to spread Pinedo's '>=4' hazard over lactations 4-10.")
    leg = []
    for tname, tdf, active in (("mortality_table", MORT.mortality_table, "INACTIVE: feeds only the mortality 'formula' slot (not the default)"),
                               ("culling_table", CULL.culling_table, "INACTIVE: feeds only the culling_age_stage 'formula' slot (not the default)"),
                               ("conception_table", CONC.conception_table, "lactation_bucket '1' rows = the ACTIVE lactation-1 reference level; '2+' rows are NOT used by the default (lactations >=2 come from odds ratios)"),
                               ("abortion_table", ABO.abortion_table, "INACTIVE: original uncited table, unwired")):
        t = tdf.copy(); t.insert(0, "table", tname); t["status_in_model"] = active; leg.append(t)
    put("legacy_bucket_tables", pd.concat(leg, ignore_index=True, sort=False),
        "The original UNCITED bucket tables ('1', '2', '3+', '2+'). These are where the grouped lactation buckets live. They are placeholders; only the conception lactation-'1' rows are active (as the lactation-1 reference). Every other mechanism no longer reads them.")
    put("conception_parity_effect_source", pd.DataFrame([
        {"source": "Cook & Green 2016 JDS 99:4699 (Tables 1-2, 8,750 UK cows, herds selected for POOR fertility)", "quantity": "odds of pregnancy by 100 / 150 DIM vs parity 1", "parity_2": "0.67 / 0.68", "parity_3": ">=3: 0.53 / 0.47", "parity_4plus": "(grouped in >=3)", "used": "yes (0.68, 0.47)"},
        {"source": "Cordeiro 2026 Theriogenology 265:118076 (Table 2, 9,178 services)", "quantity": "adjusted OR pregnancy at day 28 vs parity 1", "parity_2": "0.69", "parity_3": ">=3: 0.46", "parity_4plus": "(grouped in >=3)", "used": "cross-check"},
        {"source": "Remnant 2018 Theriogenology 114:324 (Table 2, 257,396 inseminations)", "quantity": "OR conception per insemination vs parity 1, conditional on inter-service interval", "parity_2": "0.97", "parity_3": "0.94 (parity 4: 0.92; 5+: 0.75)", "parity_4plus": "0.92 / 0.75", "used": "NO: a different quantity (per insemination); recorded as insight"},
        {"source": "Fodor 2019 CJVR 83:218", "quantity": "first-service conception 24.8 % primiparous vs 17.3 % multiparous", "parity_2": "(multiparous grouped)", "parity_3": "(grouped)", "parity_4plus": "(grouped)", "used": "direction only"}]),
        "Parity effects on conception as reported. The model uses 0.68 (lactation 2) and 0.47 (>=3, applied to lactation 3 and repeated for 4-10) on the odds scale. No source in the folder reports lactations 4-10 separately except Remnant's per-insemination ORs (different quantity).")
    put("pregnancy_loss_source", pd.DataFrame(
        [{"window_days": f"{w['days'][0]}-{w['days'][1]}", "loss_rate": w["loss_rate"], "source": "Albaaj 2022 JDS Commun 4:144 (meta-analysis, pooled over parities)", "used": "yes"} for w in P.ABORTION_ALBAAJ2022["published"]["windows"]] +
        [{"window_days": f"{w['days'][0]}-{w['days'][1]}", "loss_rate": round(w["loss_rate"], 5), "source": w["source"] + " (Lee & Kim 2007 J Vet Sci 8:283: 34/988 and 22/954, pooled over parities)", "used": "yes"} for w in P.ABORTION_ALBAAJ2022["outside_published_range"]["late_windows"]] +
        [{"window_days": k, "loss_rate": v, "source": "legacy uncited monthly rate", "used": "yes (days 1-18 and 271+ only)"} for k, v in P.ABORTION_ALBAAJ2022["outside_published_range"]["legacy_monthly_rates"].items() if k in ("1-18d", "271+d")]),
        "Pregnancy-loss windows by DAYS PREGNANT (loss rate conditional on being pregnant at window start, except Lee & Kim = proportion of all pregnancies). NO source reports a lactation-specific value that the model uses: all are pooled across parities.")
    put("gestation_length_source", pd.DataFrame([
        {"source": "Mwangi 2025 JDS 108:8685 Table 1 (9,750 gestations)", "dam_parity_at_calving": "2", "n": 3378, "mean_days_unadjusted": 278.43, "mean_days_adjusted_for_gestation_EBV": 277.11, "used": "NO"},
        {"source": "Mwangi 2025 Table 1", "dam_parity_at_calving": "3", "n": 2462, "mean_days_unadjusted": 278.94, "mean_days_adjusted_for_gestation_EBV": 277.52, "used": "NO"},
        {"source": "Mwangi 2025 Table 1", "dam_parity_at_calving": "4", "n": 1701, "mean_days_unadjusted": 279.40, "mean_days_adjusted_for_gestation_EBV": 277.99, "used": "NO"},
        {"source": "Mwangi 2025 Table 1", "dam_parity_at_calving": "5-7", "n": 2004, "mean_days_unadjusted": 279.83, "mean_days_adjusted_for_gestation_EBV": 278.29, "used": "NO"},
        {"source": "Mwangi 2025 Table 1", "dam_parity_at_calving": "8-12", "n": 208, "mean_days_unadjusted": 279.78, "mean_days_adjusted_for_gestation_EBV": 278.01, "used": "NO"},
        {"source": "Basiel 2024 JDS 107:2241 (75,256 Holstein-sired lactations)", "dam_parity_at_calving": "pooled", "n": 75256, "mean_days_unadjusted": 277.0, "mean_days_adjusted_for_gestation_EBV": None, "used": "YES: GESTATION_LENGTH_DAYS = 277 for all lactations"}]),
        "Gestation length. A cow in lactation L calves into parity L+1. The model uses ONE value (277 d) for every lactation; Mwangi reports a 1.3 d parity spread (adjusted 277.1-278.3) that is not used (smaller than one WEEK step).")
    put("later_lactation_evidence_NOT_USED", pd.DataFrame([
        {"source": "Hadley 2006 JDS 89:2286 Table 5 (10 US states, DHI)", "quantity": "% of cows culled per lactation (all culls, excl. sold for dairy)", **{f"L{i}": v for i, v in enumerate([17.9, 31.3, 37.3, 42.9, 47.3, 51.1, 54.4, 57.2, 60.2, 61.8], 1)}, "note": "lactation-specific 1..10+ (L10 = '10+')"},
        {"source": "Hadley 2006 Table 5", "quantity": "% health culls per lactation (udder, mastitis, lameness, injury, disease, reproduction; not death)", **{f"L{i}": v for i, v in enumerate([13.1, 23.2, 28.0, 32.5, 35.9, 39.0, 41.6, 43.8, 46.3, 46.3], 1)}, "note": "lactation-specific"},
        {"source": "Hadley 2006 Table 5", "quantity": "% mortality per lactation", **{f"L{i}": v for i, v in enumerate([1.9, 3.4, 4.5, 5.4, 6.0, 6.3, 6.6, 6.6, 6.9, 7.5], 1)}, "note": "lactation-specific; the model's death per lactation is close (see audit section 29)"},
        {"source": "Rilanto 2020 BMC Vet Res 16:173 Table 4 (Estonia, 107,835 multiparous cows)", "quantity": "culling hazard ratio vs parity 2 (all causes, death + slaughter)", "L1": None, "L2": 1.00, "L3": 1.38, "L4": 1.81, "L5": 2.22, "L6": 2.49, "L7": 3.13, "L8": 3.13, "L9": 3.13, "L10": 3.13, "note": "parity-specific to 6, '>=7' one category (3.13)"},
        {"source": "Pinedo 2010 / Pinedo 2014 JDS 97:2886", "quantity": "parity categories", "L1": "1", "L2": "2", "L3": "3", "L4": ">=4", "L5": ">=4", "L6": ">=4", "L7": ">=4", "L8": ">=4", "L9": ">=4", "L10": ">=4", "note": "cannot separate lactations 4-10"}]),
        "Later-lactation evidence present in the reviewed PDFs that the model does NOT currently use as input. Hadley and Rilanto both show culling still rising across lactations 5-10, whereas the model repeats Pinedo's '>=4' hazard for 5-10 (class 'repeated'). for the proposed correction and its effect (not applied).")
    ws = P.WOOD1967["parameters_by_parity"]
    put("milk_curve_source", pd.DataFrame(
        [{"curve": "Wood (active)", "lactation": k, "a": v["a"], "b (derived by this project)": round(v["b"], 5), "c (derived by this project)": round(v["c"], 6), "source": "Evangelista 2024: a, time-to-peak, peak printed; b, c derived", "lactations_used_for": f"{k}" if k < 3 else "3-10 (repeated)"} for k, v in ws.items()] +
        [{"curve": "MilkBot (alternative)", "lactation": k, "a": v["a"], "b (derived by this project)": v["b"], "c (derived by this project)": v["c"], "source": f"Ehrlich 2011 Table 1 Holstein median, parity {k}; decay d = {v['d']}", "lactations_used_for": f"{k}" if k < 3 else "3-10 (repeated)"} for k, v in P.MILKBOT_EHRLICH2011.items()]),
        "Milk-curve source parameters (lactations 1-3 only). Lactations 4-10 repeat lactation 3.")

    # ================================================== B. EFFECTIVE
    wide_tables = {}                                   # mechanism -> (DataFrame, unit label)
    long_rows = []

    def clsof(mech, L):
        return prov[mech][L]

    def add_long(mech, df, valcol_prefix, unit, status_col="status"):
        for L in LACS:
            cls, cat, srcname, note = prov[mech][L]
            for _, r in df.iterrows():
                long_rows.append({"mechanism": mech, "status": r.get("status", ""), "clock": r.get("clock", ""), "month": r.get("month", ""), "start_day": r.get("start_day", ""),
                                  "end_day": r.get("end_day", ""), "lactation": L, "value": r[f"{valcol_prefix}{L}"], "unit": unit, "row_basis": r.get(f"basis_L{L}", ""),
                                  "provenance_class": cls, "source_category": cat})

    def native_wide(mech, tname, col, unit, statuses, months_by_status=None):
        df = get_full_table(tname)
        rows = []
        for st in statuses:
            sub = df if st is None else df[df["status"] == st]
            for m in sorted(sub["month"].unique()):
                if months_by_status and st in months_by_status and m > months_by_status[st]:
                    continue
                r = {"status": st or MECH_OF_TABLE[tname][1], "clock": ("days in milk" if st in (None, "Open") and tname != "abortion_full" and tname != "pregnancy_milk_factor_full" else "days pregnant"),
                     "month": int(m), "start_day": int(m * 30), "end_day": int(m * 30 + 30)}
                for L in LACS:
                    row = sub[(sub["lactation"] == L) & (sub["month"] == m)].iloc[0]
                    r[f"L{L}"] = float(row[col]); r[f"basis_L{L}"] = row["basis"]
                rows.append(r)
        out = pd.DataFrame(rows)
        # flags per row from the reachability trace
        mech_key, fixed = MECH_OF_TABLE[tname]
        for u in UNITS:
            out[f"used_{u}"] = [int(m) in set(reach[mech_key][(s if fixed is None else fixed)][u]) for m, s in zip(out["month"], out["status"])]
        out["used_any"] = out[[f"used_{u}" for u in UNITS]].any(axis=1)
        out.to_csv(os.path.join(EFF, f"{mech}_effective_wide.csv"), index=False)
        out[out["used_any"]].to_csv(os.path.join(EFF, f"{mech}_effective_applicable_wide.csv"), index=False)
        wide_tables[mech] = (out, unit)
        add_long(mech, out[out["used_any"]], "L", unit)
        return out

    native_wide("mortality", "mortality_full", "p", "probability of death per 30 days", STATUSES)
    native_wide("other_reason_culling", "culling_age_stage_full", "p", "probability of other-reason culling per 30 days", STATUSES)
    native_wide("conception", "conception_full", "p", "monthly pregnancy probability of an eligible open cow (waiting period day 50 and last breeding day 300 applied by day)", (None,))
    ab = native_wide("pregnancy_loss", "abortion_full", "p", "monthly equivalent of the ACTIVE day hazard (Albaaj + Lee & Kim), loss probability per 30 days", (None,))
    native_wide("pregnancy_milk_factor", "pregnancy_milk_factor_full", "factor", "milk multiplier (1 = open cow)", (None,))

    # calving: deterministic gestation length per lactation
    cal = pd.DataFrame([{"status": "Pregnant", "clock": "days pregnant", "month": "", "start_day": "", "end_day": "", **{f"L{L}": float(P.GESTATION_LENGTH_DAYS) for L in LACS}, **{f"basis_L{L}": "repeated" for L in LACS}}])
    cal.to_csv(os.path.join(EFF, "calving_effective_wide.csv"), index=False); wide_tables["calving"] = (cal, "gestation length, days (deterministic, active default)")
    add_long("calving", cal, "L", "gestation length, days")

    # milk floor cull
    tab = cowmrp.domains.cow.tables.get_tables()
    fl = []
    for key, lab, fn in (("cull_day", "crossing day: first day >= 300 with potential milk <= floor (days in milk)", lambda L: float(tab["cull_day"][L - 1])),
                         ("milk300", "potential milk at day 300 (kg/day)", lambda L: float(curve_kg(np.array([300.5]), L)[0])),
                         ("floor", "milk floor (kg/day, farmer parameter)", lambda L: float(P.MILK_FLOOR_KG))):
        fl.append({"status": "Open only", "clock": "days in milk", "month": "", "start_day": "", "end_day": "", "quantity": lab, **{f"L{L}": fn(L) for L in LACS}, **{f"basis_L{L}": "policy" for L in LACS}})
    fl = pd.DataFrame(fl); fl.to_csv(os.path.join(EFF, "milk_floor_cull_effective_wide.csv"), index=False); wide_tables["milk_floor_cull"] = (fl, "policy rule; the per-step probability is 0 before the crossing day and 1 from it on (see effective_step)")
    # milk curves
    mrows = []
    for b in range(0, 21):
        r = {"status": "Open", "clock": "days in milk", "month": b, "start_day": b * 30, "end_day": b * 30 + 30}
        for L in LACS:
            r[f"L{L}"] = float(curve_kg(np.arange(b * 30, b * 30 + 30) + 0.5, L).mean()); r[f"basis_L{L}"] = P.MILK_CURVE_BASIS[P.MILK_CURVE_METHOD][L]
        mrows.append(r)
    mc = pd.DataFrame(mrows); mc.to_csv(os.path.join(EFF, "milk_curve_effective_wide.csv"), index=False); wide_tables["milk_curve"] = (mc, f"mean potential milk of an open cow over the 30 days, kg/day ({P.MILK_CURVE_METHOD} curve)")
    add_long("milk_curve", mc, "L", "kg/day")
    for mech in ("milk_floor_cull",):
        pass

    # milk curve by day, both methods (all lactations)
    days = np.arange(0, 601, 5)
    cur = {"dim": days}
    for method in ("wood", "milkbot"):
        for L in LACS:
            cur[f"{method}_L{L}"] = np.round(curve_kg(days.astype(float), L, method), 3)
    pd.DataFrame(cur).to_csv(os.path.join(EFF, "milk_curves_by_lactation.csv"), index=False)
    pd.DataFrame(cur).to_csv(os.path.join(OUT, "milk_curves_by_lactation.csv"), index=False)
    # active pregnancy-loss day hazard
    dd = np.arange(0, 291)
    pd.DataFrame({"days_pregnant": dd, "daily_loss_hazard": ABO._albaaj_daily_hazard(dd.astype(float))}).to_csv(os.path.join(EFF, "pregnancy_loss_active_daily_hazard.csv"), index=False)

    # provenance matrices
    pl = pd.DataFrame([{"mechanism": m, "lactation": L, "provenance_class": r[0], "source_category": r[1], "source": r[2], "note": r[3]} for m, byl in prov.items() for L, r in byl.items()])
    pl.to_csv(os.path.join(EFF, "provenance_matrix_long.csv"), index=False)
    pw = pl.pivot(index="mechanism", columns="lactation", values="provenance_class"); pw.columns = [f"L{c}" for c in pw.columns]
    pw.reset_index().to_csv(os.path.join(EFF, "provenance_matrix_wide.csv"), index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(EFF, "effective_long.csv"), index=False)

    # per-unit step tables (the values the model applies), via the active resolvers, one process per resolution
    for u in UNITS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--steps", u], env=dict(os.environ, COW_TIME_UNIT=u), cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-2000:]); sys.exit(1)
        print(p.stdout.strip().splitlines()[-1])

    # ================================================== HTML
    def esc(x):
        return html.escape(str(x))

    def fmt(v):
        return f"{v:.5g}" if isinstance(v, (int, float, np.floating)) else esc(v)

    h = ["<html><head><meta charset='utf-8'><title>CowMRP lookup tables, lactations 1-10</title>",
         "<style>body{font-family:sans-serif;max-width:1700px;margin:auto;padding:0 12px}table{border-collapse:collapse;font-size:12px;margin:6px 0 18px}td,th{border:1px solid #bbb;padding:2px 6px;text-align:right}"
         "th{background:#eee}td.t{text-align:left}h2{margin-top:34px;border-top:2px solid #444;padding-top:8px}.small{font-size:12px;color:#444}.leg span{padding:2px 8px;margin-right:6px}</style></head><body>",
         "<h1>CowMRP lookup tables: lactations 1-10, source evidence and effective model values</h1>",
         "<p>Generated by <code>runners/cow/plots/export_lookup_tables.py</code> from the model's active tables and resolvers (nothing illustrative). <b>Ten populated rows are not ten independent estimates:</b> the colour of every lactation column is its provenance class.</p>",
         "<p class='leg'>" + "".join(f"<span style='background:{c}'>{k}</span>" for k, c in COLOR.items()) + "</p>",
         "<p class='small'><b>direct</b> source reports this lactation/parity (graph-read values carry +/-10-15 %) &middot; <b>adapted</b> documented transformation of a source value &middot; <b>grouped</b> expanded from a grouped source category (3+, &gt;=4, &gt;=8) &middot; <b>repeated</b> another lactation's / pooled value copied, no distinct estimate &middot; <b>placeholder</b> no evidence &middot; <b>policy</b> management rule.</p>",
         "<h2 id='prov'>1. Provenance matrix (mechanism x lactation)</h2><table><tr><th>mechanism</th>" + "".join(f"<th>L{L}</th>" for L in LACS) + "</tr>"]
    for m, byl in prov.items():
        h.append(f"<tr><td class='t'>{esc(m)}</td>" + "".join(f"<td style='background:{COLOR[byl[L][0]]}' title='{esc(byl[L][1])} | {esc(byl[L][2])} | {esc(byl[L][3])}'>{byl[L][0]}</td>" for L in LACS) + "</tr>")
    h.append("</table><p class='small'>Hover a cell for source category, source and note. Full text: effective/provenance_matrix_long.csv.</p>")
    h.append("<h2 id='src'>2. Source / evidence tables (as the sources report them)</h2>")
    for name, (df, note) in src_tables.items():
        h.append(f"<h3>{esc(name)}</h3><p class='small'>{esc(note)}</p>")
        show = df if len(df) <= 60 else df.head(60)
        h.append("<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in show.columns) + "</tr>" + "".join("<tr>" + "".join(f"<td>{fmt(v) if v is not None and v == v else ''}</td>" for v in r) + "</tr>" for r in show.itertuples(index=False)) + "</table>")
        if len(df) > 60:
            h.append(f"<p class='small'>(first 60 of {len(df)} rows; full file: source_evidence/{esc(name)}.csv)</p>")
    h.append("<h2 id='eff'>3. Effective model tables, lactations 1-10 (native monthly values; the per-step values the model applies are in effective/effective_step/&lt;unit&gt;/)</h2>")
    for mech, (df, unit) in wide_tables.items():
        n_unused = int((~df["used_any"]).sum()) if "used_any" in df.columns else 0
        h.append(f"<h3>{esc(mech)}</h3><p class='small'>{esc(unit)}. Colour = provenance class of that lactation for this mechanism. Rows shown are those the active model can read at MONTH / WEEK / DAY (columns used_*)."
                 + (f" {n_unused} further table rows exist (padding, status/time combinations that cannot occur) and are in the CSV, flagged used_any = False." if n_unused else "") + "</p><table><tr>"
                 + "".join(f"<th>{esc(c)}</th>" for c in df.columns if not c.startswith("basis_")) + "</tr>")
        cols = [c for c in df.columns if not c.startswith("basis_")]
        for _, r in (df[df["used_any"]] if "used_any" in df.columns else df).iterrows():
            cells = []
            for c in cols:
                if c.startswith("L") and c[1:].isdigit():
                    L = int(c[1:]); cls = prov[mech][L][0]
                    cells.append(f"<td style='background:{COLOR[cls]}' title='{esc(cls)} | row basis: {esc(r.get('basis_' + c, ''))}'>{fmt(r[c])}</td>")
                else:
                    cells.append(f"<td class='t'>{fmt(r[c])}</td>")
            h.append("<tr>" + "".join(cells) + "</tr>")
        h.append("</table>")
    h.append("<h2>4. Not shown in this page</h2><p class='small'>Per-step probabilities at MONTH / WEEK / DAY: effective/effective_step/&lt;unit&gt;/*.csv. Raw full tables: *_full.csv. Interactive validation figures: ../validation/plotly/index.html.</p></body></html>")
    open(os.path.join(OUT, "lactation_tables.html"), "w", encoding="utf-8").write("\n".join(h))
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
        "# CowMRP lookup tables (generated by `runners/cow/plots/export_lookup_tables.py`; do not edit)\n\n"
        "- `lactation_tables.html` open this first: provenance matrix, source tables, effective tables for lactations 1-10, colour-coded.\n"
        "- `source_evidence/` the evidence as the sources report it (grouped categories kept), plus later-lactation evidence NOT used (`later_lactation_evidence_NOT_USED.csv`), plus the legacy uncited `1 / 2 / 3+` bucket tables (flagged INACTIVE).\n"
        "- `effective/` per-lactation L1..L10 values the model uses: `*_effective_wide.csv` (native monthly), `effective_long.csv`, `provenance_matrix_*.csv`, `effective_step/<unit>/*.csv` (per-step values from the active resolvers).\n"
        "- `*_full.csv` the raw full tables as the model reads them (a `cow/data/<name>.csv` override would be applied), with added columns `clock`, `start_day`, `end_day`, `used_month/week/day/any`, `use_note`. `*_applicable.csv` keeps only the rows the model can read.\n"
        "- `month` = 30-day bucket of the row's own clock: months since CALVING for Open rows, months of PREGNANCY (days pregnant / 30, absolute, NOT time since dry-off) for Pregnant and Dry rows. The full table has the same 37 months for every status; most Pregnant/Dry rows outside the real gestation period are padding that no state can read (`used_any = False`).\n\n"
        "Provenance classes: direct, adapted, grouped, repeated, placeholder, policy (definitions in the HTML and in `cow/registry.py`). Populated rows for lactations 4-10 are mostly `grouped` or `repeated`: they are NOT independent estimates.\n")
    print(f"wrote {sum(len(f) for _, _, f in os.walk(OUT))} files under {OUT}")


if __name__ == "__main__":
    if "--steps" in sys.argv:
        export_steps(sys.argv[sys.argv.index("--steps") + 1])
    else:
        main()

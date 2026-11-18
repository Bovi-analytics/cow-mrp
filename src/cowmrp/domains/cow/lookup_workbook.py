"""
cow/lookup_workbook.py -- edit the lookup tables in Excel WITHOUT making the model read Excel.

    python -m cow.lookup_workbook export [file.xlsx]    write every active lookup table (lactations 1-10) to ONE workbook
    python -m cow.lookup_workbook import [file.xlsx]    read your edits back, validate, and store them as CSV overrides in cow/data/
    python -m cow.lookup_workbook status                list the overrides currently active
    python -m cow.lookup_workbook reset [--yes]         delete all overrides (back to the built-in defaults)

HOW IT STAYS FAST
  The model never opens an Excel file. Excel is only the editing format.
    export : DataFrames (cow.lookup_tables.get_full_table) -> workbook          (once, when you want to look or edit)
    import : workbook -> validated DataFrames -> cow/data/<table>.csv           (once, after you edit)
    runtime: get_full_table() reads the CSV (or builds the default) ONCE per process and keeps the DataFrame in memory;
             the per-step probability arrays derived from it are cached too. A run therefore never waits for Excel.
  A table whose values and labels equal the built-in default is not written (and any old override for it is removed), so
  cow/data/ only ever contains real changes.

WHAT YOU CAN EDIT
  One sheet per table (mortality_full, culling_age_stage_full, conception_full, abortion_full, pregnancy_milk_factor_full,
  milk_income_full). Edit the yellow VALUE column (p = monthly probability, factor, income_per_day), and set `basis` (source /
  derived / repeated / placeholder) and `note` to what is TRUE for your values. Key columns (status, lactation, month) and the grey
  information columns (clock, start_day, end_day, used_by_model) are not read back. Every lactation 1-10 x month (x status) row must
  stay. `used_by_model = no` marks rows the model can never read (padding, e.g. Dry months 0-6): editing them has no effect.
  A `PROVENANCE` sheet shows, per mechanism and lactation, whether the value is direct, adapted, grouped, repeated or a placeholder.
  Each lactation 1-10 row is independent: changing lactation 1 does not change the lactation 2-10 rows (the odds-ratio rule that built them is not re-run).
  Requires the openpyxl package (already installed with pandas' Excel support).
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")                      # package data, ships with the wheel
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # repo root; only meaningful in a source checkout (development paths below)
DEFAULT_XLSX = os.path.join(ROOT, "runners", "cow", "outputs", "lookup_tables", "CowMRP_lookup_tables.xlsx")
BASIS = ["source", "derived", "repeated", "placeholder"]
BASIS_FILL = {"source": "C8E6C9", "derived": "FFF9C4", "repeated": "FFE0B2", "placeholder": "FFCDD2"}


def _tables():
    import cowmrp.domains.cow.probabilities, cowmrp.domains.cow.rewards, cowmrp.domains.cow.tables                       # noqa: F401  (registers every full table)
    from cowmrp.domains.cow.lookup_tables import FULL_TABLE_REGISTRY
    return FULL_TABLE_REGISTRY


def _used_flags(name):
    """{(status, month): 'yes'/'no'} from runners/cow/outputs/audit/reachability.json if it exists (runners/cow/checks/verify_biological_consistency.py)"""
    path = os.path.join(ROOT, "runners", "cow", "outputs", "audit", "reachability.json")
    mech = {"mortality_full": ("mortality", None), "culling_age_stage_full": ("other_reason_culling", None), "conception_full": ("conception", "Open"),
            "abortion_full": ("pregnancy_loss", "Pregnant"), "pregnancy_milk_factor_full": ("pregnancy_milk_factor", "Pregnant")}.get(name)
    if not os.path.isfile(path) or mech is None:
        return None
    used = json.load(open(path))["used_months"][mech[0]]
    return lambda status, month: "yes" if any(int(month) in set(used[(status if mech[1] is None else mech[1])][u]) for u in used[(status if mech[1] is None else mech[1])]) else "no"


def export_workbook(path=None):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    from cowmrp.domains.cow.lookup_tables import get_full_table
    from cowmrp.domains.cow.registry import lactation_provenance
    path = path or DEFAULT_XLSX
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reg = _tables()
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for i, line in enumerate(__doc__.strip().splitlines(), 1):
        ws.cell(i, 1, line)
    ws.column_dimensions["A"].width = 130
    ps = wb.create_sheet("PROVENANCE")
    prov = lactation_provenance()
    ps.append(["mechanism"] + [f"L{L}" for L in range(1, 11)] + ["source category / source (lactation 4 shown; see docs for all)"])
    for m, byl in prov.items():
        ps.append([m] + [byl[L][0] for L in range(1, 11)] + [f"{byl[4][1]} | {byl[4][2]}"])
    for c in ps[1]:
        c.font = Font(bold=True)
    for row in ps.iter_rows(min_row=2, min_col=2, max_col=11):
        for c in row:
            c.fill = PatternFill("solid", fgColor={"direct": "C8E6C9", "adapted": "DCEDC8", "grouped": "FFF9C4", "repeated": "FFE0B2", "placeholder": "FFCDD2", "policy": "D1C4E9"}.get(c.value, "FFFFFF"))
    ps.column_dimensions["A"].width = 26
    ps.column_dimensions["L"].width = 120
    for name, spec in reg.items():
        df = get_full_table(name)
        vcol = spec["value_col"]
        keys = ["status"] if "status" in df.columns else []
        keys += ["lactation", "month"]
        flag = _used_flags(name)
        out = df[keys].copy()
        stat = df["status"] if "status" in df.columns else pd.Series("Open" if name == "conception_full" else "Pregnant", index=df.index)
        out["clock"] = np.where(stat == "Open", "days in milk", np.where(stat.isin(["Pregnant", "Dry"]), "days pregnant", "days in milk"))
        out["start_day"] = df["month"] * 30
        out["end_day"] = df["month"] * 30 + 30
        out[vcol] = df[vcol].astype(float)
        out["basis"] = df["basis"]
        out["note"] = df["note"]
        out["used_by_model"] = [flag(s, m) for s, m in zip(stat, df["month"])] if flag else ""
        out = out.sort_values(keys).reset_index(drop=True)
        sh = wb.create_sheet(name)
        sh.append(list(out.columns))
        for r in out.itertuples(index=False):
            sh.append(list(r))
        sh.freeze_panes = "A2"
        sh.auto_filter.ref = sh.dimensions
        cols = {c: i + 1 for i, c in enumerate(out.columns)}
        for c in sh[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="FFF59D" if c.value in (vcol, "basis", "note") else "E0E0E0")
        vc, bc = cols[vcol], cols["basis"]
        dv = DataValidation(type="list", formula1='"' + ",".join(BASIS) + '"', allow_blank=False)
        sh.add_data_validation(dv)
        for row in range(2, sh.max_row + 1):
            sh.cell(row, vc).fill = PatternFill("solid", fgColor="FFFDE7")
            b = sh.cell(row, bc)
            b.fill = PatternFill("solid", fgColor=BASIS_FILL.get(b.value, "FFFFFF"))
            dv.add(b)
            sh.cell(row, vc).number_format = "0.000000"
        for col, w in (("note", 90), ("clock", 14), ("basis", 12)):
            sh.column_dimensions[chr(64 + cols[col])].width = w
        for k in keys + ["start_day", "end_day", vcol, "used_by_model"]:
            sh.column_dimensions[chr(64 + cols[k])].width = 13
    wb.save(path)
    print(f"wrote {path}  ({len(reg)} table sheets + README + PROVENANCE)")
    return path


def import_workbook(path=None):
    from cowmrp.domains.cow.lookup_tables import validate_full_coverage
    path = path or DEFAULT_XLSX
    if not os.path.isfile(path):
        raise SystemExit(f"workbook not found: {path} (run `python -m cow.lookup_workbook export` first)")
    reg = _tables()
    os.makedirs(DATA_DIR, exist_ok=True)
    changed, problems = [], []
    for name, spec in reg.items():
        try:
            sh = pd.read_excel(path, sheet_name=name)
        except ValueError:
            problems.append(f"{name}: sheet missing in the workbook")
            continue
        vcol = spec["value_col"]
        keys = (["status"] if spec["statuses"] else []) + ["lactation", "month"]
        need = keys + [vcol, "basis", "note"]
        if any(c not in sh.columns for c in need):
            problems.append(f"{name}: missing columns {[c for c in need if c not in sh.columns]}")
            continue
        new = sh[need].copy()
        new["note"] = new["note"].fillna("").astype(str)
        new["used_by_model"] = sh["used_by_model"].astype(str).values if "used_by_model" in sh.columns else ""
        try:
            validate_full_coverage(new, name, spec["n_months"], spec["statuses"], vcol)
        except ValueError as e:
            problems.append(f"{name}: {str(e).splitlines()[0]}")
            continue
        default = spec["build"]()[need].copy()
        m = default.merge(new, on=keys, suffixes=("_default", "_new"))        # new carries used_by_model
        diff_val = ~np.isclose(m[f"{vcol}_default"].astype(float), m[f"{vcol}_new"].astype(float), rtol=0, atol=1e-12)
        diff_basis = m["basis_default"] != m["basis_new"]
        path_csv = os.path.join(DATA_DIR, f"{name}.csv")
        if not diff_val.any() and not diff_basis.any():
            if os.path.exists(path_csv):
                os.remove(path_csv)
                print(f"{name}: identical to the default -> old override removed")
            continue
        order = default[keys].merge(new.drop(columns=["used_by_model"]), on=keys, how="left")   # keep the default row order
        order.to_csv(path_csv, index=False)
        n_dead = int((diff_val & (m["used_by_model"] == "no")).sum())
        warn = []
        if diff_val.any() and not diff_basis[diff_val].any() and (m.loc[diff_val, "basis_new"].isin(["source", "derived"])).any():
            warn.append("values changed but `basis` still says source/derived: set it to what is true for your values")
        if n_dead:
            warn.append(f"{n_dead} edited row(s) are marked used_by_model = no (never read, no effect)")
        changed.append((name, int(diff_val.sum()), int(diff_basis.sum()), warn))
    if problems:
        print("NOT IMPORTED (fix the workbook):\n  " + "\n  ".join(problems))
    for name, nv, nb, warn in changed:
        print(f"{name}: {nv} value(s), {nb} basis label(s) changed -> cow/data/{name}.csv" + ("".join(f"\n    WARNING: {w}" for w in warn)))
    if not changed and not problems:
        print("no changes: all tables equal the built-in defaults")
    elif changed:
        print("Overrides are used by the next run of the model (new Python process).")
    return not problems


def status():
    reg = _tables()
    found = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".csv")] if os.path.isdir(DATA_DIR) else []
    if not found:
        print("no overrides: the model uses the built-in default tables")
    for f in found:
        n = f[:-4]
        if n in reg:
            from cowmrp.domains.cow.lookup_tables import get_full_table
            vcol = reg[n]["value_col"]
            keys = (["status"] if reg[n]["statuses"] else []) + ["lactation", "month"]
            m = reg[n]["build"]().merge(get_full_table(n), on=keys, suffixes=("_d", "_o"))
            print(f"{f}: {int((~np.isclose(m[vcol + '_d'], m[vcol + '_o'], rtol=0, atol=1e-12)).sum())} value(s) differ from the default")


def reset(confirm):
    found = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.isdir(DATA_DIR) else []
    if not confirm:
        print(f"would delete {found}; add --yes to do it")
        return
    for f in found:
        os.remove(os.path.join(DATA_DIR, f))
    print(f"deleted {len(found)} override file(s)")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = next((a for a in sys.argv[2:] if not a.startswith("--")), None)
    if cmd == "export":
        export_workbook(arg)
    elif cmd == "import":
        sys.exit(0 if import_workbook(arg) else 1)
    elif cmd == "status":
        status()
    elif cmd == "reset":
        reset("--yes" in sys.argv)
    else:
        print(__doc__)

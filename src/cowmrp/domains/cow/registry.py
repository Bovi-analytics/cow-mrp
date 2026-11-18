"""
cow/registry.py
===============
What stands behind every probability: one record per spec saying which of table | formula | model is available and
active, the evidence (source, class A exact / B adapted / C insight / D grouped-repeated), units, time basis,
dependencies, lactation coverage and validation status. Plain data + two helpers -- no framework.

`check_registry()` fails loudly if a probability spec has no record, a record names a method the spec does not offer,
or the recorded active method is not the spec's. `describe()` returns the registry as a DataFrame (printed by
runners/cow/checks/audit_registry.py).
"""

import pandas as pd

TIME_BASIS_TABLE = "native monthly probability per lactation x month -> per-step by the entry-offset rule (cow/lookup_tables.py)"

METHOD_REGISTRY = {
    "mortality": {
        "units": "probability per model step (table: per native 30-d month)",
        "time_basis": "Open: months since calving (DIM clock); Pregnant/Dry: months of pregnancy (DP clock); " + TIME_BASIS_TABLE,
        "evidence": {"table": "B: Pinedo 2010 Figs 4,6 'died' (read from graph) x Miller 2008 Table 2 ratio for lactations 4-10",
                     "formula": "placeholder: uncited bucket table scaled to one annual anchor (kept for comparison)",
                     "model": "unavailable: not yet fitted (no independent fit; a model built from the table would be circular)"},
        "lactations": "1-3 derived; 4-10 derived via Miller ratio; months > 17 (open) / > 9 (pregnant) repeated",
        "validation": "runners/cow/checks/audit_exits.py: death %/lactation vs Miller 2008, %/yr vs Pinedo 2010 Table 1",
    },
    "culling_age_stage": {
        "units": "probability per model step (table: per native 30-d month)",
        "time_basis": "Open: DIM clock; Pregnant/Dry: DP clock; " + TIME_BASIS_TABLE,
        "evidence": {"table": "B: Pinedo 2010 Figs 4-7, sum of feet/legs, injury/other, mastitis, disease, udder (read from graph); lactations >= 4 repeat the '>=4' curve (D)",
                     "formula": "placeholder: uncited bucket table scaled to a residual anchor (kept for comparison)",
                     "model": "unavailable"},
        "lactations": "1-3 derived; 4 derived; 5-10 repeated",
        "validation": "runners/cow/checks/audit_exits.py: live culling %/yr by parity vs Pinedo 2010 Table 1 (16.9/23.3/30.1/40.8)",
    },
    "culling_repro": {
        "units": "probability per model step", "time_basis": "DIM steps; milk-floor crossing day by threshold-crossing mass",
        "evidence": {"table": "unavailable", "formula": "policy: cull an open cow past the last breeding day when potential milk < MILK_FLOOR_KG (farmer parameter, no biology invented)", "model": "unavailable"},
        "lactations": "1-10 (from each lactation's curve)", "validation": "runners/cow/checks/verify_state_space.py; exit share in runners/cow/checks/audit_exits.py",
    },
    "conception": {
        "units": "probability per model step (table: per native month per eligible open cow)",
        "time_basis": "DIM clock; voluntary waiting period and last breeding day applied by day; " + TIME_BASIS_TABLE,
        "evidence": {"table": "lactation 1: placeholder level (CDCB-consistent); lactations >= 2: B, odds ratios 0.68 / 0.47 from Cook & Green 2016 and Cordeiro 2026",
                     "formula": "placeholder logistic (uncited)", "model": "unavailable: not yet fitted (a model built from the same odds ratios would just re-derive the table)"},
        "lactations": "1 placeholder; 2 derived; 3-10 repeated (OR for '>=3')",
        "validation": "runners/cow/checks/audit_repro.py: days open ~137 d (CDCB), net pregnant-by-DIM odds ratios 0.67 / 0.46",
    },
    "abortion": {
        "units": "probability per model step", "time_basis": "DP clock (months of pregnancy); " + TIME_BASIS_TABLE,
        "evidence": {"table": "monthly equivalent of the active hazard (editable)", "formula": "B: Albaaj 2022 windows (days 19-90) + Lee & Kim 2007 trimesters (days 91-270); pooled over parities",
                     "model": "unavailable: not yet fitted (parity evidence is mixed; a relative-risk sensitivity, p_abortion_parity_model, exists but is not wired in)"},
        "lactations": "pooled (all lactations equal) in the default", "validation": "cumulative loss, resolution consistency (runners/cow/checks/verify_time_discretization.py)",
    },
    "calving": {
        "units": "probability per model step", "time_basis": "DP clock; entry-offset convention",
        "evidence": {"table": "deterministic trigger at GESTATION_LENGTH_DAYS = 277 d (Mwangi 2025: Holstein-sired mean 277.3 d)", "formula": "logistic around 277 d (uncited width)",
                     "model": "A: Normal(277, SD 5.9 d) from 9,750 gestations (Mwangi 2025); effect on V -0.12 %, not the default"},
        "lactations": "all lactations equal (parity effect on mean 278.4-279.8 d is ~1.4 d, not modelled)", "validation": "runners/cow/checks/verify_state_space.py, calving mass checks",
    },
}


def check_registry():
    from cowmrp.domains.cow.probabilities import probability_specs
    problems = []
    for spec in probability_specs:
        rec = METHOD_REGISTRY.get(spec["name"])
        if rec is None:
            problems.append(f"probability spec '{spec['name']}' has no registry record")
            continue
        for m, impl in spec["methods"].items():
            avail = impl is not None
            says_unavailable = str(rec["evidence"].get(m, "")).startswith("unavailable")
            if avail == says_unavailable:
                problems.append(f"'{spec['name']}.{m}': registry says {'unavailable' if says_unavailable else 'available'} but the spec says {'available' if avail else 'None'}")
    if problems:
        raise ValueError("cow/registry.py is out of step with the specs:\n  " + "\n  ".join(problems))


def describe():
    from cowmrp.domains.cow.probabilities import probability_specs
    rows = []
    for spec in probability_specs:
        rec = METHOD_REGISTRY[spec["name"]]
        for m in ("table", "formula", "model"):
            rows.append({"spec": spec["name"], "method": m, "active": spec["method"] == m,
                         "available": spec["methods"][m] is not None, "evidence": rec["evidence"][m],
                         "depends_on": ", ".join(spec["depends_on"]), "units": rec["units"], "time_basis": rec["time_basis"],
                         "lactations": rec["lactations"], "validation": rec["validation"]})
    return pd.DataFrame(rows)


# =====================================================================================================
# PER-LACTATION PROVENANCE (lookup-table traceability)
# =====================================================================================================
# One record for EVERY mechanism and EVERY lactation 1..MAX_LACTATION saying where the effective value comes from. Ten populated
# rows are NOT ten independent estimates: this record is what tells them apart. Read by runners/cow/plots/export_lookup_tables.py (the
# effective-table export), runners/cow/plots/plot_validation_plotly.py (trace labels) and `check_provenance()`.
#
# Classes (exactly one per mechanism x lactation):
#   direct       the source reports a value for THIS lactation / parity (e.g. Pinedo 2010 parity 1, 2, 3); values read from a graph are
#                still 'direct' but carry the digitization caveat in the note
#   adapted      a documented transformation of a source value for this lactation (e.g. a parity odds ratio applied to a reference level)
#   grouped      the source reports one value for a GROUP of lactations (parity 3+, >=4, >=8) and it is expanded to this lactation;
#                the source category is named
#   repeated     no distinct estimate exists for this lactation: another lactation's (or a pooled) value is copied
#   placeholder  no evidence
#   policy       a management rule or farmer parameter, not an estimate of biology
#   sensitivity  selectable alternative, not active
CLASSES = ("direct", "adapted", "grouped", "repeated", "placeholder", "policy", "sensitivity")
# class -> the `basis` labels the active table may carry for that lactation (table basis is the finer, per-row label)
CLASS_TO_BASIS = {"direct": {"derived", "repeated"}, "adapted": {"derived", "repeated"}, "grouped": {"derived", "repeated"},
                  "repeated": {"repeated"}, "placeholder": {"placeholder"}, "policy": set(), "sensitivity": set()}


def _by_lac(rule, max_lac):
    return {L: rule(L) for L in range(1, max_lac + 1)}


def lactation_provenance(max_lac=None):
    """{mechanism: {lactation: (class, source_category, source, note)}} for the ACTIVE default methods."""
    import cowmrp.domains.cow.parameters as P
    n = max_lac or P.MAX_LACTATION
    GR = "graph-read (+/-10-15 %)"

    def mort_rule(L):
        if L <= 3:
            return ("direct", f"Pinedo 2010 parity {L}", "Pinedo, De Vries & Webb 2010 JDS 93:2250, Figs 4 (open, by DIM) and 6 (pregnant, by days pregnant), code 'died'", GR)
        if L <= 8:
            return ("grouped", "Pinedo 2010 parity >=4 curve x Miller 2008 Table 2 ratio",
                    f"Pinedo 2010 Figs 4, 6 (parity >=4 group) scaled by Miller et al. 2008 JDS 91:3710 Table 2 death per lactation (lactation {L})",
                    "hazard SHAPE is the grouped >=4 curve; the lactation-specific part is Miller's ratio" + ("; Miller reports '>=8' as one value" if L == 8 else ""))
        return ("repeated", "Miller 2008 '>=8' ratio repeated", "as lactation 8 (Miller's last category is >=8)", "no distinct estimate for lactations 9-10")

    def cull_rule(L):
        if L <= 3:
            return ("direct", f"Pinedo 2010 parity {L}", "Pinedo 2010 Figs 4-7, sum of feet/legs, injury/other, mastitis, disease, udder", GR)
        if L == 4:
            return ("grouped", "Pinedo 2010 parity >=4", "Pinedo 2010 Figs 4-7, parity >=4 group", "lactation 4 is the first member of the >=4 group; " + GR)
        return ("repeated", "Pinedo 2010 parity >=4 curve repeated", "as lactation 4",
                "no distinct estimate used. NOTE: Hadley 2006 Table 5 and Rilanto 2020 Table 4 report culling still rising through lactations 5-10")

    def conc_rule(L):
        if L == 1:
            return ("placeholder", "retained conception_table level", "CDCB-consistent aggregate, no direct estimate", "lactation-1 reference level and DIM shape")
        if L == 2:
            return ("adapted", "parity 2 odds ratio 0.68", "Cook & Green 2016 JDS 99:4699 (0.68 at 150 DIM); Cordeiro 2026 Theriogenology 265 (0.69)", "applied to the lactation-1 level on the odds scale")
        if L == 3:
            return ("grouped", "parity >=3 odds ratio 0.47", "Cook & Green 2016 (0.47); Cordeiro 2026 (0.46)", "both sources report parity >=3 as one group")
        return ("repeated", "parity >=3 odds ratio 0.47 repeated", "as lactation 3",
                "no distinct estimate used. Remnant 2018 per-insemination ORs 0.92 (parity 4) and 0.75 (5+) point the same way but are a different quantity")

    mort = _by_lac(mort_rule, n)
    cull = _by_lac(cull_rule, n)
    conc = _by_lac(conc_rule, n)
    loss = _by_lac(lambda L: ("repeated", "pooled over parities",
                              "Albaaj 2022 JDS Commun 4:144 (days 19-90) and Lee & Kim 2007 J Vet Sci 8:283 (days 91-260), both pooled over lactations",
                              "the same value is used for every lactation; days 1-18 and 271+ are a legacy placeholder"), n)
    calv = _by_lac(lambda L: ("repeated", "single mean gestation length, 277 d",
                              "Basiel 2024 (Holstein-sired 277 +/- 0.15 d); Mwangi 2025 JDS 108:8685 Table 1 reports parity-specific means (278.4-279.8 d unadjusted, 277.1-278.3 adjusted), NOT used",
                              "one deterministic gestation length for all lactations; the Normal(SD 5.9 d) model is sensitivity-only"), n)
    floor = _by_lac(lambda L: ("policy", "farmer milk floor 22.7 kg/d, checked from day 300", "management rule (cow/parameters.py MILK_FLOOR_KG, LAST_BREEDING_DIM_DAYS)",
                               "crossing day follows the lactation's milk curve (" + (f"lactation {L} curve" if L <= 3 else "lactation 3 curve repeated") + ")"), n)
    pmf = _by_lac(lambda L: ("placeholder", "linear decline to 0.70 at dry-off", "uncited modelling placeholder (PREGNANCY_DECLINE_FLOOR)", "identical for every lactation"), n)
    wood = _by_lac(lambda L: ("adapted", f"Evangelista 2024 lactation {L}", "Evangelista 2024 Pesq Agropec Bras 59 Table 4 (Brazil): a, time-to-peak, peak printed",
                              "b and c DERIVED by this project; non-US herd") if L <= 3 else ("repeated", "lactation 3 repeated", "as lactation 3", "no distinct estimate"), n)
    mbot = _by_lac(lambda L: ("direct", f"Ehrlich 2011 Table 1, Holstein median, parity {L}", "Ehrlich 2011 The Bovine Practitioner 45(1):88-95 Table 1 (US DHIA, 6.46 M lactations)",
                              "printed values") if L <= 3 else ("repeated", "lactation 3 repeated", "as lactation 3", "no distinct estimate"), n)
    milk = wood if P.MILK_CURVE_METHOD == "wood" else mbot
    return {"mortality": mort, "other_reason_culling": cull, "conception": conc, "pregnancy_loss": loss, "calving": calv,
            "milk_floor_cull": floor, "pregnancy_milk_factor": pmf, "milk_curve": milk, "milk_curve_wood": wood, "milk_curve_milkbot": mbot}


def check_provenance():
    """Every mechanism x lactation has one valid class, and the class agrees with the `basis` labels of the active full table."""
    import cowmrp.domains.cow.parameters as P
    import cowmrp.domains.cow.probabilities, cowmrp.domains.cow.tables          # noqa: F401  (registers every full table)
    from cowmrp.domains.cow.lookup_tables import get_full_table
    prov = lactation_provenance()
    table_of = {"mortality": "mortality_full", "other_reason_culling": "culling_age_stage_full", "conception": "conception_full",
                "pregnancy_loss": "abortion_full", "pregnancy_milk_factor": "pregnancy_milk_factor_full"}
    problems = []
    for mech, by_lac in prov.items():
        if sorted(by_lac) != list(range(1, P.MAX_LACTATION + 1)):
            problems.append(f"{mech}: provenance does not cover lactations 1..{P.MAX_LACTATION}")
        for L, rec in by_lac.items():
            if rec[0] not in CLASSES:
                problems.append(f"{mech} L{L}: unknown class {rec[0]!r}")
    for mech, tname in table_of.items():
        df = get_full_table(tname)
        for L, rec in prov[mech].items():
            labels = set(df[df["lactation"] == L]["basis"])
            allowed = set(CLASS_TO_BASIS[rec[0]])
            if mech == "pregnancy_loss":
                allowed |= {"placeholder"}              # days 1-18 and 271+ keep the legacy rate
            if mech == "conception" and L == 1:
                allowed = {"placeholder"}
            if not labels <= allowed:
                problems.append(f"{mech} L{L}: class '{rec[0]}' but table basis labels {sorted(labels)} (allowed {sorted(allowed)})")
    if problems:
        raise ValueError("cow/registry.py provenance is out of step with the active tables:\n  " + "\n  ".join(problems))
    return prov

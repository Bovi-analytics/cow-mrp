"""
cow/lookup_tables.py
====================
Full lookup tables and the one time-discretization rule that turns them into
per-step probabilities.

WHY THIS FILE EXISTS
--------------------
Every probability / reward spec has table | formula | model methods so any one
can be swapped when a better source appears. The TABLE method is a *full* table:
one explicit row for EVERY lactation 1..MAX_LACTATION and EVERY native month
(and every status where biology differs: Open / Pregnant / Dry) -- no "2+" or
"3+" buckets. Tables are stored in their NATIVE unit (monthly probabilities by
month since calving, or by month of gestation), independent of TIME_UNIT.

TIME DISCRETIZATION (single rule, used by every table)
------------------------------------------------------
A monthly probability p_m inside native month m means a constant daily hazard
h_m = 1 - (1 - p_m)^(1/RATE_PERIOD_DAYS) on each of that month's days. A model
step is UNIT_DAYS long and the animal enters a phase (calving, conception)
somewhere INSIDE the previous step, so her true age at step k is k*U + u,
u ~ Uniform(0, U). The per-step probability is the ratio of the entry-averaged
survival at the step's two boundaries (mean_step_probability). Month, week and
day resolutions therefore describe the same biology. (Same convention the
conception / abortion / calving code already used; moved here so tables share it.)

OVERRIDING A TABLE
------------------
`python -c "import cow.probabilities; from cow.lookup_tables import export_full_tables; export_full_tables('cow/data')"`
writes every default table as CSV. Edit a CSV in cow/data/ and it is picked up
automatically (coverage of every lactation x month x status is validated; a gap is
a hard error naming the missing rows, never silently filled).

Default values are the previous bucket values REPLICATED across lactations
(column `basis` = "bucket value replicated") -- no new biological number is
introduced here; per-lactation values are an open research item (see
docs/LIMITATIONS_REGISTER.md).
"""

import os
import numpy as np
import pandas as pd

from cow.parameters import (
    UNIT_DAYS, RATE_PERIOD_DAYS, MAX_LACTATION, MAX_DIM_DAYS, MAX_DP_DAYS,
)

MONTH = RATE_PERIOD_DAYS
N_DIM_MONTHS = int(np.ceil((MAX_DIM_DAYS + 3 * MONTH) / MONTH))     # months since calving covered
N_GEST_MONTHS = int(np.ceil((MAX_DP_DAYS + 3 * MONTH) / MONTH))     # months of gestation covered
STATUSES = ("Open", "Pregnant", "Dry")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# EVIDENCE TAXONOMY -- every row of every full table carries a `basis` code (and a free-text `note`):
#   source      the cited source reports exactly this cell (lactation / month / status)
#   derived     computed from source values (unit conversion, interpolation, pooling) -- not a new observation
#   repeated    a source value copied to cells the source does not distinguish; NO evidence of a difference
#   placeholder an uncited modelling value (documented in docs/LIMITATIONS_REGISTER.md); replace when evidence exists
# A table is never allowed to look more informative than its evidence: rows copied across lactations because the
# source is pooled are labelled, not hidden.
BASIS_CODES = ("source", "derived", "repeated", "placeholder")

FULL_TABLE_REGISTRY = {}    # name -> {"build": fn() -> DataFrame, "statuses": tuple|None, "n_months": int}


# ============================================================
# TIME-CONVENTION HELPERS (moved here from probabilities/base.py)
# ============================================================

def age_offset_slots():
    """Midpoints (days) of the sub-slots that discretize the uniform entry offset u on (0, UNIT_DAYS):
    quarter-day spacing at every resolution, so a non-integer threshold is represented even at DAY
    resolution and MONTH thirds/quarters are exact."""
    n = 4 * UNIT_DAYS
    return (np.arange(n) + 0.5) * (UNIT_DAYS / n)


def mean_step_probability(survival_fn, k_max, first_age_step=0):
    """p[k] for k = 0..k_max from a survival function S(age_days) (vectorized).
    first_age_step: step index whose start age is 0 (0 for DIM-type ages, 1 for gestation dp)."""
    slot = age_offset_slots()
    Sbar = np.array([np.mean(survival_fn((k - first_age_step) * UNIT_DAYS + slot)) for k in range(k_max + 2)])
    return 1 - Sbar[1:] / np.where(Sbar[:-1] > 0, Sbar[:-1], 1.0)


def threshold_crossing_mass(threshold_days):
    """{boundary index k: share of a birth/entry cohort whose TRUE age first reaches `threshold_days`
    at step boundary k}. Age at boundary k is k*UNIT_DAYS + u, u ~ Uniform(0, UNIT_DAYS) (see above), so
    k*(u) = ceil((T - u)/UNIT_DAYS). Index 0 means 'already past the threshold at the first boundary'
    (possible only when T < UNIT_DAYS). Shares sum to 1; slot midpoints discretize u."""
    slot = age_offset_slots()
    k_u = np.maximum(np.ceil((threshold_days - slot) / UNIT_DAYS).astype(int), 0)
    mass = {}
    for k in k_u:
        mass[int(k)] = mass.get(int(k), 0.0) + 1.0 / len(slot)
    return mass


def crossing_conditional(mass, k):
    """P(cross AT boundary k | not crossed before k) from a threshold_crossing_mass dict."""
    left = 1.0 - sum(m for j, m in mass.items() if j < k)
    return 1.0 if k >= max(mass) else (mass.get(k, 0.0) / left if left > 1e-12 else 1.0)


# ============================================================
# MONTHLY TABLE -> SURVIVAL -> PER-STEP PROBABILITY
# ============================================================

def survival_from_monthly(p_month, zero_until_day=-1, zero_after_day=None):
    """S(age_days) for a monthly-probability vector (index = native month of age).
    Constant daily hazard inside each month; hazard forced to 0 on days <= zero_until_day
    (e.g. the voluntary waiting period). Linear interpolation between whole days."""
    p = np.clip(np.asarray(p_month, dtype=float), 0.0, 1.0)
    day = np.arange(len(p) * MONTH)
    h = 1.0 - (1.0 - p[day // MONTH]) ** (1.0 / MONTH)
    h = np.where(day <= zero_until_day, 0.0, h)
    if zero_after_day is not None:
        h = np.where(day >= zero_after_day, 0.0, h)   # interval [d, d+1) lies fully past the cutoff age
    S = np.concatenate(([1.0], np.cumprod(1.0 - h)))
    grid = np.arange(len(S))
    return lambda age: np.interp(age, grid, S)


def step_probability_matrix(P, k_max, first_age_step=0, zero_until_day=-1, zero_after_day=None):
    """P: (n_lactations, n_months) monthly probabilities -> (n_lactations, k_max+1) per-step probabilities."""
    return np.stack([
        mean_step_probability(survival_from_monthly(P[i], zero_until_day, zero_after_day), k_max, first_age_step)
        for i in range(P.shape[0])
    ])


def step_mean_matrix(V, k_max):
    """V: (n_lactations, n_months) DAILY values (e.g. $/day). Returns (n_lactations, k_max+1): the mean daily
    value over the UNIT_DAYS real days of each step (multiply by UNIT_DAYS for a per-step amount)."""
    out = np.zeros((V.shape[0], k_max + 1))
    n_days = V.shape[1] * MONTH
    for k in range(k_max + 1):
        d = np.minimum(k * UNIT_DAYS + np.arange(UNIT_DAYS), n_days - 1)
        out[:, k] = V[:, d // MONTH].mean(axis=1)
    return out


# ============================================================
# FULL-TABLE FRAMES, COVERAGE, OVERRIDE, EXPORT
# ============================================================

def full_grid(n_months, statuses=None):
    """Tidy grid: one row per (lactation 1..MAX_LACTATION, month 0..n_months-1[, status])."""
    lac = np.arange(1, MAX_LACTATION + 1)
    mon = np.arange(n_months)
    if statuses is None:
        L, M = np.meshgrid(lac, mon, indexing="ij")
        return pd.DataFrame({"lactation": L.ravel(), "month": M.ravel()})
    S, L, M = np.meshgrid(np.arange(len(statuses)), lac, mon, indexing="ij")
    return pd.DataFrame({"status": np.array(statuses)[S.ravel()], "lactation": L.ravel(), "month": M.ravel()})


def validate_full_coverage(df, name, n_months, statuses=None, value_col="p"):
    if "basis" not in df.columns or not df["basis"].isin(BASIS_CODES).all():
        bad = sorted(set(df["basis"].dropna().astype(str)) - set(BASIS_CODES)) if "basis" in df.columns else ["(column missing)"]
        raise ValueError(
            f"Lookup table '{name}': every row needs a `basis` in {BASIS_CODES} (evidence status of the value); "
            f"got {bad}. Keep the `basis` column when you edit an exported table; set it to what is true for YOUR values."
        )
    need = full_grid(n_months, statuses)
    keys = ["lactation", "month"] + (["status"] if statuses else [])
    missing = need.merge(df[keys + [value_col]], on=keys, how="left")
    gaps = missing[missing[value_col].isna()]
    if len(gaps):
        raise ValueError(
            f"Lookup table '{name}' does not cover every lactation x month"
            f"{' x status' if statuses else ''}: {len(gaps)} missing row(s), e.g.\n{gaps.head()}\n"
            f"Fix cow/data/{name}.csv (export a default with cow.lookup_tables.export_full_tables)."
        )
    if (df[value_col] < 0).any() or (df[value_col] > 1.0 + 1e-12).any():
        if value_col == "p":
            raise ValueError(f"Lookup table '{name}': probabilities must be in [0, 1].")


def register_full_table(name, build, n_months, statuses=None, value_col="p"):
    FULL_TABLE_REGISTRY[name] = {"build": build, "n_months": n_months, "statuses": statuses, "value_col": value_col}


_TABLE_CACHE = {}      # name -> (key, DataFrame): one validated DataFrame per table, rebuilt only if the override file or the builder changes


def clear_table_cache():
    """Forget the cached DataFrames (only needed when a builder's INPUTS are changed in-process, e.g. by a sensitivity test)."""
    _TABLE_CACHE.clear()


def get_full_table(name):
    """The active full table: cow/data/<name>.csv if present (validated), else the registered default.
    Cached in memory (keyed by the override file's modification time and the builder), so repeated calls cost nothing and the
    Excel workbook (cow/lookup_workbook.py) is never read by the model: edit in Excel, `import`, and the CSV override is picked up here."""
    spec = FULL_TABLE_REGISTRY[name]
    value_col = spec["value_col"]
    path = os.path.join(DATA_DIR, f"{name}.csv")
    key = (os.path.getmtime(path) if os.path.exists(path) else None, id(spec["build"]))
    hit = _TABLE_CACHE.get(name)
    if hit is not None and hit[0] == key:
        return hit[1].copy()
    df = pd.read_csv(path) if key[0] is not None else spec["build"]()
    validate_full_coverage(df, name, spec["n_months"], spec["statuses"], value_col)
    _TABLE_CACHE[name] = (key, df)
    return df.copy()


def table_to_array(df, n_months, statuses=None, value_col="p"):
    """Tidy table -> ndarray (n_status, n_lactations, n_months) (n_status = 1 when statuses is None)."""
    sts = statuses or (None,)
    out = np.zeros((len(sts), MAX_LACTATION, n_months))
    for si, s in enumerate(sts):
        d = df if s is None else df[df["status"] == s]
        d = d[(d["lactation"] <= MAX_LACTATION) & (d["month"] < n_months)]
        out[si, d["lactation"].values - 1, d["month"].values] = d[value_col].values
    return out


def export_full_tables(out_dir):
    """Write every registered default full table as CSV (edit a copy in cow/data/ to override)."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, spec in FULL_TABLE_REGISTRY.items():
        path = os.path.join(out_dir, f"{name}.csv")
        spec["build"]().to_csv(path, index=False)
        written.append(path)
    return written


def status_index(dp, dry_off_dp):
    """0 = Open (dp == 0), 1 = Pregnant (0 < dp < dry_off_dp), 2 = Dry (dp >= dry_off_dp)."""
    dp = np.asarray(dp)
    return np.where(dp == 0, 0, np.where(dp < dry_off_dp, 1, 2))

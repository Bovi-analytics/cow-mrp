"""cowmrp.domains.cow.farm -- a whole herd: build one, load one from farm records, project it forward, value it.

    from cowmrp.model import build_model
    from cowmrp.domains.cow.farm import load_herd, project_herd, generate_herd
    model = build_model("cow")                                     # month resolution unless COW_TIME_UNIT says otherwise
    herd = load_herd("my_herd.csv", model)                         # raw records (lactation, dim_days, dp_days) or mapped (state_id)
    result = project_herd(herd, model, mode="expected", steps=24)  # also "stochastic" and "monte_carlo"
    from cowmrp.viz.farm import write_farm_outputs                 # CSV tables + the two dashboards (static and Plotly)
    write_farm_outputs(result, "farm_results")

HERD DATA (``load_herd``). One row per cow. The cow is described in the model's own terms, in REAL DAYS; never health or DNB flags (those
are not state variables). Accepted columns:
    cow_id                optional (1..n when absent); must be unique
    lactation             1 .. MAX_LACTATION                                                (raw records)
    dim_days              days in milk, >= 0                                                (raw records)
    dp_days               days pregnant, 0 = open (default 0 when the column is absent)     (raw records)
    uim, up               the model's step indices, as written by ``generate_herd``         (alternative to the days columns)
    state_id              the model state of the cow (already mapped)                       (mapped records)
A cow is mapped to a state exactly as ``runners/cow/farm/generate.py`` always did (``state_ids_for``: ``dim_step_of_days`` and
``dp_step_of_days`` of ``cowmrp.domains.cow.parameters``, then the state with that (lactation, dim, dp)); this module adds no biological
rule. When ``state_id`` AND raw columns are given they must agree. Every problem found is reported together (``HerdDataError``) with the
row and cow_id: missing columns, non-numeric values, lactation outside 1..MAX_LACTATION, negative days, duplicate cow_id, a (lactation, dim, dp)
the model has no state for (with the closest valid states), or a state_id outside this resolution's table.
The returned table has the schema of the generated herd files: cow_id, state_id, lactation, uim, up, dim_days, dp_days, status (+ the input
days as input_dim_days / input_dp_days when raw days were given).

PROJECTION (``project_herd``, called by ``runners/cow/farm/simulate.py``): the herd is projected with the
model's own transition matrix, ``expected`` (exact distribution evolution, split into cows never replaced vs replaced at least once),
``stochastic`` (one sampled path per cow) or ``monte_carlo`` (mean / std / 5-95 %), summarised by groupings (Open / Pregnant / Dry,
lactation, DIM stage, month in milk), and linked to the model's VALUES (read-only diagnostics, never added to a reward).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cowmrp.mrp import core as mrp_core
from cowmrp.mrp.simulator import run_stochastic, run_monte_carlo, compare_methods, run_expected_origin_tracked
from cowmrp.domains.cow.parameters import (TIME_UNIT, UNIT_DAYS, MAX_LACTATION, MAX_DIM_DAYS, MAX_DP_DAYS, LAST_BREEDING_DIM_DAYS,
                                           dim_step_of_days, dp_step_of_days)
from cowmrp.domains.cow.states import repro_status
from cowmrp.domains.cow.transitions import get_replacement_state_id


class HerdDataError(ValueError):
    """The herd records are missing, invalid or inconsistent; the message lists every problem found (row, cow_id, what is wrong)."""


# ============================================================
# HERD CONFIGURATION DEFAULTS (the runner's generate.py keeps editable copies of these lists)
# ============================================================
# status: "open" | "pregnant" | "dry" | "dnb" (open AND past the last breeding day -- a region, not a state)
# ranges are inclusive, in REAL DAYS (dim_days, dp_days); lactation is [min, max].
DEFAULT_HERD_MIX = [
    {"proportion": 0.30, "status": "open",     "lactation": [1, 1],             "dim_days": [0, 150]},
    {"proportion": 0.12, "status": "open",     "lactation": [2, MAX_LACTATION], "dim_days": [0, 150]},
    {"proportion": 0.06, "status": "dnb",      "lactation": [1, MAX_LACTATION], "dim_days": [LAST_BREEDING_DIM_DAYS + 1, 400]},
    {"proportion": 0.36, "status": "pregnant", "lactation": [1, MAX_LACTATION], "dp_days": [30, 215]},
    {"proportion": 0.16, "status": "dry",      "lactation": [2, MAX_LACTATION], "dp_days": [222, 277]},
]
DEFAULT_EXACT_STATES = [      # (lactation, dim_days, dp_days, count)
    (1, 60, 0, 200), (2, 120, 0, 150), (2, 200, 100, 300), (3, 280, 230, 100),
]


def current_config(num_states=None):
    """The cow configuration a herd file belongs to (state ids are only comparable under the same one)."""
    cfg = {"time_unit": TIME_UNIT, "unit_days": UNIT_DAYS, "max_lactation": MAX_LACTATION, "max_dim_days": MAX_DIM_DAYS, "max_dp_days": MAX_DP_DAYS}
    return cfg if num_states is None else {**cfg, "num_states": int(num_states)}


# ============================================================
# STATE MAPPING AND HERD GENERATION (used by runners/cow/farm/generate.py)
# ============================================================
def status_of(states):
    """Reproductive status label per state: open / pregnant / dry, plus 'dnb' for open cows past the last breeding day."""
    s = np.char.lower(repro_status(states["dp"].values).astype(str))
    age = (states["dim"].values + 1) * UNIT_DAYS                       # mean age during the step (mid-step convention)
    return np.where((s == "open") & (age > LAST_BREEDING_DIM_DAYS), "dnb", s)


def steady_state(states, matrix, rid):
    """Stationary distribution of the closed herd: expected visits per replacement-to-replacement cycle, normalised."""
    n = len(states)
    Q = matrix.tocsr().copy().tolil()
    Q[:, rid] = 0
    e = np.zeros(n)
    e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)
    return np.clip(x, 0, None) / x.sum()


def sample_mix(states, mix, n_cows, rng):
    status = status_of(states)
    total = sum(m["proportion"] for m in mix)
    counts = [int(round(m["proportion"] / total * n_cows)) for m in mix]
    counts[0] += n_cows - sum(counts)
    ids = []
    for m, k in zip(mix, counts):
        mask = status == m["status"]
        lo, hi = m.get("lactation", [1, MAX_LACTATION])
        mask &= (states["lactation"].values >= lo) & (states["lactation"].values <= hi)
        if "dim_days" in m:
            dlo, dhi = m["dim_days"]
            mid = (states["dim"].values + 1) * UNIT_DAYS
            mask &= (mid >= dlo) & (mid <= dhi)
        if "dp_days" in m:
            plo, phi = m["dp_days"]
            mid = states["dp"].values * UNIT_DAYS
            mask &= (mid >= plo) & (mid <= phi)
        pool = states["state_id"].values[mask]
        if len(pool) == 0:
            raise ValueError(f"No model state satisfies the herd category {m} at this time resolution "
                             f"(the state space has only reachable states -- loosen the ranges).")
        ids.extend(rng.choice(pool, size=k, replace=True))
    return np.array(ids)


def _steps_of_days(days, step_of_days):
    """dim / dp step of every value of `days` (the scalar conversion of cowmrp.domains.cow.parameters, computed once per distinct value)."""
    days = np.asarray(days)
    uniq, inv = np.unique(days, return_inverse=True)
    return np.array([step_of_days(float(d)) for d in uniq], dtype=int)[inv]


def state_ids_for(states, lactation, dim_days, dp_days):
    """THE mapping of a cow described in real days to a model state (the one ``exact_ids`` of generate.py used, now vectorised):
    (lactation, dim_step_of_days(dim_days), dp_step_of_days(dp_days)) looked up in the state table. Returns (state_id array, dim steps,
    dp steps); state_id is -1 where the model has no such state."""
    lactation = np.asarray(lactation)
    dim = _steps_of_days(dim_days, dim_step_of_days)
    dp = _steps_of_days(dp_days, dp_step_of_days)
    index = pd.MultiIndex.from_arrays([states["lactation"].values, states["dim"].values, states["dp"].values])
    pos = index.get_indexer(pd.MultiIndex.from_arrays([lactation, dim, dp]))
    ids = np.where(pos >= 0, states["state_id"].values[np.clip(pos, 0, None)], -1)
    return ids, dim, dp


def exact_ids(states, entries):
    ids = []
    for lac, dim_days, dp_days, count in entries:
        q = (states["lactation"] == lac) & (states["dim"] == dim_step_of_days(dim_days)) & (states["dp"] == dp_step_of_days(dp_days))
        if not q.any():
            raise ValueError(f"EXACT_STATES entry (lactation={lac}, dim_days={dim_days}, dp_days={dp_days}) is not a "
                             f"reachable model state (dim step {dim_step_of_days(dim_days)}, dp step {dp_step_of_days(dp_days)}).")
        ids.extend([int(states.loc[q, "state_id"].iloc[0])] * count)
    return np.array(ids)


def herd_table(states, ids):
    """The herd file schema for model state ids `ids` (one cow each): cow_id, state_id, lactation, uim, up, dim_days, dp_days, status."""
    rows = states.set_index("state_id").loc[ids].reset_index()
    return pd.DataFrame({
        "cow_id": np.arange(1, len(ids) + 1), "state_id": ids,
        "lactation": rows["lactation"].values,
        "uim": rows["dim"].values, "up": rows["dp"].values,                      # resolution-neutral step columns
        "dim_days": (rows["dim"].values + 1) * UNIT_DAYS, "dp_days": rows["dp"].values * UNIT_DAYS,
        "status": status_of(rows),
    })


def generate_herd(model, mode="equilibrium", n_cows=10000, seed=42, mix=None, exact=None):
    """A synthetic herd. ``equilibrium``: cows drawn from the model's own steady-state herd; ``mix``: the ``mix`` list (default
    DEFAULT_HERD_MIX); ``exact``: the ``exact`` list (default DEFAULT_EXACT_STATES). Same draws as runners/cow/farm/generate.py for the same seed."""
    states = model.states
    rng = np.random.default_rng(seed)
    if mode == "equilibrium":
        pi = steady_state(states, model.P, get_replacement_state_id(states))
        ids = rng.choice(len(states), size=n_cows, p=pi)
    elif mode == "mix":
        ids = sample_mix(states, DEFAULT_HERD_MIX if mix is None else mix, n_cows, rng)
    elif mode == "exact":
        ids = exact_ids(states, DEFAULT_EXACT_STATES if exact is None else exact)
    else:
        raise ValueError(f"Unknown mode {mode!r} -- use equilibrium | mix | exact")
    return herd_table(states, ids)


# ============================================================
# LOADING FARM RECORDS
# ============================================================
_RAW = ("lactation", "dim_days")
MAX_REPORTED = 15


def _numeric(df, col, problems, integer=False):
    """float array of a column; non-numeric / missing entries are reported (row labels) and become NaN."""
    v = pd.to_numeric(df[col], errors="coerce")
    bad = df.index[v.isna()]
    if len(bad):
        problems.append((f"column '{col}': missing or non-numeric value", bad))
    vals = v.to_numpy(float)
    if integer:
        nonint = df.index[~v.isna() & (np.abs(vals - np.round(vals)) > 1e-9)]
        if len(nonint):
            problems.append((f"column '{col}': must be a whole number", nonint))
    return vals


def _report(problems, df, source):
    lines = []
    for msg, rows in problems:
        rows = list(rows)
        label = [f"row {df.index.get_loc(r) + 2}" + (f" (cow_id {df.loc[r, 'cow_id']})" if "cow_id" in df and r in df.index else "") for r in rows[:MAX_REPORTED]]
        more = f" ... and {len(rows) - MAX_REPORTED} more" if len(rows) > MAX_REPORTED else ""
        lines.append(f"  - {msg}: {'; '.join(label)}{more}")
    raise HerdDataError(f"{len(problems)} problem(s) in the herd records ({source}):\n" + "\n".join(lines))


def load_herd(data, model=None, check_config=True):
    """Herd records -> the herd table (cow_id, state_id, lactation, uim, up, dim_days, dp_days, status).

    ``data``: a DataFrame or the path of a CSV file. ``model``: the built cow model (built here when None). Raw records (lactation, dim_days,
    dp_days) and already mapped records (state_id; also the files written by ``generate_herd``) are both accepted, see the module docstring.
    For a CSV with a ``<name>.meta.json`` sidecar (written by generate.py) the herd's configuration must equal the current one (state ids are not
    comparable across time resolutions); ``check_config=False`` skips that. Raises ``HerdDataError`` listing every problem.
    """
    from cowmrp.model import build_model
    source = "DataFrame"
    if isinstance(data, (str, os.PathLike)):
        source = str(data)
        if not os.path.exists(source):
            raise HerdDataError(f"herd file not found: {source}")
        df = pd.read_csv(source)
        meta_path = os.path.splitext(source)[0] + ".meta.json"
    else:
        df, meta_path = data.copy(), None
    model = model or build_model("cow")
    states = model.states
    if meta_path and check_config and os.path.exists(meta_path):
        fm, live = json.load(open(meta_path)), current_config(len(states))
        bad = {k: (fm.get(k), v) for k, v in live.items() if fm.get(k) != v}
        if bad:
            raise HerdDataError(f"{source} was generated under a different configuration:\n" +
                                "\n".join(f"  {k}: herd has {a!r}, current is {b!r}" for k, (a, b) in bad.items()) +
                                "\nRegenerate it under the same time resolution (state ids are not comparable across configurations).")
    if len(df) == 0:
        raise HerdDataError(f"no cows in the herd records ({source})")
    df = df.reset_index(drop=True)
    has_state, has_days = "state_id" in df, all(c in df for c in _RAW)
    has_steps = all(c in df for c in ("lactation", "uim", "up"))
    if not (has_state or has_days or has_steps):
        raise HerdDataError(f"the herd records ({source}) need either a 'state_id' column, or 'lactation' and 'dim_days' (and 'dp_days' for "
                            f"pregnant cows), or 'lactation', 'uim' and 'up'. Found columns: {list(df.columns)}")
    problems = []
    if "cow_id" in df:
        dup = df.index[df["cow_id"].duplicated(keep=False)]
        if len(dup):
            problems.append(("duplicate cow_id", dup))
    else:
        df["cow_id"] = np.arange(1, len(df) + 1)

    ids_from_state = None
    if has_state:
        sid = _numeric(df, "state_id", problems, integer=True)
        ok = ~np.isnan(sid)
        out_of_range = df.index[ok & ((sid < 0) | (sid >= len(states)))]
        if len(out_of_range):
            problems.append((f"state_id outside 0..{len(states) - 1} (herd built for another time resolution or state table?)", out_of_range))
        ids_from_state = np.where(ok & (sid >= 0) & (sid < len(states)), sid, -1).astype(int)

    ids_from_raw = dim_s = dp_s = None
    if has_days or has_steps:
        lac = _numeric(df, "lactation", problems, integer=True)
        bad_lac = df.index[~np.isnan(lac) & ((lac < 1) | (lac > MAX_LACTATION))]
        if len(bad_lac):
            problems.append((f"lactation must be 1..{MAX_LACTATION}", bad_lac))
        if has_days:
            dim_days = _numeric(df, "dim_days", problems)
            dp_days = _numeric(df, "dp_days", problems) if "dp_days" in df else np.zeros(len(df))
            for name, arr in (("dim_days", dim_days), ("dp_days", dp_days)):
                neg = df.index[~np.isnan(arr) & (arr < 0)]
                if len(neg):
                    problems.append((f"column '{name}' must be >= 0", neg))
            valid = ~(np.isnan(lac) | np.isnan(dim_days) | np.isnan(dp_days) | (lac < 1) | (lac > MAX_LACTATION) | (dim_days < 0) | (dp_days < 0))
            ids_from_raw = np.full(len(df), -1, dtype=int)
            dim_s, dp_s = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
            if valid.any():
                i, d, p = state_ids_for(states, lac[valid].astype(int), dim_days[valid], dp_days[valid])
                ids_from_raw[valid], dim_s[valid], dp_s[valid] = i, d, p
            none = df.index[valid & (ids_from_raw < 0)]
            if len(none):
                problems.append((f"no model state at {TIME_UNIT} resolution for this (lactation, dim_days, dp_days) "
                                 f"(a pregnant cow must have been bred between day 50 and day 300 in milk; use state_ids_for / the closest valid states: "
                                 f"{_closest(states, df, none, lac, dim_s, dp_s)})", none))
        else:
            uim, up = _numeric(df, "uim", problems, integer=True), _numeric(df, "up", problems, integer=True)
            valid = ~(np.isnan(lac) | np.isnan(uim) | np.isnan(up) | (lac < 1) | (lac > MAX_LACTATION))
            index = pd.MultiIndex.from_arrays([states["lactation"].values, states["dim"].values, states["dp"].values])
            ids_from_raw = np.full(len(df), -1, dtype=int)
            if valid.any():
                pos = index.get_indexer(pd.MultiIndex.from_arrays([lac[valid].astype(int), uim[valid].astype(int), up[valid].astype(int)]))
                ids_from_raw[valid] = np.where(pos >= 0, states["state_id"].values[np.clip(pos, 0, None)], -1)
            none = df.index[valid & (ids_from_raw < 0)]
            if len(none):
                problems.append(("no model state with this (lactation, uim, up)", none))
    if problems:
        _report(problems, df, source)

    if ids_from_state is not None and ids_from_raw is not None:
        clash = df.index[ids_from_state != ids_from_raw]
        if len(clash):
            _report([("state_id disagrees with the lactation / days (or step) columns of the same row", clash)], df, source)
    ids = ids_from_state if ids_from_state is not None else ids_from_raw
    out = herd_table(states, ids)
    out["cow_id"] = df["cow_id"].to_numpy()
    if has_days and not has_state:                                  # keep the farmer's own days beside the model's (bin-centre) days
        out["input_dim_days"], out["input_dp_days"] = df["dim_days"].to_numpy(), (df["dp_days"].to_numpy() if "dp_days" in df else 0)
    return out


def _closest(states, df, rows, lac, dim_s, dp_s, k=3):
    """A few valid states (in calendar days) nearest to the first problem row, for the error message."""
    r = rows[0]
    near = states[(states["lactation"] == int(lac[r])) & ((states["dp"] > 0) == (dp_s[r] > 0))].copy()
    near["gap"] = (near["dim"] - dim_s[r]).abs() + (near["dp"] - dp_s[r]).abs()
    return "; ".join(f"(lactation={int(s.lactation)}, dim_days={int((s.dim + 1) * UNIT_DAYS)}, dp_days={int(s.dp * UNIT_DAYS)})"
                     for s in near.sort_values("gap").head(k).itertuples())


# ============================================================
# COW GROUPINGS -- vectorised: each returns one label per STATE (used by runners/cow/farm/simulate.py)
# ============================================================
def _age_dim(states):                         # mean days in milk during the step (mid-step convention)
    return (states["dim"].values + 1) * UNIT_DAYS


def g_status(states):
    dp = states["dp"].values
    from cowmrp.domains.cow.parameters import DRY_OFF_DP
    return np.where(dp >= DRY_OFF_DP, "Dry", np.where(dp > 0, "Pregnant", "Open"))   # no DNB state in this model


def g_lactation(states):
    lac = states["lactation"].values
    return np.where(lac == 1, "Lactation 1", np.where(lac == 2, "Lactation 2", "Lactation 3+"))


def g_lactation_exact(states):
    return np.array([f"Lac {int(l):02d}" for l in states["lactation"].values])


def g_dim_stage(states):
    a = _age_dim(states)
    return np.select([a <= 60, a <= 150, a <= 300], ["DIM 0-60 d", "DIM 61-150 d", "DIM 151-300 d"], default="DIM 300+ d")


def g_month_in_milk(states):
    """Month in milk as a state label (MIM 01 ... MIM 24, then 25+): month index = days in milk // 30 + 1, so MONTH steps give MIM = dim + 1."""
    m = (states["dim"].values * UNIT_DAYS) // 30 + 1
    return np.array([f"MIM {int(k):02d}" if k <= 24 else "MIM 25+" for k in m])


GROUPERS = {"status": g_status, "lactation": g_lactation, "lactation_exact": g_lactation_exact, "dim_stage": g_dim_stage,
            "month_in_milk": g_month_in_milk}
DEFAULT_GROUPERS = ["status", "lactation", "dim_stage", "month_in_milk"]


_CODE_CACHE = {}          # (grouper, id(states)) -> (states, fingerprint, codes, index); keeps `states` alive so the id cannot be reused


def _group_codes(states, grouper):
    """Integer code of every state's label plus the sorted label index, computed once per (grouper, state table).

    The index is produced by the original expression (groupby on the string labels, sorted), and the codes are positions in it,
    so the labels, their order and the index type are exactly those of the string-label groupby."""
    fp = (states["lactation"].values.copy(), states["dim"].values.copy(), states["dp"].values.copy())
    hit = _CODE_CACHE.get((grouper, id(states)))
    if hit is not None and hit[0] is states and all(np.array_equal(a, b) for a, b in zip(hit[1], fp)):
        return hit[2], hit[3]
    labels = GROUPERS[grouper](states)
    index = pd.Series(np.zeros(len(labels))).groupby(labels).sum().sort_index().index
    codes = index.get_indexer(labels)
    if len(_CODE_CACHE) > 64:
        _CODE_CACHE.clear()
    _CODE_CACHE[(grouper, id(states))] = (states, fp, codes, index)
    return codes, index


def group_probs(states, pi, grouper):
    """Probability mass per group label (sorted). Same values as pd.Series(pi).groupby(<string labels>).sum().sort_index(): the labels are
    turned into integer codes once per state table and the same pandas groupby-sum is applied to the codes (bit-identical, much faster)."""
    codes, index = _group_codes(states, grouper)
    out = pd.Series(pi).groupby(codes).sum()
    out.index = index
    return out


def group_counts(states, ids, grouper):
    return pd.Series(GROUPERS[grouper](states)[np.asarray(ids)]).value_counts().sort_index()


def mc_summary(trajs, labels, steps):
    """{step: DataFrame[mean, std, p05, p95] indexed by label} of the per-simulation group counts.

    Vectorised: the label of every state is coded once and the counts of all simulations at a step come from one bincount (the previous
    version looped over simulations and labels and compared the whole herd to each label); counts and statistics are identical."""
    labels = np.asarray(labels)
    cats = sorted(set(labels))
    code = np.searchsorted(np.array(cats), labels)                      # label code of every state
    k, n_sims = len(cats), len(trajs)
    offset = (np.arange(n_sims) * k)[:, None]
    out = {}
    for s in range(steps + 1):
        ids = np.stack([t[s] for t in trajs])                          # (sims, cows) state ids at this step
        counts = np.bincount((code[ids] + offset).ravel(), minlength=n_sims * k).reshape(n_sims, k).astype(float)
        out[s] = pd.DataFrame({"mean": counts.mean(0), "std": counts.std(0), "p05": np.percentile(counts, 5, axis=0),
                               "p95": np.percentile(counts, 95, axis=0)}, index=cats)
    return out


def averaged_group_counts(states, id_arrays, grouper):
    return pd.concat([group_counts(states, ids, grouper) for ids in id_arrays], axis=1).fillna(0.0).mean(axis=1)



# ============================================================
# VALUES -- linked to the MRP's own solve (read-only diagnostics)
# ============================================================
def herd_values(model, herd, annual_discount=0.95):
    """Per-cow values from the model's infinite-horizon solve.
    V                : value of the state reward vector (replacement cost is NOT a state reward)
    V_keep           : V minus the present value of the replacement costs of every future exit (cow, then her replacements)
    replacement_value: V_keep(cow) - [V_keep(heifer) - cost of one replacement]  (cow vs a heifer bought now, same horizon; the same
                       comparison as value_comparisons.replacement_comparison and keepcull.keep_cull_table, infinite horizon)
    pregnancy_value  : V(pregnant) - V(open at the same DIM), pregnant cows only."""
    from cowmrp.domains.cow.keepcull import replacement_values
    states = model.states
    rv = replacement_values(model, None, annual_discount)
    net = rv["net"]
    V = net["V_total"]
    sid = herd["state_id"].values
    key = pd.MultiIndex.from_arrays([states["lactation"], states["dim"], states["dp"]])
    pos = pd.Series(states["state_id"].values, index=key)
    open_ids = pos.reindex(pd.MultiIndex.from_arrays([states["lactation"].values[sid], states["dim"].values[sid],
                                                      np.zeros(len(sid), dtype=int)])).values
    cols = [c for c in ("cow_id", "state_id", "lactation", "dim_days", "dp_days") if c in herd]       # herd descriptors are copied when present
    out = herd[cols].copy()
    out["V"] = V[sid]
    out["V_keep"] = net["V_keep"][sid]
    out["replacement_value"] = net["V_keep"][sid] - rv["replace"]
    pregnant = states["dp"].values[sid] > 0
    with np.errstate(invalid="ignore"):
        pv = np.where(pregnant & ~np.isnan(open_ids), V[sid] - V[np.nan_to_num(open_ids).astype(int)], np.nan)
    out["pregnancy_value"] = pv
    out["status"] = g_status(states)[sid]
    return out


# ============================================================
# PROJECTION (called by runners/cow/farm/simulate.py)
# ============================================================
def project_herd(herd, model=None, mode="expected", steps=24, seed=42, sims=100, groupers=None, values=True, compare=False,
                 annual_discount=0.95):
    """Project ``herd`` (a table with state_id, cow_id, lactation; from ``load_herd``) ``steps`` model steps ahead.

    Returns a dict: ``current`` / ``predicted`` ({grouper: Series of cows per label}), ``evolution`` ({grouper: per-step Series}),
    ``regular`` / ``replaced`` (per-step status counts of the original herd and of cows that entered through the replacement state),
    ``distribution_summary`` (DataFrame), ``values`` (per-cow DataFrame or None), ``monte_carlo`` (DataFrame, monte_carlo mode),
    ``trajectory`` (DataFrame, stochastic mode), ``method_comparison`` (DataFrame, expected mode with compare), ``meta``, ``mode``, ``steps``.
    """
    from cowmrp.model import build_model
    model = model or build_model("cow")
    states, matrix = model.states, model.P
    groupers = list(groupers) if groupers is not None else list(DEFAULT_GROUPERS)
    for g in groupers:
        if g not in GROUPERS:
            raise ValueError(f"unknown grouper {g!r}; choose from {sorted(GROUPERS)}")
    if mode not in ("expected", "stochastic", "monte_carlo"):
        raise ValueError(f"mode must be expected | stochastic | monte_carlo, not {mode!r}")
    rid = get_replacement_state_id(states)
    n_cows, ids0 = len(herd), herd["state_id"].values
    if n_cows == 0 or ids0.min() < 0 or ids0.max() >= len(states):
        raise HerdDataError("the herd is empty or has state_id values outside this model's state table (use load_herd)")

    pi0 = np.bincount(ids0, minlength=len(states)).astype(float)
    pi0 /= pi0.sum()
    current = {g: group_counts(states, ids0, g) for g in groupers}
    evolution = {g: [] for g in groupers}
    regular = replaced = None
    mc_rows, cmp_df, trajectory = [], None, None

    if mode == "expected":
        if compare:
            cmp_df = compare_methods(matrix, pi0, steps)
        po, pt = pi0.copy(), np.zeros_like(pi0)
        orig, tag, dists = [po.copy()], [pt.copy()], [pi0.copy()]
        for _ in range(steps):
            po, pt = run_expected_origin_tracked(matrix, po, pt, 1, rid)
            orig.append(po.copy()); tag.append(pt.copy()); dists.append(po + pt)
        for step, (d, o, t) in enumerate(zip(dists, orig, tag)):
            assert np.allclose(o + t, d, atol=1e-9), f"regular + replaced does not reconstruct the herd at step {step}"
        for g in groupers:
            evolution[g] = [group_probs(states, d, g) * n_cows for d in dists]
        regular = [group_probs(states, p, "status") * n_cows for p in orig]
        replaced = [group_probs(states, p, "status") * n_cows for p in tag]
        predicted = {g: evolution[g][-1] for g in groupers}
        for step in range(steps + 1):
            tot = regular[step].add(replaced[step], fill_value=0).sum()
            assert abs(tot - n_cows) < 1e-6, f"cow count not conserved at step {step}: {tot}"
    elif mode == "stochastic":
        hist = run_stochastic(matrix, ids0, steps, seed=seed)
        for g in groupers:
            evolution[g] = [group_counts(states, hist[s], g) for s in range(steps + 1)]
        mask = mrp_core.trajectory_tagged_mask(hist, rid)
        regular = [group_counts(states, hist[s][~mask[s]], "status") for s in range(steps + 1)]
        replaced = [group_counts(states, hist[s][mask[s]], "status") for s in range(steps + 1)]
        predicted = {g: evolution[g][-1] for g in groupers}
        trajectory = pd.DataFrame(hist.T, columns=[f"step_{s}" for s in range(steps + 1)]).assign(cow_id=herd["cow_id"].values)
    else:
        trajs = run_monte_carlo(matrix, ids0, steps, n_sims=sims, seed=seed)
        predicted = {}
        for g in groupers:
            summ = mc_summary(trajs, GROUPERS[g](states), steps)
            evolution[g] = [summ[s]["mean"] for s in range(steps + 1)]
            predicted[g] = summ[steps]["mean"]
            for s, df in summ.items():
                for label, r in df.iterrows():
                    mc_rows.append({"step": s, "grouper": g, "label": label, "mean": r["mean"], "std": r["std"], "p05": r["p05"], "p95": r["p95"]})
        masks = [mrp_core.trajectory_tagged_mask(t, rid) for t in trajs]
        regular = [averaged_group_counts(states, [t[s][~m[s]] for t, m in zip(trajs, masks)], "status") for s in range(steps + 1)]
        replaced = [averaged_group_counts(states, [t[s][m[s]] for t, m in zip(trajs, masks)], "status") for s in range(steps + 1)]

    rows = []
    for g in groupers:
        for label in sorted(set(current[g].index) | set(predicted[g].index)):
            c, p = float(current[g].get(label, 0)), float(predicted[g].get(label, 0))
            rows.append({"grouper": g, "label": label, "current": c, "predicted": p, "change": p - c})
    values_df = herd_values(model, herd, annual_discount) if values else None
    return {"mode": mode, "steps": steps, "groupers": groupers, "current": current, "predicted": predicted, "evolution": evolution,
            "regular": regular, "replaced": replaced, "distribution_summary": pd.DataFrame(rows), "values": values_df,
            "monte_carlo": pd.DataFrame(mc_rows) if mc_rows else None, "trajectory": trajectory, "method_comparison": cmp_df,
            "meta": {"n_cows": n_cows, "num_states": len(states), "config": current_config()}}

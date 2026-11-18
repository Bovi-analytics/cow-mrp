"""
cow/rewards/lactationcurve_milk_income.py
==========================================
A NEW, ISOLATED, selectable milk-income reward that prices milk production
predicted by the actual Bovi-Analytics `lactationcurve` package
(https://pypi.org/project/lactationcurve/,
https://github.com/Bovi-analytics/lactation_curve_core) instead of the
existing Cow model's `milk_yield_kg`.

    Cow state (dim, lactation)
        -> lactationcurve.fitting.milkbot_model(t, a, b, c, d)   [ACTUAL package call]
        -> predicted milk quantity (kg/day)
        -> cow.rewards.base.MILK_PRICE_PER_KG (EXISTING, unmodified)
        -> reward_lactationcurve_milk_income

THIS IS DELIBERATELY SEPARATE from everything already in place:
  - NOT wired into cow/rewards/__init__.py's BASE_REWARD_SPECS/compute_rewards()
    -- the existing reward aggregation, 'total', discounting, and every other
    reward component are UNTOUCHED and produce byte-identical output whether
    or not this module even exists.
  - Does NOT call, modify, or duplicate cow/states.py::milk_yield_kg,
    milk_yield_spec, pregnancy_adjusted_expected, or milk_state.
  - Does NOT touch cow/probabilities/, cow/transitions.py, mrp_core.py,
    mrp_engine.py, or another domain.
  - Reuses ONLY cow.rewards.base.MILK_PRICE_PER_KG and cow.parameters.UNIT_DAYS
    (both existing, read-only, unmodified) so the SAME price/economics apply
    to this reward as to the existing one -- nothing about pricing is
    reinvented.
  - Reproduces its own tiny parity-bucketing helper locally (not imported
    from cowmrp.domains.cow.states) so this module has zero import-time coupling to the
    existing milk pathway.

`selectable`: this module exposes `reward_lactationcurve_milk_income_spec`,
shaped exactly like every other Cow reward spec (name/sign/depends_on/
method/methods) and runnable
through the SAME generic `mrp_engine.run_specs` machinery every other reward
already uses -- e.g. `run_specs([reward_lactationcurve_milk_income_spec],
states, context=probs, has_sign=True)`. A caller opts in by explicitly
including this spec in such a call; nothing here forces it into the default
reward computation.

WHY EVERY `lactationcurve` IMPORT BELOW IS LAZY (inside function bodies, not
at module top level): the package requires Python >=3.12,<3.13 (per its own
PyPI classifiers) and is installed in THIS project's Python 3.12 interpreter
only (`py -3.12`), NOT in the project's default Python 3.13 interpreter that
`cow/`, `mrp_engine.py`, etc. normally run under. A top-level `import
lactationcurve` would make merely IMPORTING this file crash under the
project's normal interpreter, breaking "current/default reward behavior must
remain exactly unchanged." Lazy imports mean: this file always imports
cleanly everywhere; the package is only actually required at the moment
someone calls the fitting utility or the runtime evaluation function -- and
if it's missing there, that surfaces as Python's own clear ImportError, not a
silently broken module.

CALIBRATION STATE: no real DIM + observed milk-weight dataset exists in this
project (searched repeatedly, confirmed absent). This module therefore ships
with NO fitted parameters, hard-coded or otherwise -- see
DEFAULT_PARAMS_PATH / load_lactationcurve_milkbot_params() below.
Selecting/calling this reward before that file exists (or is complete) raises
a clear error naming exactly what's missing; it never invents a value or
falls back to the existing Cow milk model.

TO CALIBRATE LATER (once a real CSV exists): run, under the Python 3.12
interpreter that has `lactationcurve` installed --
    py -3.12 cow/rewards/lactationcurve_milk_income.py real_data.csv \\
        --dim-column dim --milk-column milk_yield_kg --parity-column lactation
This performs a REAL fit via lactationcurve.fitting.get_lc_parameters and
writes DEFAULT_PARAMS_PATH. It is a one-time, OFFLINE step -- it must never
be run from inside MRP/reward evaluation, and this module never calls it
itself.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

from cowmrp.domains.cow.parameters import UNIT_DAYS
from cowmrp.domains.cow.rewards.base import MILK_PRICE_PER_KG

DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lactationcurve_milkbot_params.json")


def _parity_bucket(lactation):
    """1st, 2nd, or 3rd-and-later lactation -- the same 3-bucket convention
    cow/states.py::milk_lactation_bucket uses, reproduced locally (not
    imported) so this module has zero dependency on the existing milk
    pathway."""
    lactation = np.asarray(lactation)
    return np.select([lactation == 1, lactation == 2], [1, 2], default=3)


# ============================================================
# OFFLINE CALIBRATION (never called from reward/state evaluation)
# ============================================================

def fit_and_save_lactationcurve_milkbot(
    csv_path,
    dim_column="dim",
    milk_column="milk_yield_kg",
    parity_column="lactation",
    output_path=None,
    fitting="frequentist",
    data_source_note=None,
):
    """
    OFFLINE ONLY. Reads a real CSV of observed dairy-cow test-day records and
    fits MilkBot parameters PER PARITY BUCKET (1, 2, 3+ -- matching the
    existing Cow convention) using the ACTUAL
    `lactationcurve.fitting.get_lc_parameters` function -- a genuine fit
    against real data, never invented. Saves the result to `output_path`
    (default DEFAULT_PARAMS_PATH) so the MRP runtime loads a frozen result
    instead of ever re-fitting.

    csv_path: path to a CSV with at least `dim_column`, `milk_column`, and
        `parity_column`.
    dim_column: column of days-in-milk (real calendar days) per test-day record.
    milk_column: column of observed milk yield (kg/day) per test-day record.
    parity_column: column of lactation/parity number -- REQUIRED (not
        optional) so each Cow parity bucket gets its own fitted curve; a
        single curve fit across all parities pooled together would not match
        how every other milk model in this project (existing MilkBot/Wood,
        the lookup table) is structured, and would silently assume
        parity-independence this project does not otherwise assume anywhere.
    output_path: where to write the frozen JSON parameter file (default:
        lactationcurve_milkbot_params.json, next to this module).
    fitting: "frequentist" (default) -- recorded in the output's metadata for
        provenance; Bayesian fitting is not implemented.
    data_source_note: optional human-readable description of where csv_path
        came from, stored in the output's metadata; defaults to the csv's
        absolute path.

    Returns the same dict that gets written to `output_path`.
    """
    from lactationcurve.fitting import get_lc_parameters  # lazy -- see module docstring
    import lactationcurve as _lc

    output_path = output_path or DEFAULT_PARAMS_PATH
    df = pd.read_csv(csv_path)

    required = {dim_column, milk_column, parity_column}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{csv_path!r} is missing required column(s) {sorted(missing_cols)} -- "
            f"a real fit needs '{dim_column}' (days in milk, real days), "
            f"'{milk_column}' (observed milk yield, kg/day), and "
            f"'{parity_column}' (lactation/parity number)."
        )

    lac_bucket = _parity_bucket(df[parity_column].values)
    params_by_parity = {}
    n_records_by_parity = {}
    for bucket in (1, 2, 3):
        mask = lac_bucket == bucket
        n = int(mask.sum())
        n_records_by_parity[str(bucket)] = n
        if n < 2:
            raise ValueError(
                f"Parity bucket {bucket} has only {n} record(s) in {csv_path!r} -- "
                f"lactationcurve requires at least 2 non-missing (dim, milk) pairs to "
                f"fit at all, and a defensible fit of 4 MilkBot parameters needs "
                f"meaningfully more real records than that. Provide more data for "
                f"this parity bucket before fitting."
            )
        a, b, c, d = get_lc_parameters(
            df.loc[mask, dim_column].to_numpy(),
            df.loc[mask, milk_column].to_numpy(),
            model="milkbot",
        )
        params_by_parity[str(bucket)] = {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}

    payload = {
        "model": "milkbot",
        "parameters_by_parity": params_by_parity,
        "fitting_metadata": {
            "fitting_method": fitting,
            "data_source": data_source_note or os.path.abspath(csv_path),
            "n_records_by_parity": n_records_by_parity,
            "fit_date": datetime.date.today().isoformat(),
            "lactationcurve_package_version": getattr(_lc, "__version__", "unknown"),
        },
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_lactationcurve_milkbot_params(path=None):
    """
    Loads the frozen, offline-fit MilkBot parameters written by
    fit_and_save_lactationcurve_milkbot(). Raises a clear error naming
    exactly what's missing -- never invents or substitutes a value, never
    falls back to the existing Cow milk model.
    """
    path = path or DEFAULT_PARAMS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"lactationcurve milk-income reward selected, but no fitted MilkBot "
            f"parameters exist at {path!r}. This project does not invent, guess, "
            f"or substitute a parameter value. Populate this file by running "
            f"fit_and_save_lactationcurve_milkbot(csv_path=...) -- or, from the "
            f"command line, `py -3.12 cow/rewards/lactationcurve_milk_income.py "
            f"<real_data.csv>` -- against a real CSV containing DIM + observed "
            f"milk production + parity, using the actual lactationcurve package."
        )
    with open(path) as f:
        raw = json.load(f)
    params_by_parity = raw.get("parameters_by_parity", {})
    result = {}
    missing = []
    for bucket in (1, 2, 3):
        entry = params_by_parity.get(str(bucket))
        if entry is None or any(entry.get(k) is None for k in ("a", "b", "c", "d")):
            missing.append(bucket)
        else:
            result[bucket] = {k: float(entry[k]) for k in ("a", "b", "c", "d")}
    if missing:
        raise ValueError(
            f"{path!r} exists but is not calibrated for parity bucket(s) {missing} "
            f"(a/b/c/d missing or null). Re-run fit_and_save_lactationcurve_milkbot "
            f"with data covering every parity bucket."
        )
    return result


# ============================================================
# RUNTIME EVALUATION (calls the ACTUAL lactationcurve package;
# NEVER fits -- only evaluates already-frozen parameters)
# ============================================================

def lactationcurve_milkbot_milk(dim, lactation, params=None):
    """
    Predicted milk quantity (kg/day) via lactationcurve.fitting.milkbot_model
    -- a genuine call into the actual installed package, not a manually
    reproduced equation.

    dim: Cow-state DIM, in this project's MODEL-STEP units (converted to
        real days via UNIT_DAYS, the same convention every other Cow milk
        function uses).
    lactation: Cow-state lactation/parity number.
    params: FOR TESTING/INJECTION ONLY -- a dict {1: {"a":..,"b":..,"c":..,
        "d":..}, 2: {...}, 3: {...}}. Production callers should never pass
        this; omitting it (the default) loads the real, saved fit via
        load_lactationcurve_milkbot_params(), which is what makes the result
        a genuine calibrated value rather than an arbitrary one.
    """
    if params is None:
        params = load_lactationcurve_milkbot_params()  # raises clearly if uncalibrated -- checked BEFORE importing the package, so an uncalibrated call always reports "no fitted parameters", never a confusing package-import error

    from lactationcurve.fitting import milkbot_model  # lazy -- see module docstring

    dim = np.asarray(dim)
    dim_days = dim * UNIT_DAYS
    lac_bucket = _parity_bucket(lactation)
    t = np.maximum(dim_days, 0)

    out = np.zeros_like(t, dtype=float)
    for bucket in (1, 2, 3):
        mask = lac_bucket == bucket
        if not mask.any():
            continue
        p = params[bucket]
        out[mask] = milkbot_model(t[mask], a=p["a"], b=p["b"], c=p["c"], d=p["d"])
    return np.maximum(0, out)


# ============================================================
# REWARD SPEC (same shape as every other Cow reward spec;
# runnable through the
# existing mrp_engine.run_specs machinery, NOT registered in
# cow/rewards/__init__.py::BASE_REWARD_SPECS)
# ============================================================

def lactationcurve_milk_income_formula(states, probs):
    """
    Cow!milk_income's exact economics (milk_actual * MILK_PRICE_PER_KG *
    UNIT_DAYS), with milk_actual sourced from lactationcurve's MilkBot
    prediction instead of milk_yield_kg. Existing pregnancy/health adjustment
    is NOT applied here -- this reward prices lactationcurve's raw
    (dim, lactation)-conditional prediction directly, matching the minimal
    scope requested; reapplying pregnancy/health downstream would be a
    separate, later decision, not assumed here.
    """
    dim = states['dim'].values
    lac = states['lactation'].values
    milk_qty = lactationcurve_milkbot_milk(dim, lac)
    return milk_qty * MILK_PRICE_PER_KG * UNIT_DAYS


reward_lactationcurve_milk_income_spec = {
    'name': 'reward_lactationcurve_milk_income',
    'sign': +1,
    'depends_on': {
        'state': ['dim', 'lactation'],
        'derived_state': [], 'probability_output': [], 'reward_output': [],
        'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': lactationcurve_milk_income_formula, 'model': None},
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "OFFLINE: fit lactationcurve MilkBot parameters from a real "
            "DIM + milk-yield + parity CSV. Run under the Python 3.12 "
            "interpreter that has `lactationcurve` installed, e.g. "
            "`py -3.12 cow/rewards/lactationcurve_milk_income.py data.csv`."
        )
    )
    parser.add_argument("csv_path")
    parser.add_argument("--dim-column", default="dim")
    parser.add_argument("--milk-column", default="milk_yield_kg")
    parser.add_argument("--parity-column", default="lactation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = fit_and_save_lactationcurve_milkbot(
        args.csv_path,
        dim_column=args.dim_column,
        milk_column=args.milk_column,
        parity_column=args.parity_column,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))

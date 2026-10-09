"""
config.py
=========
The single place to define a benchmark run. Edit this file, then run:

    python benchmark.py

WHAT'S IN HERE
--------------
1. EXPERIMENTS   -- the model configs to compare (Model A / B / C / ...).
                    Each one overrides a subset of cow/parameters.py's
                    state-space knobs and runs in its own clean subprocess
                    (see benchmark_worker.py), so different MAX_DIM_DAYS /
                    MAX_LACTATION / TIME_UNIT per model never bleed into
                    each other.
2. TRACK_STATES  -- which cow states to plot milk-production trajectories
                    for (per-step, not cumulative), both finite-horizon
                    and infinite-horizon (run to steady-state).
3. Trajectory / output settings.

STATE SCHEMA for TRACK_STATES entries
--------------------------------------
    lactation : int, 1 .. MAX_LACTATION for that model
    dim_days  : int >= 0   (days in milk -- NOT the model's internal step
                             count; converted automatically)
    dp_days   : int >= 0   (days pregnant; 0 = not pregnant)
The state is just (lactation, DIM, DP); reproductive status (open / pregnant /
dry) is read off dp, and there is no health, flag or milk-level variable.

If a (lactation, dim_days, dp_days) combination isn't a real state,
benchmark_worker.py will raise a clear error listing what IS valid for the
rest of your query -- fix TRACK_STATES and re-run.
"""

# ============================================================
# OUTPUT
# ============================================================
OUTPUT_DIR = "outputs/benchmark"

# ============================================================
# EXPERIMENTS (Model A / B / C / ...)
# ============================================================
# 'overrides' keys map 1:1 to environment variables read by
# cow/parameters.py: COW_TIME_UNIT, COW_MAX_LACTATION,
# COW_MAX_DIM_DAYS, COW_MAX_DP_DAYS. Omit a key to keep that
# knob at parameters.py's own default.
EXPERIMENTS = [
    {
        "name": "Model A",
        "overrides": {
            "COW_TIME_UNIT": "month",
            "COW_MAX_LACTATION": 10,
            "COW_MAX_DIM_DAYS": 1000,
            "COW_MAX_DP_DAYS": 300,
        },
        "gamma_annual": 0.95,
        "num_years": 5,
        "run_finite": True,
        "run_infinite": True,
        "save_csv": True,
        "save_npz": True,
        "save_parquet": True,
    },
    {
        "name": "Model B",
        "overrides": {
            "COW_TIME_UNIT": "week",
            "COW_MAX_LACTATION": 10,
            "COW_MAX_DIM_DAYS": 1000,
            "COW_MAX_DP_DAYS": 300,
        },
        "gamma_annual": 0.95,
        "num_years": 5,
        "run_finite": True,
        "run_infinite": True,
        "save_csv": True,
        "save_npz": True,
        "save_parquet": True,
    },
    {
        "name": "Model C",
        "overrides": {
            "COW_TIME_UNIT": "day",
            "COW_MAX_LACTATION": 2,
            "COW_MAX_DIM_DAYS": 400,
            "COW_MAX_DP_DAYS": 300,
        },
        "gamma_annual": 0.95,
        "num_years": 1,
        "run_finite": True,
        "run_infinite": True,
        "save_csv": True,
        "save_npz": True,
        "save_parquet": True,
    },
]

# ============================================================
# STATES TO TRACK (milk production trajectory, per model)
# ============================================================
TRACK_STATES = [
    {"label": "Fresh heifer",           "lactation": 1, "dim_days": 0,   "dp_days": 0},
    {"label": "Peak lac2",              "lactation": 2, "dim_days": 90,  "dp_days": 0},
    {"label": "Late lac3, Open",        "lactation": 3, "dim_days": 250, "dp_days": 0},
]

# ============================================================
# VALUE COMPARISONS (diagnostic state-value differences, per model)
# ============================================================
# Each entry names a comparison from cow.value_comparisons.VALUE_COMPARISON_
# FUNCTIONS (the modular add/remove point -- see that module's docstring)
# plus the "params" it needs to identify the state(s) being compared. Any
# *_days key is real calendar days (converted to model steps automatically
# by benchmark_worker.py, same convention as TRACK_STATES above), so one
# entry works sensibly across Model A/B/C without editing per model.
#
# These are READ-ONLY DIAGNOSTICS computed from the already-retained V(s)
# -- they are reported alongside the reward breakdown but are NEVER added
# to rewards['total']. Add or remove entries freely; nothing else needs to
# change (no Core/Engine/benchmark_worker.py edits) unless you're adding a
# genuinely new comparison TYPE, in which case see cow/value_comparisons.py.
VALUE_COMPARISONS = [
    {"label": "Pregnancy value - lac2, DIM~150d",
     "comparison": "pregnancy_value",
     "params": {"lactation": 2, "dim_days": 150}},
    {"label": "Replacement value - lac2, DIM~150d, Open",
     "comparison": "replacement_value",
     "params": {"lactation": 2, "dim_days": 150, "dp_days": 0}},
]

# ============================================================
# TRAJECTORY SETTINGS
# ============================================================
# Infinite-horizon trajectory runs forward until the DISTRIBUTION
# converges to steady-state (L1 change between steps < tol), capped at
# max_steps as a safety net.
TRAJECTORY_CONVERGENCE_TOL = 1e-6
TRAJECTORY_MAX_STEPS = 2000

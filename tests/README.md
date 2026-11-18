# Tests

**Running.** `pytest` runs the fast set only (about 1 min: expected-distribution checks, month and week). Monte Carlo / stochastic weekly and all day-resolution parity tests are marked `slow` (about 13 min together) and the clean-install tests `install`; run them before a release with `pytest -m slow` and `pytest -m install`. Users can select any simulation mode in the package; the tests just do not run every mode at every resolution each time.

Run from the repository root with the project venv (`pip install -e ".[dev,plot,parquet]"`; the runner tests write Parquet files and the plot tests need matplotlib/plotly):

    pytest                       # the fast default set: unit, integration, regression, boundary, performance (about 1 min)
    pytest -m install            # builds the wheel, installs it in a fresh venv, uses it from another directory (needs network, about 1 min)

| Folder | What it protects |
|---|---|
| `unit/` | the generic engine on a hand-solvable 3-state MRP; the domain interface |
| `integration/` | `cowmrp.model` / `cowmrp.io` on the real domains; the migrated runner scripts use the package build |
| `integration/test_cow_projection.py` | cow A vs B against `baselines/projection_golden.json` (month, week via worker process, day = slow test) |
| `integration/test_cow_keepcull.py` | keep/cull table against the original runner (`baselines/keepcull_golden.json`) and against `replacement_comparison` |
| `integration/test_cow_farm.py` | whole-farm loader (raw and mapped records, invalid records), state mapping equal to generate.py, projection against the original runners (`baselines/farm_golden.json`, month / week / day), outputs, runners, example script |
| `integration/test_group_probs.py` | `group_probs` (integer-coded grouping) equals the original string-label groupby with `np.array_equal`: every grouper, edge-case inputs, month in-process, week (fast) and day (slow) in a child process |
| `regression/` | the scientific reference: `baselines/golden_month.json` (matrix checksums, state and nonzero counts, full and reachable values) |
| `boundary/` | `cowmrp.mrp` imports no domain and names no domain column; a non-animal toy domain (`machine_domain/`) runs the whole pipeline |
| `install/` | clean install: wheel contents, isolated interpreter (`-I`), unrelated working directory, golden numbers reproduced |
| `performance/` | generous time budgets, memory of `replacement_split`, vectorised `group_summary` |

The golden file was captured from the implementation in version 0.1.0. A failing regression test means a model number moved: find the cause;
never update the baseline to make a test pass.

## Known failures of the older check scripts (pre-existing, not regressions)

These fail identically in the tree before any packaging change; they are not part of this suite.

| Script | Reason |
|---|---|
| `runners/_shared/checks/verify_migration_parity.py` | compares whole-table means to the old 1,210-state baseline (stale since the full state table was introduced) |
| `runners/cow/checks/verify_lookup_export.py` | needs the archived `outputs/lookup_tables/effective` |
| `runners/cow/checks/audit_methods.py` | pandas 3: `pd.read_json(<literal string>)` is read as a path (needs `io.StringIO`) |
| `runners/cow/checks/audit_metrics.py` | needs arguments: `run <label> [--units month,week,day]`, `compare <labelA> <labelB> [--units ...]`, `list`; runs and compares correctly |
| `runners/cow/checks/compare_state_conventions.py` | arguments `UNIT VARIANT SCENARIO` (`month\|week\|day`, `current\|nominal`, `base\|vwp70\|conc80`) or none for the whole grid; the `current` variant runs, the `nominal` variant fails (it builds `StateConstraint` with an older signature) |

All other check scripts pass (`verify_state_space`, `verify_state_semantics`, `verify_reward_vector`, `verify_value_comparisons`,
`verify_time_discretization`, `verify_biological_consistency`, `verify_farm`, `verify_state_sufficiency`,
`verify_lactationcurve_milk_income_reward`, `verify_mrp_boundary`).

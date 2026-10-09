# CowMRP

A Markov Reward Process (MRP) model of a dairy cow, built to experiment with
states and rewards. The state is three variables, `(lactation, dim, dp)`:
lactation number, days in milk and days pregnant, in model-step units.

## Layout

| Folder | Purpose |
|---|---|
| `mrp/` | Domain-free MRP engine: `core.py` (math primitives), `engine.py` (spec orchestration/validation), `simulator.py` |
| `cow/` | The cow domain: parameters, states, probabilities, transitions, rewards, lookup tables, evidence |
| `runners/` | Scripts that use the above: `benchmark.py` (+ `config.py`), `farm/{build,generate,simulate}.py` |
| `projection/` | `run_projection.py`: project a cow's state distribution and rewards over the next N years (time axes in days) |
| `notebooks/` | `mrp_engine.ipynb`, the notebook runner |
| `verify/` | Validation and audit scripts |

Dependency direction is one-way: `cow` -> `mrp.engine` -> `mrp.core`.
`PRIMARY.md` is the engineering architecture record.

## Setup

Python 3.12 or newer.

```
pip install -r requirements.txt
```

## Usage

Scripts add the project root to `sys.path`, so they run from any directory.

```
python runners/benchmark.py              # benchmark Model A/B/C (see runners/config.py)
python projection/run_projection.py      # edit the scenarios at the top of the file first
python runners/farm/generate.py         # build a starting herd -> outputs/farm/farm_data.csv
python runners/farm/simulate.py --mode expected --steps 24   # herd projection -> outputs/farm/results/dashboard.html + dashboard_plotly.html
python verify/verify_state_space.py      # structural checks
python verify/verify_time_discretization.py
```

Time resolution (`day`, `week` or `month`) is set in `cow/parameters.py` and can be
overridden with the `COW_TIME_UNIT` environment variable.

### Choosing the cow to project

In `projection/run_projection.py`, `SCENARIOS` lists the cows. Every `dict(...)` line without a leading `#` is run; put `#` in front to switch one off. Each needs a `label`, a `time_unit`
(`day`, `week` or `month`), a `lactation` (1-10) and a state in calendar days (`dim_days`, `dp_days`) or model steps (`dim`, `dp`). `YEARS` sets the horizon.
Figures are written to `outputs/projection/` (open `index.html`) with time in days.

### Farm dashboards

`runners/farm/simulate.py` projects the herd in `outputs/farm/farm_data.csv` with the model's own transition matrix. It writes a static `dashboard.html` and an interactive
`dashboard_plotly.html` with the same layout: Open / Pregnant / Dry cards and summary, current-vs-predicted and evolution charts per grouping (status, lactation, DIM stage),
and a replacement-aware section (original herd, replacement herd, overlaid). A herd built at one time resolution is refused at another; regenerate it with `--time-unit`.

All generated files are written to `outputs/`, which is not tracked.

## License

MIT, see `LICENSE`. Copyright (c) 2026 Sumit Sharma.

## Notes

- `verify/verify_migration_parity.py` compares against a saved baseline at
  `outputs/baseline/Model_A/worker_result.json`; that file is not in the repository.
- `verify/validate_model_documentation.py` checks a local `docs/` folder that is not in the repository.
- Lookup tables can be overridden by CSV files in `cow/data/` (see `cow/data/README.md`).

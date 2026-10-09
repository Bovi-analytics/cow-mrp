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
| `projection/` | `run_projection.py`: project a cow's state distribution and rewards over the next N years |
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
python verify/verify_state_space.py      # structural checks
python verify/verify_time_discretization.py
```

Time resolution (`day`, `week` or `month`) is set in `cow/parameters.py` and can be
overridden with the `COW_TIME_UNIT` environment variable.

All generated files are written to `outputs/`, which is not tracked.

## Notes

- `verify/verify_migration_parity.py` compares against a saved baseline at
  `outputs/baseline/Model_A/worker_result.json`; that file is not in the repository.
- `verify/validate_model_documentation.py` checks a local `docs/` folder that is not in the repository.
- Lookup tables can be overridden by CSV files in `cow/data/` (see `cow/data/README.md`).

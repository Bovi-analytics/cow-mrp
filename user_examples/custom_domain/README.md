# Writing your own domain

A domain is a folder (a Python package) whose name is the domain name. Put it anywhere on `sys.path`, then `build_model("<name>")`.
`insemination/` is a complete, small example. **Its numbers are illustrative, not biological or economic estimates.**

| file | required | must provide |
|---|---|---|
| `__init__.py` | yes | can be empty |
| `parameters.py` | yes | `TIME_UNIT` (`"day"`, `"week"` or `"month"`) and `UNIT_DAYS` (positive integer, days per step) |
| `states.py` | yes | `build_<name>_states()` returning a DataFrame with a `state_id` column equal to the row number `0..n-1` |
| `probabilities.py` | yes | `compute_probabilities(states)` returning a dict of arrays, one value per state |
| `transitions.py` | yes | `compute_transitions(states, probs)` returning `(rows, cols, vals)` for the matrix `P[from, to]` |
| `rewards.py` | optional | `compute_rewards(states, probs)` returning a dict of arrays that includes `"total"`; without it there are no values |
| `notebook_hooks.py` | optional | `INITIAL_STATE`, `replacement_state_id(states)`, `STRICT`, ... see `cowmrp.domain` |

Rules the engine enforces: every row of the matrix sums to 1 (unless the domain sets `STRICT = False`); `validate_domain(...)` lists missing
pieces before anything is built. The engine itself contains no domain knowledge: it only assembles the matrix and solves for values.

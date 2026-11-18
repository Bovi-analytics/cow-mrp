# runners/

One folder per module. A module's runner uses its domain (`domains/<name>/`) with the generic engine (`mrp/`); no model logic lives here.

| Module | Runs | Notes |
|---|---|---|
| `cow/` | `run.py` (model + values), `benchmark.py`, `projection.py`, `farm/` | checks in `checks/`, figures in `plots/` |
| `cow/farm/keepcull.py` | `--excel` (accounting workbook), `--set NAME=VALUE` (economic scenario) | see the module docstring |
| `youngstock/` | `run.py` (calf -> heifer chain, cohort fates) | |
| `_shared/` | `common.py`, `new_domain.py`, `scale_benchmark.py` + `scale_report.py` (adapters: `<module>/scale_inputs.py`; `run.py --scale`), `checks/` | cross-module code and checks |

Every module has the same shape: `run.py`, `checks/`, `plots/`, `outputs/<kind>/[<time unit>/]`.

**Outputs.** Files are generated, never hand-edited. A script gets its folder from
`common.module_outputs(module, kind, ...)`; the allowed kinds (`model`, `validation`, `audit`, `lookup_tables`, `benchmark`,
`baseline`, `farm`, `projection`, `notebook`, `population`, ...) are the keys of `KINDS` in `_shared/common.py`. An unknown kind
or module is an error, so no stray folders appear.

**Shared build.** `_shared/engine_run.py` builds a domain with the generic engine (`mrp/`) and writes the standard model files,
so each `run.py` is thin and an engine improvement reaches every domain at once.

**Baselines.** `python runners/_shared/checks/verify_baseline.py [module]` compares a module's model manifest (state count,
non-zeros, matrix checksum, value means) with `outputs/baseline/model_<unit>.json`; `--save` writes it, only after you have
checked the change. Run it after any change to `mrp/`. Cow's older `verify_migration_parity.py` is kept.

**Run everything:** `python runners/run_all.py [--time-unit month|week|day] [--only cow,youngstock]`.

**Add a domain:** `python runners/_shared/new_domain.py <name>` creates `domains/<name>/` (stubs to fill in) and the full
`runners/<name>/` module. `run_all.py` picks it up as soon as it has a `run.py`.

**Maintainers' documentation check.** `_shared/checks/validate_model_documentation.py` compares the code with the maintainers' own
local documentation folder, which is not part of this repository. It is meant for the maintainers' working copy only and is
not needed to install, use or test the package; without that folder it reports the missing files and stops.

## The shared notebook

One notebook runs any domain: `runners/_shared/notebook/mrp_engine.ipynb`. It holds no domain logic; the domain comes from the
environment variable `MRP_DOMAIN` and is loaded from `domains/<name>/` (states, probabilities, transitions, optional rewards,
`parameters`). Optional extras (start state, discount convention, replacement state, grouping labels, `STRICT`) are read from
`domains/<name>/notebook_hooks.py`; a step whose hook is missing is skipped. Run it with `python runners/<module>/run.py --notebook`
(cow, youngstock) or `python runners/_shared/run_notebook.py <domain>`; the executed copy and the notebook's exports go to
`runners/<name>/outputs/notebook/`. `mrp/` is not involved in the choice of domain, and the notebook sits in `_shared` (not in
`mrp/`) because it imports domains.

## Review pages (cow, youngstock)

For a domain expert: one overview page listing every probability and reward of a domain (small picture, range, unit), and one detail
page per item (large interactive figure plus a "what to check" line). `python runners/<cow|youngstock>/run.py --review` (or
`python runners/_shared/review.py <domain>`) writes
`runners/<name>/outputs/review/index.html`. What to show is declared in `domains/<name>/review.py` (`NOTES`, `build_items(model)`); the
shared builder `runners/_shared/review.py` only draws, from the model's own arrays. The cow's older validation, projection and dashboard figures
(`runners/cow/outputs/validation`, `projection`, `dashboard`) are unchanged; its review pages sit beside them in `outputs/review/`.


## Local report folder (`views/`)

`python runners/run_all.py --views [--only cow,youngstock] [--skip validation,farm] [--steps 24] [--n-cows 10000]` (or `python runners/<cow|youngstock>/run.py --views`)
builds the browsable reports into the Git-ignored folder `views/` and writes the central page `views/index.html`. Its top part is the review grid of each domain
(small plots of every probability and reward, click for the full figure; the tiles come from the review builder, `runners/_shared/review.py`, so a new review item
appears automatically). Below it, the other reports are listed by domain,
resolution and type and says plainly when one is missing, failed or not available for a domain. The reports are built from the current package (nothing is copied
from older outputs). **Every new HTML report is added as one `Report(...)` entry in `runners/_shared/views.py`** so it appears on the central page.
`python runners/_shared/views.py --index-only` rebuilds only the page. Plotly is stored once (`views/_assets/plotly.min.js`). Use the project `.venv`.

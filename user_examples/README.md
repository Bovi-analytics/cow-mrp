# CowMRP examples

Five notebooks that show how to use CowMRP after `pip install`. Each notebook starts with an install cell and imports only from the
installed package (`cowmrp...`). Open them in Jupyter, run the cells top to bottom, then replace the example CSV with your own.

```
pip install "cowmrp[plot,excel]"        # the notebooks use Plotly (plot) and the keep/cull workbook (excel)
pip install notebook                    # or jupyterlab, if you do not have Jupyter yet
jupyter notebook
```

Download this folder (the notebooks and `data/`, plus `custom_domain/` for notebook 5) and open the notebooks from inside it:
they read `data/...` relative to their own folder.

| notebook | what it shows | your own input |
|---|---|---|
| `01_cow_csv_keepcull.ipynb` | cows from a CSV are mapped to monthly states, the reward of each state is looked up, and every cow gets a break-even sale value and a keep or cull decision; shows how to change economic settings for one run and writes the Excel accounting workbook | `HERD_CSV`, `OFFER`, `MY_SETTINGS` |
| `02_farm_projection.ipynb` | where the herd will be in 24 months (open, pregnant, dry, by lactation, original cows against replacements): a figure in the notebook and the interactive Plotly dashboard | `HERD_CSV`, `STEPS` |
| `03_individual_rewards.ipynb` | the reward vector: milk income, pregnancy reward, cull reward, mortality, feed and reproduction cost and their total, for every state and for every cow in your file | `HERD_CSV` |
| `04_youngstock_500.ipynb` | 500 young animals: how many are in each stage after 6 to 36 months and how many reach first calving, with a Plotly chart (the youngstock domain has no rewards; monthly resolution only) | the animals CSV |
| `05_custom_domain.ipynb` | your own states, probabilities and rewards on the generic engine, shown with a small insemination model: the transition matrix, the reward vector, state values and a distribution over time | the `custom_domain/insemination` folder |

## Your own CSV files

**Cows** (notebooks 1-3): one row per cow, columns `cow_id` (optional label), `lactation` (1-10), `dim_days` (days in milk), `dp_days` (days
pregnant, 0 if open). The loader maps each cow to a model state from these three numbers and lists any cow it cannot map. If you do not point
`HERD_CSV` at a file, `data/cows_1000.csv` is used.

**Young stock** (notebook 4): `stage` (CALF, HEIFER, BREEDING, PREGNANT), `age_months`, `dp_months`. Default: `data/youngstock_500.csv`.

**Both example files are synthetic.** `cows_1000.csv` is a random sample of 1,000 cows from a herd that an earlier version of CowMRP generated
from its own model; `youngstock_500.csv` is random with a fixed seed. Neither contains real animals. `make_data.py` documents how they were made.

## What the package provides and what the examples add

Everything the notebooks calculate comes from the installed package: `load_herd` (mapping cows to states and checking the file), `build_model` (states,
probabilities, rewards, transition matrix), `keep_cull_table` and `keep_cull_summary`, `write_keepcull_workbook` (Excel), `project_herd` and
`write_farm_outputs` (projection and dashboards), the `rewards` of the model, `evolve_distribution`, and for your own domain `load_domain`,
`validate_domain`, `build_model` and `evaluate`. The notebooks only choose inputs, print tables and draw figures. Notebook 5's `insemination` folder is
**example code written by the user side** (states, probabilities, transitions, rewards); the engine behind it is the package's.

## Limits you should know

* **Cows (1-3).** `dp_days` can be left out of the file (then every cow is treated as open). The model sees only lactation, days in milk and days
  pregnant: no health, milk record or body condition. The table shows the day value of the model state a cow was mapped to, which can differ from your
  number. The example CSV is month-binned synthetic data: at week or day resolution 28 of its 1,000 cows have a days-in-milk and days-pregnant combination that
  does not fit the model's rules (for example conception before day 50 in milk), so they have no state; the loader stops, lists them and suggests the closest valid states.
* **Keep/cull.** The break-even value compares a cow with a replacement heifer and does not credit the salvage value of a sale; your offer is that credit.
  Economic defaults are model inputs, not validated farm economics (notebook 1 lists each with its recorded source).
* **Projection (2).** Values are expected cows (averages), not a forecast of individual animals, and nothing outside the model (buying cows, a change of management) happens.
* **Young stock (4).** The domain has states, probabilities and transitions only: **no rewards, no values, no keep/cull, no herd loader**. The notebook's CSV is in
  months, so it needs month resolution. The semen-mix setting is a constant of the domain without a run-time switch.
* **Your own domain (5).** Supported through the public interface; `insemination` is a toy with illustrative numbers. There is no helper that reads states or
  probabilities from a spreadsheet: you write the four small Python files.
* **Daily resolution.** The CowMRP package supports monthly, weekly and daily resolution (`COW_TIME_UNIT`); the five notebooks are illustrative examples, not limits of the package. You can build your own daily-resolution workflows, with cow data that is compatible with the daily state rules (days in milk and days pregnant that the model allows). The bundled `cows_1000.csv` is intended for monthly resolution and does not map completely at weekly or daily resolution (28 of 1,000 cows have no state). The complete example notebooks have **not been verified** at daily resolution, and daily runtime and memory requirements (about 2 million cow states) have **not been benchmarked**. The youngstock notebook (4) is limited to monthly resolution because its CSV has ages in months.

## Domain, time resolution, economic settings

* **Domain.** `build_model("cow")` for cows (lactation 1-10, days in milk, days pregnant), `build_model("youngstock")` for calves and heifers,
  or the name of your own domain folder (notebook 5). Keep/cull and the herd projections exist for the cow domain only; the young stock domain
  has no rewards.
* **Time resolution.** Month is the default. Set `COW_TIME_UNIT` to `week` or `day` **before the first import of cowmrp** (the notebooks show
  where: `os.environ["COW_TIME_UNIT"] = "week"`). One resolution per Python session. Day resolution has about 2 million states.
* **Economic settings.** Anything you do not set uses the default. In notebook 1, put the settings you want to change in `MY_SETTINGS`
  (for example `{"REARING_COST": 2800}`); the notebook lists every setting that can be changed.

## Files

* `make_notebooks.py` writes the notebooks from the `NN_*.py` files next to them (same code, so either can be edited).
* `check_installed.py` is the local installation test (below). It is for maintainers.

## Testing the examples against the installed package

```
python check_installed.py
```

It builds the wheel and sdist, creates a clean virtual environment, installs the wheel there (not the source), checks that `cowmrp` is imported
from `site-packages`, runs the scripts and executes every notebook in the clean environment. It uploads nothing.

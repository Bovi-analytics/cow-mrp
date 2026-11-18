# cow/data -- optional table overrides

Drop a CSV here to replace a default lookup table; it is picked up automatically
(`cow/lookup_tables.py::get_full_table`). Nothing in this folder is needed to run the model.

Names: `mortality_full`, `culling_age_stage_full`, `conception_full`, `abortion_full`, `milk_income_full`.

1. Export the defaults to see the exact layout:
   `python -c "import cow.probabilities, cow.rewards; from cow.lookup_tables import export_full_tables; export_full_tables('model_outputs/lookup_tables')"`
2. Copy the CSV you want to change into this folder (keep the filename), edit the value column
   (`p` = MONTHLY probability by month since calving / month of gestation; `income_per_day` for milk income).
3. Every lactation 1..MAX_LACTATION x every month (x status Open/Pregnant/Dry where present) must be present --
   a gap is a hard error naming the missing rows. Probabilities must be in [0, 1].

Values are native-monthly and independent of TIME_UNIT: the time-discretization rule in
`cow/lookup_tables.py` converts them to per-step probabilities at day / week / month.
Delete the file to go back to the default.

Two-clock tables (`mortality_full`, `culling_age_stage_full`): the `status` column selects the clock.
`Open` rows: `month` = month since calving (DIM). `Pregnant` and `Dry` rows: `month` = month of PREGNANCY (days pregnant),
because the evidence (Pinedo et al. 2010) reports pregnant cows against days pregnant. Dry is the late part of the same clock;
rows must still exist for every month. `basis` must stay in {source, derived, repeated, placeholder}.

**Excel workflow.** `python -m cowmrp.domains.cow.lookup_workbook export` -> edit `outputs/lookup_tables/CowMRP_lookup_tables.xlsx` -> `python -m cowmrp.domains.cow.lookup_workbook import`. The import writes only real changes here as `<table>.csv`; the model reads these CSVs (never the workbook) and caches them in memory. `python -m cowmrp.domains.cow.lookup_workbook status` / `reset --yes`.

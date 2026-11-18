"""
runners/cow/farm/keepcull.py
========================
Keep / cull table for every cow in the herd file over one or more horizons (years). Each cow is compared with a fresh
replacement heifer bought now, over the SAME horizon and discount (cow/value_comparisons.py::replacement_comparison):

    value_N = keep_N(cow) - [keep_N(heifer) - cost of one replacement]      (USD, expected, discounted)

keep_N counts the replacement cost of every future exit, for the cow and her replacements alike. Read value_N as the
BREAK-EVEN SALE VALUE: culling and replacing her now beats keeping her only if the sale brings more than value_N (the
salvage of a sale is not credited in the model's own comparison, a documented limitation, and no sale price is invented
here; --excel writes a workbook that shows the sale of the cow now as its own column). Give the offer you actually get with --offer (USD per cow) and the table adds a KEEP / CULL call per horizon:
CULL when offer > value_N. Without --offer the table only ranks the cows (rank 1 = lowest break-even = first to sell).
One backward solve per horizon values every state; the herd is a lookup. No model logic lives here: the calculation is
cowmrp.domains.cow.keepcull, which applies value_comparisons.replacement_comparison to every cow of the herd.

    python runners/cow/farm/keepcull.py [--years 1,3,5] [--offer 900] [--farm-csv runners/cow/outputs/farm/farm_data.csv] [--gamma 0.95]
                                        [--time-unit month|week|day] [--out-dir DIR] [--excel] [--set NAME=VALUE ...]

OUTPUTS (runners/cow/outputs/farm/keepcull/): keepcull.csv (one row per cow: value_<N>y for each horizon, rank_longest, and with
--offer decision_<N>y and `stable`) and summary.csv (value spread and, with --offer, cull counts per horizon and lactation).
--excel also writes keepcull_accounting.xlsx (KEEP, REPLACE, immediate salvage and break-even per cow and horizon, as Excel formulas).

SCENARIOS: --set NAME=VALUE (repeatable) overrides an economic setting for this run only, e.g. --set REARING_COST=2800 --set MILK_PRICE_PER_KG=0.52.
The supported settings are listed in cowmrp.domains.cow.economics.ECONOMIC_SETTINGS; anything not named keeps its default. The results go to
keepcull_scenario/ (not over the default outputs) together with keepcull.settings.json, which records every overridden value. Without --set nothing changes
and no extra file is written.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root
from runners.cow.farm.build import set_time_unit_from_argv
set_time_unit_from_argv()

import argparse
import os

from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS
from cowmrp.domains.cow.rewards import compute_rewards
from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
from cowmrp.domains.cow.keepcull import keep_cull_table, keep_cull_summary          # the calculation lives in the package
from cowmrp.model import Model, build_model
from runners.cow.farm.build import FARM_DIR
from runners.cow.farm.simulate import load_herd


def keepcull_table(states, probs, matrix, farm_df, rid, years, gamma_annual=0.95, offer=None):
    """Kept for callers of the previous version of this script; same result as cowmrp.domains.cow.keepcull.keep_cull_table (rid is derived there)."""
    return keep_cull_table(farm_df, Model(states, probs, compute_rewards(states, probs), matrix), years, gamma_annual, offer)


summary = keep_cull_summary


def _parse_overrides(items):
    """['NAME=VALUE', ...] -> {NAME: float}; a malformed item is an error."""
    out = {}
    for item in items:
        name, sep, val = item.partition("=")
        try:
            out[name.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"--set expects NAME=VALUE with a number, got {item!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1,3,5", help="comma list of horizons in years")
    ap.add_argument("--offer", type=float, default=None, help="sale offer per cow in USD; adds KEEP/CULL calls (CULL when offer > value)")
    ap.add_argument("--farm-csv", default=os.path.join(FARM_DIR, "farm_data.csv"))
    ap.add_argument("--gamma", type=float, default=0.95, help="annual discount")
    ap.add_argument("--time-unit", default=None, help="handled before import (month|week|day)")
    ap.add_argument("--out-dir", default=os.path.join(FARM_DIR, "keepcull"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                    help="scenario run: override an economic setting for this run only (repeatable), e.g. --set REARING_COST=2800 --set MILK_PRICE_PER_KG=0.52; "
                         "see cowmrp.domains.cow.economics.ECONOMIC_SETTINGS. Results go to <out-dir>_scenario unless --out-dir is given; the settings are recorded in keepcull.settings.json")
    ap.add_argument("--excel", action="store_true", help="also write keepcull_accounting.xlsx (KEEP, REPLACE, immediate salvage, break-even per cow and horizon; needs openpyxl)")
    a = ap.parse_args()
    years = [int(x) for x in a.years.split(",")]

    overrides = _parse_overrides(a.overrides)
    if overrides and a.out_dir == os.path.join(FARM_DIR, "keepcull"):
        a.out_dir += "_scenario"                      # a scenario never overwrites the default outputs
    from contextlib import nullcontext
    if overrides:
        from cowmrp.domains.cow.economics import economic_overrides, reprice_model, validate_overrides
        try:
            validate_overrides(overrides)
        except ValueError as e:
            raise SystemExit(f"--set: {e}")
    with (economic_overrides(**overrides) if overrides else nullcontext()) as record:
        model = build_model("cow")
        if overrides:
            model = reprice_model(model)              # rewards with the overridden prices; states, probabilities and P are the same objects
        herd, _ = load_herd(a.farm_csv, len(model.states))
        table = keep_cull_table(herd, model, years, a.gamma, a.offer)
        summ = keep_cull_summary(table, years)
        os.makedirs(a.out_dir, exist_ok=True)
        table.to_csv(os.path.join(a.out_dir, "keepcull.csv"), index=False)
        summ.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
        if a.excel:
            from cowmrp.domains.cow.keepcull_excel import write_keepcull_workbook
            print("workbook:", write_keepcull_workbook(herd, os.path.join(a.out_dir, "keepcull_accounting.xlsx"), model, years, a.gamma))
        if overrides:
            import json
            with open(os.path.join(a.out_dir, "keepcull.settings.json"), "w", encoding="utf8") as f:
                json.dump(record, f, indent=2)
            print("NON-DEFAULT economic settings (scenario run): " + ", ".join(f"{n} = {o['value']:g} (default {o['default']:g})" for n, o in record["overrides"].items()))

    print(f"{len(table)} cows | {TIME_UNIT} ({UNIT_DAYS} d/step) | horizons {years} y | gamma {a.gamma} | "
          f"replacement cost {cost_per_replacement():,.0f} | value = break-even sale value (salvage not credited)"
          + (f" | offer {a.offer:,.0f} per cow" if a.offer is not None else " | no offer given: ranking only"))
    print(summ[summ["lactation"] == "all"].round(1).to_string(index=False))
    if "stable" in table:
        print(f"decision unchanged across horizons for {100 * table['stable'].mean():.1f}% of cows")
    print(f"written to {a.out_dir}")


if __name__ == "__main__":
    main()

"""cowmrp.domains.cow.economics -- temporary overrides of the economic constants, for scenario runs.

What this module does
    The prices and costs of the cow rewards are constants in ``cowmrp.domains.cow.rewards.base``; each reward module copies the ones it uses when it is
    imported. ``economic_overrides`` replaces the named constants, in every module that holds a copy, for the duration of a ``with`` block and restores
    the original values when the block ends (also after an exception). A constant that is not named is never touched, so it keeps exactly its default.
    ``reprice_model`` recomputes the reward vector with the constants as they are at that moment and returns a new model whose ``states``, ``probs`` and
    ``P`` are the SAME objects as the input model: the transition matrix and the biological probabilities cannot change.

        from cowmrp.model import build_model
        from cowmrp.domains.cow.economics import economic_overrides, reprice_model
        from cowmrp.domains.cow.keepcull import keep_cull_table

        model = build_model("cow")
        with economic_overrides(MILK_PRICE_PER_KG=0.52, REARING_COST=2800) as record:
            scenario = reprice_model(model)                       # rewards with the overridden prices; P unchanged
            table = keep_cull_table(herd, scenario, years=(1, 3, 5))      # the KEEP/CULL layer reads REARING_COST while the block is open
        # here every constant is back to its default; ``record`` lists what was overridden

    ``REARING_COST`` is not part of the reward vector. It is read by the KEEP/CULL layer (``replacement_cost_vector`` and ``cost_per_replacement``), so a
    keep/cull call must be made inside the ``with`` block for the override to apply.

Limits (read before use)
    * NOT THREAD-SAFE. The override replaces module-level names of the running process. It is meant for sequential scripts and notebooks. Do not use it
      from threads, or from several scenarios at the same time in one process. Nested ``with economic_overrides`` blocks are refused.
    * It overrides only the settings in ``ECONOMIC_SETTINGS``. The discount rate (``annual_discount``) and the sale offer (``offer``) are already
      arguments of the keep/cull, farm and projection functions and are not handled here.
    * Overrides are scenarios. Results computed with them are not the package defaults, and the regression baselines apply to the defaults only.
    * ``ECONOMIC_SETTINGS`` describes the existing defaults (read from ``rewards/base.py``); it is not their source. Metadata that is not recorded in
      the repository is stored as ``None``; nothing is filled in by guess.
"""
from __future__ import annotations

import difflib
import importlib
import math
import warnings
from contextlib import contextmanager

import numpy as np

_BASE = "cowmrp.domains.cow.rewards.base"
_R = "cowmrp.domains.cow.rewards."

# name -> reward modules that hold their own copy of the constant (``rewards.base`` itself is always patched as well).
# tests/integration/test_cow_economics.py scans the package and fails if a module holds a copy that is not listed here.
HOLDERS = {
    "MILK_PRICE_PER_KG": (_R + "milk_income", _R + "lactationcurve_milk_income"),
    "FEED_PRICE_PER_KG_DM": (_R + "feed_cost",),
    "DRY_PERIOD_DAILY_COST": (_R + "feed_cost",),
    "CALF_VALUE": (_R + "pregnancy_reward",),
    "BEEF_PRICE_PER_KG": (_R + "cull_reward",),
    "DRESSING_PCT": (_R + "cull_reward",),
    "CULLING_COST": (_R + "cull_reward",),
    "DISPOSAL_COST": (_R + "mortality_cost",),
    "AI_COST": (_R + "reproduction_cost",),
    "DIAGNOSIS_COST": (_R + "reproduction_cost",),
    "SYNCHRONIZATION_COST": (_R + "reproduction_cost",),
    "ABORTION_COST": (_R + "reproduction_cost",),
    "DAYS_OPEN_PENALTY_PER_DAY": (_R + "reproduction_cost",),
    "REARING_COST": (_R + "replacement_cost",),
}

# Descriptive metadata (what the repository records; None = not recorded). Sources: comments in rewards/base.py.
_META = {
    "MILK_PRICE_PER_KG": dict(unit="USD per kg milk", price_year=None, geography="US", affects="rewards (milk_income)",
                              source="USDA NASS All Milk Price; the price year is not recorded"),
    "FEED_PRICE_PER_KG_DM": dict(unit="USD per kg dry matter", price_year=None, geography=None, affects="rewards (feed_cost)",
                                 source="no source recorded (rewards/base.py)"),
    "DRY_PERIOD_DAILY_COST": dict(unit="USD per dry cow per day", price_year=None, geography=None, affects="rewards (feed_cost)",
                                  source="no source recorded; an incremental premium, not the total dry-cow ration cost (rewards/base.py)"),
    "CALF_VALUE": dict(unit="USD per calving", price_year="2026", geography="US (regional auction reports)", affects="rewards (pregnancy_reward)",
                       source="2026 US auction and USDA AMS reports, 50% heifer / 50% bull calves (rewards/base.py); market figures"),
    "BEEF_PRICE_PER_KG": dict(unit="USD per kg DRESSED weight", price_year="2023-2024", geography="US", affects="rewards (cull_reward)",
                              source="USDA AMS Livestock Market News cull-cow reports, dressed basis (rewards/base.py)"),
    "DRESSING_PCT": dict(unit="fraction (carcass / live weight)", price_year=None, geography=None, affects="rewards (cull_reward)",
                         source="a cull-dairy-cow carcass-traits study, 325 kg carcass / 701.5 kg live (rewards/base.py); medium confidence"),
    "CULLING_COST": dict(unit="USD per culled cow", price_year=None, geography=None, affects="rewards (cull_reward)",
                         source="several sale-barn fee schedules and one producer-reported cost of 55.75 (rewards/base.py)"),
    "DISPOSAL_COST": dict(unit="USD per dead cow", price_year="2002 (US study)", geography="US and Ontario", affects="rewards (mortality_cost)",
                          source="a 2002 US study and Ontario rendering fees; described as the weakest-sourced value in rewards/base.py"),
    "AI_COST": dict(unit="USD per service", price_year=None, geography=None, affects="rewards (reproduction_cost)",
                    source="general ranges of 20-100 (rewards/base.py)"),
    "DIAGNOSIS_COST": dict(unit="USD per service", price_year=None, geography=None, affects="rewards (reproduction_cost)",
                           source="5-10 per head for palpation or blood test (rewards/base.py)"),
    "SYNCHRONIZATION_COST": dict(unit="USD per service", price_year=None, geography=None, affects="rewards (reproduction_cost)",
                                 source="single Ovsynch protocol cost breakdown (rewards/base.py)"),
    "ABORTION_COST": dict(unit="USD per pregnancy loss", price_year=None, geography=None, affects="rewards (reproduction_cost)",
                          source="no source recorded in rewards/base.py"),
    "DAYS_OPEN_PENALTY_PER_DAY": dict(unit="USD per open day", price_year=None, geography=None, affects="rewards (reproduction_cost)",
                                      source="3.20-5.10 per open day is attributed to De Vries (2006) but is UNVERIFIED: it is not in the paper; off by default"),
    "REARING_COST": dict(unit="USD per replacement heifer", price_year="2024", geography="US", affects="KEEP/CULL layer only (not the reward vector)",
                         source="2,258-2,651 range, midpoint used; no single published source cited (rewards/base.py)"),
}
_NONNEGATIVE = set(HOLDERS)          # a price or cost is not negative by definition
_FRACTION = {"DRESSING_PCT"}         # carcass / live weight: a fraction, so in (0, 1]


def _base():
    return importlib.import_module(_BASE)


def default_value(name):
    """The default of ``name`` as defined in ``rewards/base.py`` at this moment (inside an override block it is the overridden value; use ``DEFAULTS``)."""
    return getattr(_base(), name)


ECONOMIC_SETTINGS = {name: dict(_META[name], name=name, currency="USD", default=float(default_value(name)), holders=HOLDERS[name]) for name in HOLDERS}
DEFAULTS = {name: s["default"] for name, s in ECONOMIC_SETTINGS.items()}

_ACTIVE = None        # the record of the open override block, or None


class EconomicSettingsWarning(UserWarning):
    """A factual note about an overridden setting (not an error)."""


def active_record():
    """The record of the currently open ``economic_overrides`` block, or ``None`` (so reporting code can note non-default settings without a new argument)."""
    return _ACTIVE


def validate_overrides(values):
    """The validated ``{name: float}`` of ``values``; raises ``ValueError`` (unknown name, non-finite, negative price or cost, DRESSING_PCT outside (0, 1])."""
    out = {}
    for name, v in values.items():
        if name not in HOLDERS:
            close = difflib.get_close_matches(name, list(HOLDERS), n=3, cutoff=0.5)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"{name!r} is not an overridable economic setting.{hint} Supported: {', '.join(sorted(HOLDERS))}. "
                             f"(The discount rate and the sale offer are arguments of the keep/cull functions; biological, management and computational values are not overridable.)")
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
            raise ValueError(f"{name} must be a finite number, got {v!r}")
        v = float(v)
        if name in _NONNEGATIVE and v < 0:
            raise ValueError(f"{name} is a price or cost and cannot be negative, got {v:g}")
        if name in _FRACTION and not 0 < v <= 1:
            raise ValueError(f"{name} is a fraction of live weight and must be in (0, 1], got {v:g}")
        out[name] = v
    return out


def _notes(values):
    """Factual notes about the overridden settings (what the repository records; no judgement about correctness)."""
    notes = []
    if ("BEEF_PRICE_PER_KG" in values) != ("DRESSING_PCT" in values):
        one, other = ("BEEF_PRICE_PER_KG", "DRESSING_PCT") if "BEEF_PRICE_PER_KG" in values else ("DRESSING_PCT", "BEEF_PRICE_PER_KG")
        notes.append(f"{one} is overridden but {other} is not. cull_reward multiplies body weight by DRESSING_PCT and by BEEF_PRICE_PER_KG, and the repository "
                     f"documents the price as a dressed-weight price; check that the two values use the same basis.")
    if values.get("DAYS_OPEN_PENALTY_PER_DAY", 0.0) > 0:
        notes.append("DAYS_OPEN_PENALTY_PER_DAY is above 0. The default is 0: the model's documentation records that this penalty overlaps flows the model already "
                     "books (milk, calving, replacement) and that the model's implied cost per open day was about 6 USD with it, against the 3.20-5.10 attributed to De Vries (2006) "
                     "(that attribution is unverified: the range is not in the paper).")
    if "CALF_VALUE" in values:
        notes.append("CALF_VALUE is one blended value per calving (50% heifer / 50% bull calves at the default); it is not split by calf type.")
    if "REARING_COST" in values:
        notes.append("REARING_COST is not in the reward vector. It changes the KEEP/CULL layer (the one-time cost and the present value of future replacements); "
                     "run keep/cull calls inside the same override block.")
    return notes


def _record(values):
    over = {}
    for name, v in values.items():
        meta = ECONOMIC_SETTINGS[name]
        over[name] = {"default": meta["default"], "value": v, "unit": meta["unit"], "currency": meta["currency"], "price_year": meta["price_year"],
                      "geography": meta["geography"], "source": meta["source"], "affects": meta["affects"]}
    return {"overrides": over, "notes": _notes(values)}


@contextmanager
def economic_overrides(**values):
    """Temporarily set the named economic constants (see the module docstring). Yields a record: ``{"overrides": {name: {default, value, unit, ...}}, "notes": [...]}``.

    Unknown names and non-finite, negative (prices and costs) or out-of-range (DRESSING_PCT must be in (0, 1]) values raise ``ValueError`` before anything is changed.
    Not thread-safe; nested blocks raise ``RuntimeError``. With no arguments the block changes nothing."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("economic_overrides cannot be nested: close the open block first (module-level override, not thread-safe)")
    values = validate_overrides(values)
    record = _record(values)
    saved = []
    try:
        _ACTIVE = record
        for name, v in values.items():
            for modname in (_BASE,) + HOLDERS[name]:
                mod = importlib.import_module(modname)
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, v)
        for note in record["notes"]:
            warnings.warn(note, EconomicSettingsWarning, stacklevel=3)
        yield record
    finally:
        for mod, name, old in reversed(saved):
            setattr(mod, name, old)
        _ACTIVE = None


def reprice_model(model):
    """A new model whose rewards are recomputed with the economic constants as they are now (inside an ``economic_overrides`` block: the overridden ones).

    ``states``, ``probs`` and ``P`` are the same objects as in ``model``; only ``rewards`` is new. Outside any block, or with no overrides, the rewards equal
    ``model.rewards`` exactly."""
    from cowmrp.domains.cow.rewards import compute_rewards
    return model._replace(rewards=compute_rewards(model.states, model.probs))

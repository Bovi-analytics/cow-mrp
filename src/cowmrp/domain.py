"""cowmrp.domain -- the interface a domain (cow, youngstock, or your own) provides to the generic MRP engine.

The MRP engine (``cowmrp.mrp``) is pure mathematics: sparse matrices, discounting, distributions, simulation. It knows no animal,
no state variable and no unit. A **domain** supplies everything that has meaning:

    <package>/parameters.py      TIME_UNIT ('day'|'week'|'month'), UNIT_DAYS (days per model step)     [required]
    <package>/states.py          build_<name>_states() -> DataFrame, one row per state, with a 'state_id' column [required]
    <package>/probabilities      compute_probabilities(states) -> dict of arrays                       [required]
    <package>/transitions.py     compute_transitions(states, probs) -> (rows, cols, vals)              [required]
    <package>/rewards            compute_rewards(states, probs) -> dict that includes 'total'          [optional]
    <package>/notebook_hooks.py  optional metadata (the "hooks" below)                                 [optional]

Optional hooks (module-level names in ``notebook_hooks.py``; absent = the documented default):

    STRICT                  True: every transition row must sum to 1 (default). False: rows are reported, not normalised
    INITIAL_STATE           dict of state-variable values for the default start state (default: state_id 0)
    gamma_per_step(annual)  the domain's own per-step discount convention (default: generic annual -> per-step)
    replacement_state_id(states)   state id a replaced animal re-enters (enables the replaced/never-replaced split)
    group_labels(states)    one label per state, for grouped summaries;   GROUP_NAME   what the grouping is called
    EXPORT_COLUMN_NAMES     {state variable: exported column name} (default: names unchanged)

A domain is found by shipped name (``"cow"``, ``"youngstock"``) or by the dotted path of any importable package
(``"my_package.my_domain"``) or by the name of any top-level package on the path (``"my_domain"``; the last part is ``<name>`` in ``build_<name>_states``). Use ``validate_domain`` to see what is missing.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType

REQUIRED = {
    "parameters": ("TIME_UNIT", "UNIT_DAYS"),
    "states": (),                       # build_<name>_states is checked separately (the name depends on the domain)
    "probabilities": ("compute_probabilities",),
    "transitions": ("compute_transitions",),
}


@dataclass(frozen=True)
class Domain:
    name: str
    package: str
    parameters: ModuleType
    states: ModuleType
    probabilities: ModuleType
    transitions: ModuleType
    rewards: ModuleType | None = None
    hooks: ModuleType | None = None

    # ---- metadata, always available -------------------------------------------------------------------------
    def hook(self, name, default=None):
        """A hook from ``notebook_hooks.py``, or `default` when the domain has no such hook."""
        return getattr(self.hooks, name, default) if self.hooks is not None else default

    @property
    def time_unit(self) -> str:
        return self.parameters.TIME_UNIT

    @property
    def unit_days(self) -> int:
        return self.parameters.UNIT_DAYS

    @property
    def has_rewards(self) -> bool:
        return self.rewards is not None

    @property
    def strict(self) -> bool:
        return bool(self.hook("STRICT", True))

    @property
    def export_column_names(self) -> dict:
        return dict(self.hook("EXPORT_COLUMN_NAMES", {}) or {})

    @property
    def build_states(self):
        return getattr(self.states, f"build_{self.name}_states")


def _package_of(domain: str) -> str:
    """A dotted path is used as is; a plain name is a shipped domain, else a top-level package of that name (your own domain)."""
    if "." in domain:
        return domain
    shipped = f"cowmrp.domains.{domain}"
    return shipped if importlib.util.find_spec(shipped) is not None else domain


def _optional(package: str, sub: str):
    return importlib.import_module(f"{package}.{sub}") if importlib.util.find_spec(f"{package}.{sub}") else None


def load_domain(domain) -> Domain:
    """Find a domain by shipped name (``"cow"``) or dotted package path (``"my_package.my_domain"``). A ``Domain`` is returned unchanged."""
    if isinstance(domain, Domain):
        return domain
    package = _package_of(domain)
    if importlib.util.find_spec(package) is None:
        raise ImportError(f"domain {domain!r} not found (looked for package {package!r})")
    mods = {sub: _optional(package, sub) for sub in ("parameters", "states", "probabilities", "transitions", "rewards", "notebook_hooks")}
    missing = [s for s in ("parameters", "states", "probabilities", "transitions") if mods[s] is None]
    if missing:
        raise ImportError(f"domain {package!r} has no {', '.join(missing)} module(s); a domain provides parameters, states, probabilities and transitions")
    return Domain(name=package.rsplit(".", 1)[-1], package=package, parameters=mods["parameters"], states=mods["states"],
                  probabilities=mods["probabilities"], transitions=mods["transitions"], rewards=mods["rewards"], hooks=mods["notebook_hooks"])


def validate_domain(domain) -> list[str]:
    """Problems that would stop a domain from building (an empty list = the interface is complete). Looks at names only; it builds nothing."""
    dom = load_domain(domain)
    problems = []
    for sub, names in REQUIRED.items():
        mod = getattr(dom, sub)
        problems += [f"{dom.package}.{sub} is missing {n}" for n in names if not hasattr(mod, n)]
    if not hasattr(dom.states, f"build_{dom.name}_states"):
        problems.append(f"{dom.package}.states is missing build_{dom.name}_states()")
    if dom.rewards is not None and not hasattr(dom.rewards, "compute_rewards"):
        problems.append(f"{dom.package}.rewards is present but has no compute_rewards()")
    if not isinstance(dom.unit_days, int) or dom.unit_days < 1:
        problems.append("UNIT_DAYS must be a positive integer number of days per step")
    if dom.time_unit not in ("day", "week", "month"):
        problems.append(f"TIME_UNIT must be 'day', 'week' or 'month', not {dom.time_unit!r}")
    return problems

"""
cow/state_constraints.py
========================
Which (lactation, dim, dp) combinations can exist -- as a declared, named, testable rule set. This module answers ONE
question (biological/temporal feasibility of a state). It does not build the state table (cow/states.py), compute
transitions (cow/transitions.py), check that transitions land on existing states, or check reachability
(cow/state_checks.py, runners/cow/checks/verify_state_semantics.py): those are separate concerns.

TIMING SEMANTICS THE RULES ARE DERIVED FROM (cow/lookup_tables.py, cow/tables.py `_entry_ages`)
  The model is observed at step boundaries ("instants"), U = UNIT_DAYS apart. A cow enters a phase at a uniform time
  inside the previous step, so
    * the age since calving at the instant of step k is   A = k*U + u,        u ~ Uniform(0, U)   (dim = k)
    * a pregnant cow conceived in step e at time s into that step; at the first instant after it (dp = 1) her
      gestation age is v = U - s,  v in (0, U);  at dp = j it is  (j-1)*U + v                       (dp = j)
    * so a cow in (dim = k, dp = j) conceived at age   a = A - ((j-1)*U + v) = (e+1)*U + u - v,   e = k - j,
      with u and v in (0, U). The state does NOT record u or v. It is not an exact-age interval: the same state
      pools cows whose (u, v) differ, and for states whose (u, v) are constrained by a rule (the first conception
      steps after the waiting period) the true (u, v) are NOT uniform (a known limitation of the offset coupling at the waiting-period edge). A rule can therefore only say whether a state is POSSIBLE (some (u, v) satisfies
      it), never what its exact age is.

RULE KINDS
  biological  : follows from the order of events (nothing can be conceived before the cow has calved).
  management  : an imposed policy/assumption (voluntary waiting period, last breeding day).
  convention  : a consequence of the time discretization (none needed so far; listed for completeness).
  numerical   : a bound that keeps the table finite (gestation range, reachable DIM). Not biology.
  Reachability and transition closure are checked elsewhere; they are requirements of the MRP, not state rules.

The split into biological and imposed constraints follows the approach described by Kalantari & Cabrera (2012,
J Dairy Sci 95:6160); the rules and code are this model's own.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import math
import numpy as np

from cowmrp.domains.cow.parameters import UNIT_DAYS, VOLUNTARY_WAITING_PERIOD_DAYS, STATE_DIM_CAP_DAYS


@dataclass(frozen=True)
class RuleParams:
    """Parameters the rules read. Defaults are the model's own; tests pass others explicitly.
    vwp_by_lactation lets a lactation-specific waiting period be declared (default: one value for all lactations,
    as in the model); the closure check in verify_state_semantics.py then shows whether the probabilities agree."""
    unit_days: int = UNIT_DAYS
    vwp_days: float = VOLUNTARY_WAITING_PERIOD_DAYS               # conception impossible through this day since calving
    dim_cap_days: float = STATE_DIM_CAP_DAYS                      # no conception step starts after this day (numerical extent)
    dp_max: Optional[int] = None                                  # numerical bound on dp (None = unbounded)
    dim_max: Optional[int] = None                                 # numerical bound on dim (None = unbounded)
    vwp_by_lactation: Optional[dict] = None                       # {lactation: vwp_days}; missing keys use vwp_days

    def vwp_for(self, lactation):
        """Waiting period (days) per state, as an array if `lactation` is an array."""
        if self.vwp_by_lactation is None:
            return self.vwp_days
        lac = np.asarray(lactation)
        out = np.full(lac.shape, float(self.vwp_days))
        for L, d in self.vwp_by_lactation.items():
            out = np.where(lac == L, float(d), out)
        return out


@dataclass(frozen=True)
class StateConstraint:
    name: str
    kind: str                       # biological | management | convention | numerical
    rationale: str                  # why the rule exists
    formula: str                    # the mathematical statement, in terms of dim, dp, e = dim - dp, U
    holds: Callable                 # (dim, dp, lactation, params) -> bool array
    valid_examples: Callable        # params -> list of (dim, dp) that this rule accepts and that exercise it
    invalid_examples: Callable      # params -> list of (dim, dp) that this rule must reject ([] if none exist)


def _pregnant(dp):
    return np.asarray(dp) > 0


def _e_min_vwp(p):
    """Smallest conception step the waiting period allows: (e+2)*U > vwp+1  <=>  e >= floor((vwp+1)/U) - 1."""
    return max(0, math.floor((p.vwp_days + 1) / p.unit_days) - 1)


def _e_first_blocked_dim_cap(p):
    """Smallest conception step that starts after the table's day cap: e*U > cap."""
    return math.floor(p.dim_cap_days / p.unit_days) + 1


CONSTRAINTS = (
    StateConstraint(
        'conception_after_calving', 'biological',
        'A pregnant cow conceived in a step that starts after the calving that opened the lactation. Step 0 is the first '
        'step after calving, so e = dim - dp = 0 (conception in the very first step window) is allowed here; whether it '
        'is possible is decided by the waiting-period rule, not by this one. "dp < dim" would wrongly forbid it.',
        'dp == 0  or  e >= 0   (equivalently dp <= dim)',
        lambda dim, dp, lac, p: ~_pregnant(dp) | (np.asarray(dim) - np.asarray(dp) >= 0),
        lambda p: [(3, 3), (5, 2)],
        lambda p: [(0, 1), (2, 3)]),
    StateConstraint(
        'conception_after_vwp', 'management',
        'No conception through day vwp after calving (first possible conception day vwp+1). The conception age of a '
        'cow in (k, j) is (e+1)*U + u - v < (e+2)*U, so the state is possible iff some (u, v) puts it above vwp+1. The '
        'waiting period restarts at every calving (calving resets dim and dp), so the rule is identical in every '
        'lactation unless a lactation-specific waiting period is declared.',
        'dp == 0  or  (e + 2) * U > vwp + 1',
        lambda dim, dp, lac, p: ~_pregnant(dp) | ((np.asarray(dim) - np.asarray(dp) + 2) * p.unit_days > p.vwp_for(lac) + 1),
        lambda p: [(_e_min_vwp(p) + 1, 1), (_e_min_vwp(p) + 3, 2)],
        lambda p: ([(_e_min_vwp(p), 1)] if _e_min_vwp(p) >= 1 else [])),
    StateConstraint(
        'conception_within_dim_cap', 'numerical',
        'The table holds every conception step up to STATE_DIM_CAP_DAYS, each with its whole pregnancy diagonal. The last '
        'breeding day is NOT a state rule: past it p_preg is 0 (cow/probabilities/conception.py), the states still exist, '
        'so a cow observed there can be valued and the layout does not depend on the breeding policy.',
        'dp == 0  or  e * U <= dim_cap_days',
        lambda dim, dp, lac, p: ~_pregnant(dp) | ((np.asarray(dim) - np.asarray(dp)) * p.unit_days <= p.dim_cap_days),
        lambda p: [(_e_first_blocked_dim_cap(p) - 1 + 1, 1)],
        lambda p: [(_e_first_blocked_dim_cap(p) + 1, 1)]),
    StateConstraint(
        'gestation_within_range', 'numerical',
        'The dp axis stops at the last gestation step (calving is certain from there). Not biology: it keeps the table finite.',
        'dp <= dp_max   (only if dp_max is declared)',
        lambda dim, dp, lac, p: np.ones(np.broadcast(dim, dp).shape, bool) if p.dp_max is None else (np.asarray(dp) <= p.dp_max),
        lambda p: [(60, p.dp_max)] if p.dp_max is not None else [],
        lambda p: [(60, p.dp_max + 1)] if p.dp_max is not None else []),
    StateConstraint(
        'dim_within_reachable_range', 'numerical',
        'The dim axis stops at the last reachable step (cull step of an open cow, last conception step + gestation + loss). Not biology.',
        'dim <= dim_max   (only if dim_max is declared)',
        lambda dim, dp, lac, p: np.ones(np.broadcast(dim, dp).shape, bool) if p.dim_max is None else (np.asarray(dim) <= p.dim_max),
        lambda p: [(p.dim_max, 0)] if p.dim_max is not None else [],
        lambda p: [(p.dim_max + 1, 0)] if p.dim_max is not None else []),
)


def feasible_mask(dim, dp, params=None, lactation=None):
    """Boolean array: True where (dim, dp) satisfies every constraint. `lactation` only matters if the params declare a
    lactation-specific waiting period; with the model's single waiting period the result is identical for every
    lactation (the rules depend on time since calving and since conception only)."""
    params = params or RuleParams()
    dim, dp = np.asarray(dim), np.asarray(dp)
    lac = np.ones(np.broadcast(dim, dp).shape, int) if lactation is None else np.broadcast_to(np.asarray(lactation), np.broadcast(dim, dp).shape)
    ok = np.ones(np.broadcast(dim, dp).shape, dtype=bool)
    for c in CONSTRAINTS:
        ok &= np.asarray(c.holds(dim, dp, lac, params))
    return ok


def violated(dim, dp, params=None, lactation=1):
    """Names of the constraints a single (dim, dp) violates (for tests and error messages)."""
    params = params or RuleParams()
    d, p_, l_ = np.asarray([dim]), np.asarray([dp]), np.asarray([lactation])
    return [c.name for c in CONSTRAINTS if not bool(np.asarray(c.holds(d, p_, l_, params))[0])]


def conception_age_interval(dim, dp, params=None):
    """(inf, sup) in days of the conception age of a cow observed in pregnant state (dim, dp): (e*U, (e+2)*U)."""
    u = (params or RuleParams()).unit_days
    e = dim - dp
    return e * u, (e + 2) * u


def bin_days(dim, params=None):
    """Nominal day range (first, last) of the bin the cow's age lies in at the instant. MONTH: dim 0 -> (1, 30) = MIM 1.
    This is a LABEL; the cow's position inside the bin is not tracked."""
    u = (params or RuleParams()).unit_days
    return dim * u + 1, (dim + 1) * u


def describe_state(lactation, dim, dp, params=None):
    """Plain-language meaning of a state under the semantics above (display only)."""
    params = params or RuleParams()
    u = params.unit_days
    out = {'lactation': int(lactation), 'bin_number': int(dim) + 1, 'age_bin_days': bin_days(dim, params)}
    if dp == 0:
        out['status'] = 'open'
        return out
    e = int(dim) - int(dp)
    out['status'] = 'pregnant'
    out['gestation_bin_days'] = ((dp - 1) * u + 1, dp * u)
    out['conception_step'] = e
    out['conception_age_days_between'] = conception_age_interval(dim, dp, params)
    out['feasible'] = bool(feasible_mask(dim, dp, params, lactation))
    return out

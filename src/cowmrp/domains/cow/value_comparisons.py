"""
cow/value_comparisons.py
=========================
Cow Domain definitions of scientifically meaningful state-value
comparisons, built entirely on the generic `mrp_core.state_values` /
`state_value_diff` primitives. This module owns BIOLOGICAL MEANING ONLY --
which two states, held otherwise identical, represent a scientifically
valid "with event" vs "without event" comparison. All arithmetic (state
lookup, value differencing) is delegated to `mrp_core`; nothing here
duplicates that logic, and nothing here is new Core or Engine machinery.

Each comparison function returns a plain dict --
`{"value": ..., "V_state": ..., "V_comparison": ...}` -- so callers (e.g.
`benchmark_worker.py`'s reporting layer) can show both raw values and the
difference without re-deriving anything. Every comparison function has
this same return shape, which is what lets the reporting layer stay
generic across comparisons (it never needs to know pregnancy-specific or
replacement-specific meaning -- it just reads three standard keys).

MODULARITY: `VALUE_COMPARISON_FUNCTIONS` (bottom of this file) is the
add/remove point. Adding a new comparison (pregnancy loss, conception,
day-open, lameness, disease, treatment, ...) means: (1) write one small
function here shaped like `pregnancy_value`/`replacement_value` below --
define two attribute-query dicts, call `mrp_core.state_values` -- and (2)
add one line to the registry dict. No Core, Engine, or `benchmark_worker.py`
change is ever needed for a new comparison; `config.py`'s `VALUE_COMPARISONS`
list (which comparisons actually run, and at which states) is the only
other place a user edits, and only that file. Removing a comparison is the
reverse: delete its `config.py` entries (and its registry line if retiring
it entirely). `pregnancy_value` and `replacement_value` are two concrete
examples of this pattern, not the architecture itself -- this module should
only ever grow by adding functions shaped like these, never by adding new
lookup/arithmetic machinery.

NOT reward specs: neither function is registered in `cow/rewards/`, neither
is appended to `reward_specs`, and neither affects `rewards['total']` or
`reward_environment_total`. These are read-only diagnostics -- call them
against an already-computed `states`/`V` pair (e.g. `benchmark_worker.py`'s
retained `V_finite`/`V_infinite` columns). Nothing here runs automatically
as part of the model pipeline (build_states -> compute_probabilities ->
compute_transitions -> compute_rewards -> evaluate_*_horizon); this module
is a consumer of that pipeline's output, not a participant in it. Where a
caller (`benchmark_worker.py`) *does* report these values alongside the
reward breakdown, they are always clearly separate diagnostic columns/keys,
never folded into any reward total.

COUNTERFACTUAL VALIDITY of `pregnancy_value`: comparing dp=1 (just
conceived) against dp=0 (Open) at the SAME dim/lactation is a valid
counterfactual pair, not merely a state comparison -- an Open cow's own
transitions already include next-period conception probability, so this
reads as "conceived this step vs. didn't, both still facing the same
ongoing biology/economics going forward," not "pregnant forever" vs
"never pregnant." Treat any causal reading of the resulting number as only
as trustworthy as the underlying transition/reward model itself (no empirical predictive validation
has been performed on this model). Not a decision/policy value: nothing
here chooses an action.

GESTATION-STAGE GENERALIZATION (dp > 1): `pregnancy_value`
also accepts dp > 1, comparing V(Pregnant-or-Dry, dim, dp) against
V(Open, dim, dp=0) at the SAME current dim -- the same "same-DIM-now"
pairing De Vries (2006) uses for his month-of-gestation VP_n (his
VP_i,k,mo,n,t = CV_n - CV_0 at fixed month-in-lactation mo, varying n).
This is a DIFFERENT counterfactual reading than the dp=1 case above: it
answers "what is this cow's pregnancy (now dp steps along) worth,
relative to an otherwise-identical cow who has been open this whole
time and is only just now, at this same dim, facing her first real
chance to conceive" -- not "did she conceive this exact step." Both
readings are legitimate, but they are not the same question, and the
gap between them is not corrected for here. As dp grows, the Open
comparison state (dp=0) increasingly represents a state the real herd
dynamics would rarely produce at that dim (most Open cows either
conceive or are culled well before then) -- the comparison remains
numerically well-defined (see BIOLOGICAL VALIDITY below) but its
practical interpretation should be read the same cautious way replace-
ment_value's fixed-comparison-state caveat already is elsewhere in this
file: a valid difference between two well-defined values, not a claim
about a typically-observed animal.

COUNTERFACTUAL VALIDITY of `replacement_value`: this is a DIFFERENT kind
of comparison than pregnancy_value -- not two possible futures of the same
cow, but the current cow's own state against a different animal's state
(a fresh replacement heifer). It answers "is this cow's own expected
future value higher or lower than a fresh replacement's, from here?", not
"what would happen to this cow if she were replaced" (replacing her ends
her trajectory; it does not put her IN the replacement's trajectory).
Uses the EXISTING replacement state and machinery exactly as
`cow/transitions.py` defines it (see BIOLOGICAL VALIDITY below) -- no new
replacement formula. It is also not a decision/policy value: a positive
`replacement_value` does not mean "the model recommends keeping her" --
that would require comparing against the *actual* economics of replacing
her now (salvage value, purchase/rearing cost, herd-slot availability),
which is the retention-pay-off (RPO) question, deliberately not
implemented here (see the design spec's "Not implemented" list -- still
true).

BIOLOGICAL VALIDITY: `pregnancy_value` queries dp=0 (Open) for the
comparison state and a caller-supplied dp >= 1 (default 1, i.e. "just
conceived") for the pregnant state, both at a matched dim/lactation.
The state is just (lactation, dim, dp); a dp at or past DRY_OFF_DP is a dry
cow (cow/states.py::repro_status), so a large dp correctly resolves into the
dry stage, matching De Vries (2006)'s months 8-9 of gestation (his "cows were not
lactating in mo 8 and 9"). `replacement_value` compares the caller-supplied current state
against the EXACT fixed replacement state `cow/transitions.py` itself
routes every culled/dead cow to -- `lactation=1, dim=0, dp=0` (verified
directly from `cow/transitions.py::get_replacement_state_id` -- not assumed,
not re-derived). Both queried states structurally exist in the state space for
any valid attribute combination (`cow/states.py` builds the full
lactation x dim x dp cross product), and `state_values`/
`state_value_diff` (via `find_state_id`) raise a clear `ValueError` with
valid-value hints if a caller's attributes don't resolve to a real row --
no new validation logic is added here, matching how the rest of this repo
already treats state-space construction (full cross product; DYNAMICS, not
table pruning, express what's actually reachable). One thing this does NOT
enforce: conception is only actually *modeled* as possible (p_preg > 0)
from dim >= ~50 days (see `cow/probabilities/conception.py`) -- querying
`pregnancy_value` at an earlier dim returns a real, internally consistent
number for a structurally valid but dynamically unreachable comparison,
not an error. `replacement_value`'s known caveat: the replacement state is
the SAME fixed state for every cow regardless of herd composition --
`cow/transitions.py` assumes unconstrained replacement-heifer supply (no
herd-slot/availability dimension exists), so this comparison is only
meaningful under that same assumption.

KNOWN CAVEAT (applies to `pregnancy_value`, and to `replacement_value`
wherever `reproduction_cost` differs between the two compared states): a
sensitivity experiment found that `cow/rewards/reproduction_cost.py`'s `DAYS_OPEN_PENALTY_PER_DAY`
and `ABORTION_COST` are recomputed fresh at EVERY visited state rather than
charged once, and account for 63-99% of `pregnancy_value`'s magnitude
depending on `TIME_UNIT` granularity. These functions report the model's
current value differences faithfully; they do not correct for this known
reward-formula issue, which remains out of scope for this module.
"""
import contextlib
import io

import numpy as np

from cowmrp.mrp.core import state_values, find_state_id, evaluate_infinite_horizon, evaluate_finite_horizon


def _require_pregnant_dp(dp):
    """A pregnancy in progress means dp >= 1 (dp == 0 is the open comparison state)."""
    if dp < 1:
        raise ValueError(f"dp must be >= 1 (a pregnancy in progress); got dp={dp}. "
                          f"dp=0 is the Open comparison state, not a valid pregnant query.")


def pregnancy_value(states, V, lactation, dim, dp=1):
    """
    Value of pregnancy = V(pregnant-or-dry, dp) - V(open), at matched
    lactation/DIM. dp defaults to 1 (the marginal value of
    conceiving THIS step rather than remaining open, the original
    behavior of this function, unchanged for existing callers). dp > 1
    generalizes this to a gestation-stage curve comparable to De Vries
    (2006)'s VP_n by month of gestation -- see the module docstring's
    "GESTATION-STAGE GENERALIZATION" section for the counterfactual
    reading this represents and how it differs from the dp=1 case.

    Returns {"value": V_state - V_comparison, "V_state": V_pregnant,
    "V_comparison": V_open}. Raises ValueError (propagated from
    mrp_core.find_state_id) if the requested lactation/dim/dp does
    not resolve to real states in `states`, or if dp < 1.
    """
    _require_pregnant_dp(dp)
    V_pregnant, V_open = state_values(
        states, V,
        query_A={"lactation": lactation, "dim": dim, "dp": dp},
        query_B={"lactation": lactation, "dim": dim, "dp": 0},
    )
    return {"value": V_pregnant - V_open, "V_state": V_pregnant, "V_comparison": V_open}


def decompose_pregnancy_value(states, matrix, rewards, gamma, lactation, dim, dp=1):
    """
    Per-reward-component breakdown of pregnancy_value at the given dp --
    how much of V(pregnant-or-dry, dp) - V(open) each of the
    BASE_REWARD_SPECS terms (six; replacement_cost is not a state reward) contributes, the same idea De Vries (2006)'s
    Table 3 uses (milk sales / calf sales / replacement cost / feed cost
    / breeding cost / other costs), generalized across dp instead of one
    point. Reproduces (and generalizes) the decomposition pattern
    verify_value_comparisons.py's check #8 already exercised at dp=1.

    Requires the transition matrix, the full per-component `rewards`
    dict, and `gamma` directly -- unlike pregnancy_value/replacement_value
    (which only need one pre-solved V), this re-solves the infinite-
    horizon value function once per component: V = (I - gamma*P)^-1 * R
    is linear in R for a fixed matrix, so summing each component's own
    V(sign * rewards[name]) diff at (pregnant, open) exactly reproduces
    the diff of the summed reward -- exploited, not merely assumed (see
    verify_value_comparisons.py for the numeric regression check of this
    identity).

    Raises RuntimeError if the signed sum of BASE_REWARD_SPECS does not
    equal rewards['total'] (true whenever no ENABLE_REWARD_* environmental
    component is folded into total, this engine's default) -- this
    decomposition only covers the base components and would silently
    under-count otherwise, rather than fail loudly.

    Returns {component_name: contribution}. sum(contribution.values())
    equals pregnancy_value(states, evaluate_infinite_horizon(matrix,
    rewards['total'], gamma), lactation, dim, dp)['value'].
    """
    import numpy as np
    from cowmrp.domains.cow.rewards import BASE_REWARD_SPECS

    total_check = sum(spec['sign'] * np.asarray(rewards[spec['name']]) for spec in BASE_REWARD_SPECS)
    if not np.allclose(total_check, rewards['total'], rtol=0, atol=1e-6):
        raise RuntimeError(
            "decompose_pregnancy_value: sum of BASE_REWARD_SPECS does not equal rewards['total'] -- "
            "likely an ENABLE_REWARD_* environmental component is folded into total; this "
            "decomposition only covers the 8 base components and would silently under-count."
        )

    _require_pregnant_dp(dp)
    id_pregnant = find_state_id(states, lactation=lactation, dim=dim, dp=dp)
    id_open = find_state_id(states, lactation=lactation, dim=dim, dp=0)

    contributions = {}
    for spec in BASE_REWARD_SPECS:
        name, sign = spec['name'], spec['sign']
        V_component = evaluate_infinite_horizon(matrix, sign * np.asarray(rewards[name]), gamma)
        contributions[name] = V_component[id_pregnant] - V_component[id_open]
    return contributions


def replacement_value(states, V, lactation, dim, dp):
    """
    Value of keeping this cow's own state rather than replacing her =
    V(current state) - V(replacement state), where the replacement state
    is the EXACT fixed state `cow/transitions.py` routes every
    culled/dead cow to. See the module docstring for the exact
    verification and for why this is a different kind of comparison than
    pregnancy_value (cross-animal, not same-animal counterfactual).

    Returns {"value": V_current - V_replacement, "V_state": V_current,
    "V_comparison": V_replacement}. Raises ValueError (propagated from
    mrp_core.find_state_id) if the requested current-state attributes do
    not resolve to a real state in `states`.

    NOTE: V here is the value of the state reward vector, which
    no longer contains replacement_cost. This difference therefore EXCLUDES
    every replacement cost; the comparison that accounts for it (future exits
    of both lineages, plus the heifer bought now, same horizon) is
    `replacement_comparison` below.
    """
    V_current, V_replacement = state_values(
        states, V,
        query_A={"lactation": lactation, "dim": dim, "dp": dp},
        query_B={"lactation": 1, "dim": 0, "dp": 0},
    )
    return {"value": V_current - V_replacement, "V_state": V_current, "V_comparison": V_replacement}


# ============================================================
# COW VERSUS REPLACEMENT HEIFER, WITH THE REPLACEMENT COST ACCOUNTED SEPARATELY
# ============================================================
# The state reward vector (rewards['total']) does NOT contain replacement_cost: the transition matrix already sends
# every exit (death, culling, calving at the last lactation) to the replacement state (1, 0, 0), and the cost of the
# replacement is charged here, over the SAME horizon and discount as the values being compared.
#
# ACCOUNTING CONVENTION (identical for both animals, so the comparison is like for like):
#   V_total(s)   = expected discounted sum of rewards['total'] from state s       (horizon H, or infinite)
#   PV_rc(s)     = expected discounted sum of the replacement-cost vector from s   (every FUTURE exit of the lineage
#                  starting in s costs one replacement; same matrix, same gamma, same H)
#   keep(s)      = V_total(s) - PV_rc(s)          the cow, then her replacements, paying for each replacement
#   heifer       = V_total(h) - PV_rc(h) - C      a fresh heifer in the replacement state h = (1, 0, 0), bought now
#                  (C = cost of one replacement, charged once at time 0)
#   value        = keep(s) - heifer = [V_total(s) - V_total(h)] - [PV_rc(s) - PV_rc(h)] + C
# The salvage that selling the current cow now would bring is NOT credited (the comparison is "cumulative expected
# reward of the cow vs a replacement heifer", not a retention pay-off); it is a documented omission, not a hidden one.

def _solve(matrix, reward, gamma, num_years):
    """Same solver and horizon for every stream: infinite when num_years is None, else the engine's finite horizon."""
    if num_years is None:
        return np.asarray(evaluate_infinite_horizon(matrix, reward, gamma))
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(evaluate_finite_horizon(matrix, reward, gamma, num_years, UNIT_DAYS))


def replacement_net_values(states, matrix, rewards, probs, gamma, num_years=None):
    """Per-state V_total, PV_rc and keep = V_total - PV_rc (arrays over states), one horizon for all three."""
    from cowmrp.domains.cow.rewards import replacement_cost_vector
    V_total = _solve(matrix, rewards['total'], gamma, num_years)
    PV_rc = _solve(matrix, replacement_cost_vector(states, probs), gamma, num_years)
    return {"V_total": V_total, "PV_replacement_cost": PV_rc, "V_keep": V_total - PV_rc}


def replacement_comparison(states, matrix, rewards, probs, gamma, lactation, dim, dp, num_years=None, net=None):
    """Cow in (lactation, dim, dp) versus a fresh replacement heifer bought now, same horizon (see the convention
    above). Returns {"value": keep - heifer, "V_state": keep, "V_comparison": heifer (net of the replacement cost
    paid now), plus the parts: V_total_state, V_total_heifer, PV_replacement_cost_state, PV_replacement_cost_heifer,
    replacement_cost_now, horizon}. Raises ValueError (from find_state_id) if the state does not exist.
    `net` may carry a precomputed replacement_net_values(...) result to avoid re-solving for many states."""
    from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
    net = net or replacement_net_values(states, matrix, rewards, probs, gamma, num_years)
    i = find_state_id(states, lactation=lactation, dim=dim, dp=dp)
    h = find_state_id(states, lactation=1, dim=0, dp=0)
    C = float(cost_per_replacement())
    keep_s = float(net["V_keep"][i])
    heifer = float(net["V_keep"][h]) - C
    return {"value": keep_s - heifer, "V_state": keep_s, "V_comparison": heifer,
            "V_total_state": float(net["V_total"][i]), "V_total_heifer": float(net["V_total"][h]),
            "PV_replacement_cost_state": float(net["PV_replacement_cost"][i]),
            "PV_replacement_cost_heifer": float(net["PV_replacement_cost"][h]),
            "replacement_cost_now": C,
            "horizon": "infinite" if num_years is None else f"{num_years} years"}


# Registry: name -> function. The add/remove point for new comparisons --
# see the module docstring's MODULARITY section. config.py's
# VALUE_COMPARISONS list references these names; nothing else in the repo
# depends on this dict's contents beyond that lookup.
VALUE_COMPARISON_FUNCTIONS = {
    "pregnancy_value": pregnancy_value,
    "replacement_value": replacement_value,
}

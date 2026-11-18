"""
verify_state_space.py
=====================
Dependency-free checks for the three-variable (lactation, DIM, DP) state space,
the precomputed milk/cull tables, and the pipeline built on them.
Run: python verify_state_space.py     (exit 1 on any failure)
Resolution: set COW_TIME_UNIT=day|week|month before running.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import sys
import numpy as np

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def check_tables():
    from cowmrp.domains.cow.parameters import (UNIT_DAYS, MAX_DIM, MAX_DP, MAX_LACTATION, DRY_OFF_DP, DRY_OFF_DP_DAYS, LAST_BREEDING_DIM,
                                MILK_FLOOR_KG, LAST_BREEDING_DIM_DAYS)
    from cowmrp.domains.cow.curves import curve_kg
    from cowmrp.domains.cow.tables import get_tables, state_milk_kg
    t = get_tables()
    # 1. step-mean x UNIT_DAYS summed over the first N steps ~= the continuous curve integrated over the
    #    window shifted by half a step (mean entry offset U/2): the table follows the entry convention.
    n = int(round(300 / UNIT_DAYS))
    for b in range(1, MAX_LACTATION + 1):
        days = np.arange(n * UNIT_DAYS) + 0.5 + UNIT_DAYS / 2.0
        direct = curve_kg(days, b).sum()
        stepped = (t["milk_curve"][b - 1, :n] * UNIT_DAYS).sum()
        check(f"step-mean sum matches the half-step-shifted continuous curve (lactation {b})",
              abs(direct - stepped) < 0.01 * direct, f"{direct:.1f} vs {stepped:.1f} kg over {n * UNIT_DAYS} d")
    # 2. milk-floor cull: the cull is spread over steps around the crossing, and its mean timing equals the
    #    real-day crossing at every resolution (E[(k+1)*U] == crossing day: the cow leaves at the END of the
    #    step in which she crosses, an offset that is uniform(0, U] -- mean U/2 -- by the entry convention).
    fine = np.arange(LAST_BREEDING_DIM_DAYS, (MAX_DIM + 2) * UNIT_DAYS) + 0.5
    for b in range(1, MAX_LACTATION + 1):
        y = curve_kg(fine, b)
        day = fine[np.where(y <= MILK_FLOOR_KG)[0][0]]
        p = t["cull_p"][b - 1]
        surv, mean_end = 1.0, 0.0
        for k in range(MAX_DIM + 1):
            mean_end += surv * p[k] * (k + 1) * UNIT_DAYS
            surv *= 1 - p[k]
        check(f"milk-floor cull timing matches the real-day crossing (lactation {b})",
              abs(mean_end - day) <= 1.0 and surv < 1e-9, f"mean cull step-end {mean_end:.1f} d vs crossing day {day:.1f}")
        check(f"floor never applied before LAST_BREEDING_DIM_DAYS (lactation {b})",
              (p[: max(LAST_BREEDING_DIM - 1, 0)] == 0).all() and (t["cull_dim"][b - 1] >= LAST_BREEDING_DIM))
    # 3. pregnancy factor
    pf = t["preg_factor"]
    check("preg_factor[:, 0] == 1 (open cows)", (pf[:, 0] == 1.0).all())
    check("preg_factor dry == 0 for every lactation", (pf[:, np.arange(MAX_DP + 1) >= DRY_OFF_DP] == 0).all())
    check("preg_factor non-increasing in dp for every lactation", (np.diff(pf, axis=1) <= 1e-12).all())
    check("dry-off step follows the mid-step rule (first dp whose mid-step age dp*U >= dry-off day)",
          DRY_OFF_DP * UNIT_DAYS >= DRY_OFF_DP_DAYS > (DRY_OFF_DP - 1) * UNIT_DAYS, f"DRY_OFF_DP={DRY_OFF_DP}")
    # 4. dim bound and state_milk
    check("dim_max within MAX_DIM", t["dim_max"] <= MAX_DIM, f"dim_max={t['dim_max']}")
    m = state_milk_kg(np.array([5, 5]), np.array([2, 2]), np.array([0, DRY_OFF_DP]))
    check("state_milk dry == 0 and open > 0", m[1] == 0.0 and m[0] > 0.0)


def check_pipeline():
    from cowmrp.mrp import core as mrp_core, engine as mrp_engine
    from cowmrp.domains.cow.parameters import UNIT_DAYS, MAX_DIM, MAX_DP, MAX_LACTATION, LAST_BREEDING_DIM, LAST_BREEDING_DIM_DAYS
    from cowmrp.domains.cow.states import variable_domains, derived_variables
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.rewards import compute_rewards
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
    from cowmrp.domains.cow.tables import get_tables

    from cowmrp.domains.cow.states import build_cow_states
    states = build_cow_states()
    check("state columns are exactly lactation, dim, dp, state_id",
          list(states.columns) == ['lactation', 'dim', 'dp', 'state_id'], str(list(states.columns)))
    dim_max = get_tables()['dim_max']
    check("state axes follow the reachable bounds (dim <= dim_max, dp <= dp_max, lactation <= MAX_LACTATION)",
          states['dim'].max() == dim_max and states['dp'].max() == get_tables()['dp_max']
          and states['lactation'].max() == MAX_LACTATION, f"{len(states)} states")

    probs = compute_probabilities(states)
    check("probability keys", set(probs) == {'p_dead', 'p_cull', 'p_preg', 'p_abort', 'p_calve', 'p_resume'}, str(sorted(probs)))
    ok = all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() for v in probs.values())
    check("all probabilities finite and in [0,1]", ok)
    dim, dp, lac = states['dim'].values, states['dp'].values, states['lactation'].values
    first_blocked = int(np.ceil(LAST_BREEDING_DIM_DAYS / UNIT_DAYS))   # earliest possible age >= cutoff from here on
    check("p_preg == 0 for pregnant/dry cows and once the earliest age in the step is past the last breeding day",
          (probs['p_preg'][(dp > 0) | (dim >= first_blocked)] <= 1e-12).all())
    cd = get_tables()['cull_dim'][np.minimum(lac, 3) - 1]
    zone = (dp == 0) & (dim >= cd)
    check("p_cull == 1 for every open cull-zone state", (probs['p_cull'][zone] == 1.0).all(), f"{int(zone.sum())} states")
    check("floor is not applied to pregnant cows or before cull_dim",
          (probs['p_cull'][~zone] < 1.0).all())

    rows, cols, vals = compute_transitions(states, probs)
    matrix = mrp_core.assemble_matrix(states, rows, cols, vals)
    mrp_core.validate_row_stochasticity(matrix, states)
    check("every row sums to 1 (row-stochastic)", True, f"{len(states)} states, nnz={matrix.nnz}")
    rid = get_replacement_state_id(states)
    check("replacement state is (lactation 1, dim 0, dp 0)",
          tuple(states.iloc[rid][['lactation', 'dim', 'dp']]) == (1, 0, 0))
    bad = [i for i in np.where(zone)[0] if matrix[i].nnz != 1 or matrix[i].indices[0] != rid]
    check("cull-zone rows go only to the replacement state", not bad, f"{len(bad)} violations")

    rewards = compute_rewards(states, probs)
    check("milk_produced_kg == 0 for dry cows",
          (rewards['milk_produced_kg'][dp >= __import__('cowmrp.domains.cow.parameters', fromlist=['x']).DRY_OFF_DP] == 0).all())
    check("reward total finite", np.isfinite(rewards['total']).all())

    # DIM-bound equivalence: V(replacement) must not depend on the unreachable DIM axis
    gamma = mrp_core.annual_to_per_step_gamma(0.95, UNIT_DAYS)
    V = mrp_core.evaluate_infinite_horizon(matrix, rewards['total'], gamma)
    domains_full = dict(variable_domains)
    domains_full['dim'] = list(range(0, MAX_DIM + 1))
    s2 = mrp_engine.build_states(domains_full, derived_variables)
    p2 = compute_probabilities(s2)
    r2, c2, v2 = compute_transitions(s2, p2)
    m2 = mrp_core.assemble_matrix(s2, r2, c2, v2)
    V2 = mrp_core.evaluate_infinite_horizon(m2, compute_rewards(s2, p2)['total'], gamma)
    # pruning equivalence: the unpruned Cartesian product (same dim_max) gives the same V(replacement)
    full_dm = mrp_engine.build_states(variable_domains, derived_variables)
    pf = compute_probabilities(full_dm)
    rf, cf, vf = compute_transitions(full_dm, pf)
    mf = mrp_core.assemble_matrix(full_dm, rf, cf, vf)
    Vf = mrp_core.evaluate_infinite_horizon(mf, compute_rewards(full_dm, pf)['total'], gamma)
    check("pruning unreachable states is lossless: V(replacement) identical to the unpruned state space",
          abs(V[rid] - Vf[get_replacement_state_id(full_dm)]) < 1e-6 * max(1.0, abs(V[rid])),
          f"{len(states)} reachable-by-rule vs {len(full_dm)} Cartesian states")
    from scipy.sparse.csgraph import breadth_first_order
    reach = len(breadth_first_order(matrix.tocsr(), rid, directed=True, return_predecessors=False))
    t_ = get_tables()
    expected = int(np.where(states['dp'].values > 0, np.isin(states['dim'].values - states['dp'].values, t_['entry_steps']),
                            states['dim'].values <= t_['dim_reachable_max']).sum())
    check("reachable states == pregnant states on a conception step with p > 0 plus open states up to the last re-opening DIM (the rest exists by design)", reach == expected,
          f"{reach} reachable, {expected} expected, {len(states)} states")
    check("V(replacement) identical with the full MAX_DIM axis (DIM bound is lossless)",
          abs(V[rid] - V2[get_replacement_state_id(s2)]) < 1e-6 * max(1.0, abs(V[rid])),
          f"{V[rid]:.4f} vs {V2[get_replacement_state_id(s2)]:.4f}; {len(states)} vs {len(s2)} states")


def check_blocks():
    """Structural 'blocks': transitions that must be IMPOSSIBLE in this biology."""
    from cowmrp.mrp import core as mrp_core, engine as mrp_engine
    from cowmrp.domains.cow.parameters import UNIT_DAYS, MAX_LACTATION, MAX_DP, DRY_OFF_DP, VOLUNTARY_WAITING_PERIOD_DAYS, LAST_BREEDING_DIM, LAST_BREEDING_DIM_DAYS
    from cowmrp.domains.cow.states import variable_domains, derived_variables
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
    from cowmrp.domains.cow.tables import get_tables

    from cowmrp.domains.cow.states import build_cow_states
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = mrp_core.assemble_matrix(states, rows, cols, vals).tocoo()
    r, c = matrix.row, matrix.col
    lac, dim, dp = (states[k].values for k in ('lactation', 'dim', 'dp'))
    rid = get_replacement_state_id(states)
    dim_max = get_tables()['dim_max']
    nd = np.minimum(dim[r] + 1, dim_max)

    to_repl = c == rid
    open_src = dp[r] == 0
    preg_src = ~open_src
    ok_open = open_src & ((lac[c] == lac[r]) & (dim[c] == nd) & ((dp[c] == 0) | (dp[c] == 1)))
    # abort -> open, or abort -> conceives again in the remaining half step (RESUME_BREEDING_AFTER_LOSS: dp = 1)
    ok_abort = preg_src & ((lac[c] == lac[r]) & (dim[c] == nd) & ((dp[c] == 0) | (dp[c] == 1)) & (dp[r] >= 1))
    ok_calve = preg_src & ((lac[c] == lac[r] + 1) & (dim[c] == 0) & (dp[c] == 0) & (lac[r] < MAX_LACTATION))
    ok_remain = preg_src & ((lac[c] == lac[r]) & (dim[c] == nd) & (dp[c] == np.minimum(dp[r] + 1, MAX_DP)))
    allowed = to_repl | ok_open | ok_abort | ok_calve | ok_remain
    check("every transition is one of: replace / conceive / stay open / abort / calve / stay pregnant",
          allowed.all(), f"{int((~allowed).sum())} illegal of {len(r)}")
    check("no transition ever skips a lactation or leaves 1..MAX_LACTATION",
          (lac[c] >= 1).all() and (lac[c] <= MAX_LACTATION).all() and ((lac[c] - lac[r]) <= 1).all())
    check("max-lactation cows that calve are replaced, never advanced",
          not ((lac[r] == MAX_LACTATION) & (lac[c] == MAX_LACTATION + 1)).any())
    conceive = open_src & (dp[c] == 1) & ~to_repl
    check("conception from an OPEN cow always lands on dp = 1", conceive.any() and (dp[c][open_src & (dp[c] > 0) & ~to_repl] == 1).all())
    check("no conception strictly inside the voluntary waiting period (fine resolutions)",
          (probs['p_preg'][(dim + 2) * UNIT_DAYS <= VOLUNTARY_WAITING_PERIOD_DAYS] == 0).all())
    first_blocked = int(np.ceil(LAST_BREEDING_DIM_DAYS / UNIT_DAYS))
    check("no conception once the earliest age in the step is past the last breeding day",
          (dim[r][conceive & (matrix.data > 1e-12)] < first_blocked).all())
    p_calve_early = probs['p_calve'][(dp > 0) & (dp * UNIT_DAYS + UNIT_DAYS < 150)]
    check("no calving long before term (dp*U + U < 150 d)", (p_calve_early == 0).all())
    check("no abortion for open cows", (probs['p_abort'][dp == 0] == 0).all())
    check("re-conception after a loss (p_resume) only for pregnant cows and only while breeding is still possible",
          (probs['p_resume'][(dp == 0) | (dim >= first_blocked)] <= 1e-12).all() and (probs['p_resume'][dp > 0] > 0).any())
    resumed = preg_src & (dp[c] == 1) & ~to_repl & (dp[r] != 0)
    check("a pregnant cow reaches dp = 1 only by losing her pregnancy and conceiving again (never by 'skipping' gestation)",
          (probs['p_abort'][r][resumed & (matrix.data > 1e-12)] > 0).all() and (dim[c][resumed & (matrix.data > 1e-12)] == nd[resumed & (matrix.data > 1e-12)]).all())
    check("dry cows give no milk but still carry mortality/culling rows",
          (probs['p_dead'][dp >= DRY_OFF_DP] > 0).all())


def check_architecture():
    """Table coverage + evidence labels, method structure, and milk-curve shape for lactations 1..MAX_LACTATION."""
    import pandas as pd
    import cowmrp.domains.cow.probabilities, cowmrp.domains.cow.rewards                      # registers every full table
    from cowmrp.mrp.engine import resolve_method
    from cowmrp.domains.cow.parameters import MAX_LACTATION, MILK_CURVE_BASIS, MILK_CURVE_SOURCE_LACTATION
    from cowmrp.domains.cow.lookup_tables import FULL_TABLE_REGISTRY, get_full_table, BASIS_CODES
    from cowmrp.domains.cow.probabilities import probability_specs
    from cowmrp.domains.cow.rewards import reward_specs
    from cowmrp.domains.cow.curves import curve_kg

    for name in FULL_TABLE_REGISTRY:
        df = get_full_table(name)
        check(f"table '{name}' has an explicit row for every lactation 1..{MAX_LACTATION}",
              sorted(df['lactation'].unique()) == list(range(1, MAX_LACTATION + 1)))
        counts = df['basis'].value_counts().to_dict()
        check(f"table '{name}' labels the evidence of every row (source/derived/repeated/placeholder)",
              set(df['basis']) <= set(BASIS_CODES), str(counts))

    # method structure: every spec exposes table/formula/model, the active method resolves, and selecting an
    # unavailable one is a hard error naming what IS available (never a silent fallback)
    for spec in list(probability_specs) + list(reward_specs):
        keys = set(spec['methods'])
        check(f"spec '{spec['name']}' exposes methods {{table, formula|existing_pipeline, model}}",
              keys >= {'table', 'model'} or keys == {'existing_pipeline'} or keys >= {'formula', 'model'}, str(sorted(keys)))
        resolve_method(spec)
    probe = dict(probability_specs[0])
    probe['method'] = 'model'
    try:
        resolve_method(probe)
        check("selecting an unavailable 'model' method raises", False)
    except Exception as e:
        check("selecting an unavailable 'model' method raises, naming the available methods",
              'available' in str(e).lower() or 'table' in str(e), type(e).__name__)

    # lactation evidence: explicit status for every lactation, no pretence of lactation-specific data
    for method in ('wood', 'milkbot'):
        st = MILK_CURVE_BASIS[method]
        check(f"milk curve '{method}': evidence status recorded for lactations 1..{MAX_LACTATION}",
              sorted(st) == list(range(1, MAX_LACTATION + 1)), str({k: v for k, v in st.items() if k in (1, 3, 4, MAX_LACTATION)}))
    check("Wood lactations beyond the source's last reported lactation are labelled 'repeated', not 'source'",
          all(MILK_CURVE_BASIS['wood'][l] == 'repeated' for l in range(4, MAX_LACTATION + 1)))

    # curve shape: rise -> single peak -> decline, positive, plausible peak timing and 305-d yield
    days = np.arange(1, 701) + 0.5
    for method in ('wood', 'milkbot'):
        for lac in range(1, MAX_LACTATION + 1):
            y = curve_kg(days, lac, method=method)
            pk = int(np.argmax(y))
            ok = ((y > 0).all() and (np.diff(y[:pk + 1]) >= -1e-9).all() and (np.diff(y[pk:]) <= 1e-9).all()
                  and 15 <= days[pk] <= 130 and 5000 <= y[:305].sum() <= 16000)
            check(f"{method} curve, lactation {lac}: rise -> peak -> decline, peak day {days[pk]:.0f}, 305-d {y[:305].sum():.0f} kg", ok)


if __name__ == "__main__":
    check_tables()
    check_architecture()
    check_pipeline()
    check_blocks()
    if FAILED:
        print(f"\nVERIFY_FAILED: {FAILED}")
        sys.exit(1)
    print("\nVERIFY_OK")

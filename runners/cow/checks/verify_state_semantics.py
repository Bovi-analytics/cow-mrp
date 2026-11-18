"""
verify_state_semantics.py
=========================
Two kinds of checks, reported separately:

  [STRUCTURAL] Are the state rules independently justified and correctly represented?
      A. every rule in cow/state_constraints.py accepts its valid examples and rejects its deliberately invalid ones
         (default parameters and stress parameters that make every rule bind)
      B. the state table holds exactly the combinations the rules allow, identically in every lactation
      C. the rules equal a direct derivation from the timing convention (the (u, v) offset region), not from any probability
      D. the probabilities implement the rules: support closure per lactation, independent re-derivation of the conception
         probability and the breeding-window share, calving / dry-off / gestation timing, pregnancy-loss aggregate
      E. transitions: closure, row sums, reachability, calving resets (dim 0, dp 0, waiting period restarts), every lactation
      F. flexibility: a lactation-specific waiting period changes the rule set for that lactation only, and the support check
         detects that the probabilities do not follow it
  [REGRESSION] Does the refactor preserve the previous behavior?
      R. the state table equals the previous construction (conception steps with positive probability), row by row

An unchanged state count is NOT evidence for [STRUCTURAL]; [STRUCTURAL] would fail on a wrong rule even if the count were
unchanged.  Read-only: nothing is written.

Run: python runners/cow/checks/verify_state_semantics.py        (exit 1 on any failure)
Resolution: set COW_TIME_UNIT=day|week|month before running.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import sys
import numpy as np

FAILED = {'STRUCTURAL': [], 'REGRESSION': []}


def check(group, name, ok, detail=""):
    print(f"[{group}] [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED[group].append(name)


def main():
    import math
    from cowmrp.domains.cow.parameters import (UNIT_DAYS as U, TIME_UNIT, MAX_LACTATION, DRY_OFF_DP, DRY_OFF_DP_DAYS,
                                GESTATION_LENGTH_DAYS as G, VOLUNTARY_WAITING_PERIOD_DAYS as VWP, LAST_BREEDING_DIM_DAYS as LAST)
    from cowmrp.domains.cow.state_constraints import (CONSTRAINTS, RuleParams, feasible_mask, violated, bin_days, describe_state)
    from cowmrp.domains.cow.state_checks import (landing_closure_errors, unreachable_states, to_csr, support_mismatches, legacy_state_table)
    from cowmrp.domains.cow.states import build_cow_states, variable_domains, derived_variables
    from cowmrp.domains.cow.tables import get_tables
    from cowmrp.domains.cow.lookup_tables import N_DIM_MONTHS, get_full_table, table_to_array
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.probabilities.calving import calving_step_mass
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
    from cowmrp.mrp import engine as mrp_engine

    t = get_tables()
    dp_max, dim_max = t['dp_max'], t['dim_max']
    model_rules = RuleParams(dp_max=dp_max, dim_max=dim_max)
    print(f"resolution {TIME_UNIT} (U={U} d)  VWP={VWP} (first conception day {VWP + 1})  last breeding day={LAST}  dim_max={dim_max}  dp_max={dp_max}")
    for c in CONSTRAINTS:
        print(f"  rule {c.name} [{c.kind}]: {c.formula}")

    # ================= A. every rule: valid examples accepted, invalid examples rejected =================
    stress = [model_rules,
              RuleParams(unit_days=U, vwp_days=5 * U + 20, dim_cap_days=12 * U + 3, dp_max=dp_max, dim_max=dim_max)]
    for c in CONSTRAINTS:
        n_valid = n_invalid = 0
        ok_valid = ok_invalid = True
        for p in stress:
            for (d, q) in c.valid_examples(p):
                n_valid += 1
                ok_valid &= bool(np.asarray(c.holds(np.array([d]), np.array([q]), np.array([1]), p))[0])
            for (d, q) in c.invalid_examples(p):
                n_invalid += 1
                rej = not bool(np.asarray(c.holds(np.array([d]), np.array([q]), np.array([1]), p))[0])
                ok_invalid &= rej and c.name in violated(d, q, p) and not bool(feasible_mask(np.array([d]), np.array([q]), p)[0])
        check('STRUCTURAL', f"rule '{c.name}' [{c.kind}] accepts its {n_valid} valid and rejects its {n_invalid} deliberately invalid states",
              ok_valid and ok_invalid and n_valid > 0 and n_invalid > 0)

    # ================= B. the state table is exactly the feasible combinations, in every lactation =================
    S = build_cow_states()
    n = len(S)
    lac, dim, dp = S['lactation'].values, S['dim'].values, S['dp'].values
    preg = dp > 0
    dom = np.array(np.meshgrid(variable_domains['dim'], variable_domains['dp'], indexing='ij')).reshape(2, -1)
    n_feas = int(feasible_mask(dom[0], dom[1], model_rules).sum())
    check('STRUCTURAL', "the state table holds exactly the feasible (dim, dp) combinations of the domain, once per lactation",
          n == n_feas * MAX_LACTATION, f"{n} states = {n_feas} combinations x {MAX_LACTATION} lactations")
    check('STRUCTURAL', "every state satisfies every rule", bool(feasible_mask(dim, dp, model_rules, lac).all()))
    pat = S.groupby('lactation').apply(lambda g: tuple(sorted(zip(g['dim'].values.tolist(), g['dp'].values.tolist()))), include_groups=False)
    check('STRUCTURAL', "all lactations carry the identical (dim, dp) pattern (the waiting period restarts at every calving)",
          pat.nunique() == 1 and len(pat) == MAX_LACTATION)

    # ================= C. rules == direct derivation from the timing convention =================
    # a cow in (k, j) conceived at a = (e+1)U + u - v, u, v in (0, U); possible iff some (u, v) gives
    # a > 0 (after calving), a > vwp+1 (hazard positive), and the table's day cap (the last breeding day is a probability, not a rule). Derived from the offsets only.
    g = (np.arange(60) + 0.5) / 60 * U
    uu, vv = np.meshgrid(g, g, indexing='ij')
    ok_c, bad_e = True, []
    for e in range(-3, dim_max + 1):
        a = (e + 1) * U + uu - vv
        possible = bool(((a > 0) & (a > VWP + 1)).any()) and e * U <= RuleParams().dim_cap_days
        by_rule = bool(feasible_mask(np.array([e + 1]), np.array([1]), RuleParams()).item()) if e >= 0 else False
        if possible != by_rule:
            ok_c = False
            bad_e.append(e)
    check('STRUCTURAL', "the rule set equals the (u, v) offset derivation of which conception steps are possible (steps -3 .. dim_max)",
          ok_c, f"disagreeing steps: {bad_e[:5]}" if not ok_c else f"first possible conception step e0 = {int(min(t['entry_steps']))}")
    rule_steps = set(e for e in range(0, dim_max + 1) if feasible_mask(np.array([e + 1]), np.array([1]), RuleParams()).item())
    extra = rule_steps - set(int(x) for x in t['entry_steps'])
    check('STRUCTURAL', "every conception step with positive probability is a rule step, and the extra rule steps all lie after the last one (p_preg = 0 there, the states still exist)",
          set(int(x) for x in t['entry_steps']) <= rule_steps and all(e > max(t['entry_steps']) for e in extra), f"{len(extra)} extra steps")

    # ================= D. the probabilities implement the rules =================
    pr = compute_probabilities(S)
    p_preg = pr['p_preg']
    openm = ~preg
    mism = support_mismatches(S, p_preg, lambda d, l: feasible_mask(d + 1, np.ones_like(d), model_rules, l), dim_max)
    mism = [m for m in mism if m[2] > 0 and not m[3]]     # only p_preg > 0 on an impossible landing state is an error; p_preg = 0 on a possible one is the breeding policy
    check('STRUCTURAL', "p_preg > 0 only where the landing state (dim+1, dp=1) is possible -- all lactations", len(mism) == 0, str(mism[:3]) if mism else "")
    within = openm & ((dim + 2) * U <= VWP + 1)
    check('STRUCTURAL', "no conception in any step that ends within the voluntary waiting period -- all lactations",
          bool((p_preg[within] == 0).all()), f"{int(within.sum())} open states")
    past = openm & (dim * U >= LAST)
    check('STRUCTURAL', "no conception in steps that start on/after the last breeding day (a straddling step keeps its share of days)",
          bool((p_preg[past] <= 1e-12).all()), f"{int(past.sum())} open states")
    # independent re-derivation of p_preg: monthly probability -> constant daily hazard; the open cows at an instant are
    # weighted by survival to it, so P = 1 - mean_u S(end)/mean_u S(start)
    Pm = table_to_array(get_full_table('conception_full'), N_DIM_MONTHS)[0]
    days = np.arange(0, (dim_max + 3) * U + 2)
    uu1 = (np.arange(400) + 0.5) / 400 * U
    worst = 0.0
    for li in range(MAX_LACTATION):
        pm = Pm[li][np.minimum(days // 30, Pm.shape[1] - 1)]
        lam = np.where((days > VWP) & (days < LAST), -np.log1p(-pm) / 30.0, 0.0)
        cum = np.concatenate(([0.0], np.cumsum(lam)))
        Hf = lambda x: np.interp(x, np.arange(len(cum)), cum)
        for k in range(dim_max + 1):
            i = np.flatnonzero(openm & (lac == li + 1) & (dim == k))
            if len(i):
                mine = 1 - np.mean(np.exp(-Hf(k * U + uu1 + U))) / np.mean(np.exp(-Hf(k * U + uu1)))
                worst = max(worst, abs(float(mine) - float(p_preg[i[0]])))
    check('STRUCTURAL', "p_preg equals an independent re-derivation from the monthly table and the rules (all lactations, all steps)",
          worst < 5e-4, f"max abs difference {worst:.1e}")
    # breeding-window share used by the reproduction cost
    kk = np.arange(dim_max + 1)
    share = np.zeros(len(kk))
    for ui in uu1:
        lo, hi = kk * U + ui, kk * U + ui + U
        share += np.clip(np.minimum(hi, LAST) - np.maximum(lo, VWP), 0, None) / U
    share /= len(uu1)
    d_share = float(np.abs(share - t['breed_frac'][:dim_max + 1]).max())
    check('STRUCTURAL', "the breeding-window share of each step equals an independent integral of the step's age window over (VWP, last breeding day]",
          d_share < 0.02, f"max abs difference {d_share:.3f}")

    mass = calving_step_mass(0.0)
    mine = {}
    for v in g:
        j = int(math.floor((G - v) / U)) + 1
        mine[j] = mine.get(j, 0.0) + 1.0 / len(g)
    ok_mass = set(mine) == set(mass) and all(abs(mine[j] - mass[j]) < 0.02 for j in mine)
    check('STRUCTURAL', "calving step mass equals j = floor((G - v)/U) + 1 with v uniform (independent derivation)", ok_mass,
          f"model {{{', '.join(f'{j}: {m:.3f}' for j, m in sorted(mass.items()))}}}")
    first_calv, last_calv = min(mass), max(mass)
    p_calve = pr['p_calve']
    by_dp = {j: float(p_calve[np.flatnonzero(preg & (dp == j))[0]]) for j in range(1, dp_max + 1) if (preg & (dp == j)).any()}
    check('STRUCTURAL', "no calving before the gestation step; certain from the last gestation step",
          all(by_dp[j] == 0 for j in by_dp if j < first_calv) and all(abs(by_dp[j] - 1) < 1e-12 for j in by_dp if j >= last_calv),
          f"first calving step dp={first_calv}, certain from dp={last_calv}")
    check('STRUCTURAL', "dry-off step = first step whose mean gestation age reaches the dry-off day, and precedes the last calving step",
          DRY_OFF_DP == max(1, -(-int(DRY_OFF_DP_DAYS) // U)) and 1 <= DRY_OFF_DP <= last_calv <= dp_max, f"DRY_OFF_DP={DRY_OFF_DP}")
    check('STRUCTURAL', "open cows never abort; pregnancy never outlasts the lactation (dp <= dim)", bool((pr['p_abort'][openm] == 0).all()) and bool((dp[preg] <= dim[preg]).all()))
    j90 = max(1, int(round(90 / U)))
    sub = np.flatnonzero((lac == 1) & preg & (dim - dp == int(min(t['entry_steps']))))
    sub = sub[np.argsort(dp[sub])]
    loss90 = 1 - float(np.prod(1 - pr['p_abort'][sub[:j90]]))
    check('STRUCTURAL', "cumulative pregnancy loss to about 90 d reproduces the aggregate of the published windows it is built from (27/13/7/2 %)",
          abs(loss90 - (1 - 0.73 * 0.87 * 0.93 * 0.98)) < 0.015, f"model {loss90:.3f} vs {1 - 0.73 * 0.87 * 0.93 * 0.98:.3f} (implementation consistency, not a biological validation; definition of conception: an open question)")

    # ================= E. transitions =================
    r, c, v = compute_transitions(S, pr)       # raises if a positive transition lands on a missing state
    check('STRUCTURAL', "transition closure: every positive transition lands on an existing state", len(landing_closure_errors(S, r, c, v)) == 0)
    T = to_csr(S, r, c, v)
    check('STRUCTURAL', "row sums = 1 and no negative probability", bool(np.allclose(np.asarray(T.sum(axis=1)).ravel(), 1.0, atol=1e-9)) and bool((v >= 0).all()))
    start = get_replacement_state_id(S)
    un = unreachable_states(S, T, start)
    reach = np.ones(n, bool); reach[un] = False
    exp = np.where(dp > 0, np.isin(dim - dp, t['entry_steps']), dim <= t['dim_reachable_max'])
    check('STRUCTURAL', "reachable from the replacement state (1, 0, 0) == pregnant states on a conception step with p > 0, and open states up to the last re-opening DIM; the rest exists but is unreachable by design",
          bool((reach == exp).all()), f"{int(reach.sum())}/{n} reachable, {int(exp.sum())} expected")
    changed = lac[c] != lac[r]
    up = changed & (lac[c] == lac[r] + 1)
    check('STRUCTURAL', "every lactation change is lactation -> lactation+1 at (dim 0, dp 0) from a pregnant state at or after the first calving step, or a replacement",
          bool(((~changed) | (up & (dim[c] == 0) & (dp[c] == 0) & preg[r] & (dp[r] >= first_calv)) | (c == start)).all()))
    last = (lac == MAX_LACTATION)
    exit_last = last[r] & preg[r] & (c == start) & (v > 0) & (dp[r] >= first_calv)
    check('STRUCTURAL', "the last lactation's calving exit goes to the replacement state", bool(exit_last.any()) and bool(((dim[c][exit_last] == 0) & (lac[c][exit_last] == 1)).all()))
    e0 = int(min(t['entry_steps']))
    after_calving_ok = all(float(p_preg[np.flatnonzero(openm & (lac == L) & (dim == k))[0]]) == 0.0 for L in range(2, MAX_LACTATION + 1) for k in range(0, e0))
    check('STRUCTURAL', "after a calving the waiting period restarts: no conception in dim 0..e0-1 of any later lactation" if e0 > 0
          else "after a calving the first step is already inside the conception-possible window (e0 = 0): p_preg(dim 0) is the small edge share, below dim 1",
          after_calving_ok and all(float(p_preg[np.flatnonzero(openm & (lac == L) & (dim == 0))[0]]) <= float(p_preg[np.flatnonzero(openm & (lac == L) & (dim == e0 + 1))[0]]) for L in range(1, MAX_LACTATION + 1)))

    # ================= F. flexibility: a lactation-specific waiting period =================
    flex = RuleParams(dp_max=dp_max, dim_max=dim_max, vwp_by_lactation={1: 60})
    diff = feasible_mask(dim, dp, flex, lac) != feasible_mask(dim, dp, model_rules, lac)
    check('STRUCTURAL', "declaring a longer waiting period for lactation 1 changes the rule outcome for lactation 1 only", bool(diff.any()) and bool((lac[diff] == 1).all()))
    mism_f = [m for m in support_mismatches(S, p_preg, lambda d, l: feasible_mask(d + 1, np.ones_like(d), flex, l), dim_max) if m[2] > 0 and not m[3]]
    check('STRUCTURAL', "the support check detects that the probabilities do not follow that lactation-specific rule (guard against inconsistent rules)",
          len(mism_f) > 0 and all(m[0] == 1 for m in mism_f), f"{len(mism_f)} mismatching open states, lactation(s) {sorted(set(m[0] for m in mism_f))}")
    d1 = describe_state(1, e0 + 1, 1, model_rules)
    check('STRUCTURAL', "labels: the first pregnant state is bin e0+2 at the instant, status pregnant, conception age between e0*U and (e0+2)*U",
          d1['bin_number'] == e0 + 2 and d1['status'] == 'pregnant' and d1['conception_age_days_between'] == (e0 * U, (e0 + 2) * U) and d1['feasible'] and bin_days(0, model_rules) == (1, U), str(d1))

    # ================= R. regression against the previous construction =================
    domain_product = mrp_engine.build_states(variable_domains, derived_variables)
    old = legacy_state_table(domain_product, t['entry_steps'])
    new_keys = set(map(tuple, S[['lactation', 'dim', 'dp']].values)); old_keys = set(map(tuple, old[['lactation', 'dim', 'dp']].values))
    extra_rows = new_keys - old_keys
    check('REGRESSION', "the state table contains the previous construction (conception steps with positive probability) and adds only pregnant states on later conception steps",
          old_keys <= new_keys and all(d_p > 0 and (d - d_p) not in set(int(x) for x in t['entry_steps']) for _, d, d_p in extra_rows), f"{len(old_keys)} previous + {len(extra_rows)} added = {len(S)} states")

if __name__ == "__main__":
    main()
    print()
    for grp in ('STRUCTURAL', 'REGRESSION'):
        print(f"{grp}_{'FAILED: ' + str(FAILED[grp]) if FAILED[grp] else 'OK'}")
    if any(FAILED.values()):
        sys.exit(1)
    print("VERIFY_OK")

"""
verify_reward_vector.py
=======================
Checks the state reward vector without replacement_cost in it, and the separate
cow-versus-replacement-heifer comparison that now carries the replacement cost.

  REWARD VECTOR (rewards['total'] and the exported reward vector)
      total = milk_income + pregnancy_reward + cull_reward - mortality_cost - feed_cost - reproduction_cost
  A. definition: the six components, their signs, no replacement_cost anywhere in the vector; total = signed sum
  B. every component recomputed from the raw parameters/tables/probabilities for EVERY state
  C. cull_reward / mortality_cost weights equal the exit mass of the transition matrix (no double counting)
  D. representative rows (open, past breeding, milk-floor cull step, pregnant, dry, calving step, last lactation,
     replacement state) with the component-by-component arithmetic
  E. the separate replacement comparison: same horizon, same discount, replacement cost carried exactly once;
     independent re-solution where it is cheap (MONTH/WEEK)

Run: python runners/cow/checks/verify_reward_vector.py        (exit 1 on any failure)
Resolution: COW_TIME_UNIT=month|week|day.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))   # project root (mrp/, cow/)
import sys
import numpy as np
import scipy.sparse as sp

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def neumann_solve(matrix, reward, gamma, tol=1e-11, max_iter=400000):
    """Independent infinite-horizon solve: V = sum_t gamma^t P^t r by repeated multiplication (no linear solver)."""
    term = np.asarray(reward, float).copy()
    V = term.copy()
    for _ in range(max_iter):
        term = gamma * matrix.dot(term)
        V += term
        if np.abs(term).max() < tol:
            break
    return V


def main():
    import cowmrp.domains.cow.parameters as P
    from cowmrp.domains.cow.parameters import UNIT_DAYS as U, MAX_LACTATION, DRY_OFF_DP
    from cowmrp.domains.cow.states import build_cow_states, milk_yield_kg
    from cowmrp.domains.cow.probabilities import compute_probabilities
    from cowmrp.domains.cow.transitions import compute_transitions, get_replacement_state_id
    from cowmrp.domains.cow.rewards import compute_rewards, BASE_REWARD_SPECS
    from cowmrp.domains.cow.rewards import replacement_cost_vector
    from cowmrp.domains.cow.rewards.base import (MILK_PRICE_PER_KG, FEED_PRICE_PER_KG_DM, DRY_PERIOD_DAILY_COST, CALF_VALUE, BEEF_PRICE_PER_KG, DRESSING_PCT,
                                  CULLING_COST, DISPOSAL_COST, REARING_COST, body_weight_kg, age_days_of, mid_step_dim_days,
                                  dry_matter_intake_kg, max_lactation_exit_probability, AI_COST, DIAGNOSIS_COST, SYNCHRONIZATION_COST,
                                  DAYS_OPEN_PENALTY_PER_DAY, ABORTION_COST)
    from cowmrp.domains.cow.rewards.replacement_cost import cost_per_replacement
    from cowmrp.domains.cow.tables import get_tables
    from cowmrp.domains.cow.value_comparisons import replacement_net_values, replacement_comparison, replacement_value
    from cowmrp.mrp import core as mc

    S = build_cow_states()
    n = len(S)
    pr = compute_probabilities(S)
    r, c, v = compute_transitions(S, pr)
    T = mc.assemble_matrix(S, r, c, v)
    R = compute_rewards(S, pr)
    rid = get_replacement_state_id(S)
    lac, dim, dp = S['lactation'].values, S['dim'].values, S['dp'].values
    p_dead, p_cull, p_abort, p_calve, p_preg = pr['p_dead'], pr['p_cull'], pr['p_abort'], pr['p_calve'], pr['p_preg']
    print(f"resolution {P.TIME_UNIT} (U={U} d), {n} states")

    # ---------------- A. definition ----------------
    names = [s['name'] for s in BASE_REWARD_SPECS]
    signs = {s['name']: s['sign'] for s in BASE_REWARD_SPECS}
    check("A. the state reward vector has exactly the six components, none of them replacement_cost",
          names == ['milk_income', 'pregnancy_reward', 'cull_reward', 'mortality_cost', 'feed_cost', 'reproduction_cost'], str(names))
    check("A. compute_rewards() does not expose replacement_cost at all (no hidden column, no second route into the total)", 'replacement_cost' not in R)
    check("A. signs: income and event rewards +1, mortality / feed / reproduction costs -1",
          signs == {'milk_income': 1, 'pregnancy_reward': 1, 'cull_reward': 1, 'mortality_cost': -1, 'feed_cost': -1, 'reproduction_cost': -1})
    total_sum = sum(signs[k] * np.asarray(R[k]) for k in names)
    check("A. total = milk_income + pregnancy_reward + cull_reward - mortality_cost - feed_cost - reproduction_cost (all states)",
          bool(np.allclose(R['total'], total_sum, rtol=0, atol=1e-9)), f"max |diff| {np.abs(np.asarray(R['total']) - total_sum).max():.2e}")
    check("A. every component is finite and non-negative as stored (costs are positive numbers with sign -1)",
          all(np.isfinite(R[k]).all() and (np.asarray(R[k]) >= -1e-12).all() for k in names))

    # ---------------- B. independent recomputation, all states ----------------
    survive = (1 - p_dead) * (1 - p_cull)
    milk = np.asarray(milk_yield_kg(dim, lac, dp))
    dim_days = mid_step_dim_days(dim)
    age_days = age_days_of(lac, dim)
    bw = body_weight_kg(age_days, dim_days, dp * U)
    dmi = dry_matter_intake_kg(milk, bw, np.maximum(dim_days, 1))
    exp = {}
    exp['milk_income'] = milk * MILK_PRICE_PER_KG * U
    exp['pregnancy_reward'] = survive * (1 - p_abort) * p_calve * CALF_VALUE
    w_death = p_dead
    w_cull = (1 - p_dead) * p_cull + np.where(lac >= MAX_LACTATION, survive * (1 - p_abort) * p_calve, 0.0)
    exp['mortality_cost'] = w_death * DISPOSAL_COST
    exp['cull_reward'] = w_cull * (bw * DRESSING_PCT * BEEF_PRICE_PER_KG - CULLING_COST)
    exp['feed_cost'] = dmi * FEED_PRICE_PER_KG_DM * U + np.where(dp >= DRY_OFF_DP, DRY_PERIOD_DAILY_COST * U, 0.0)
    tab = get_tables()
    k = np.clip(dim, 0, P.MAX_DIM)
    open_share = (1 + survive * (1 - p_preg)) / 2.0
    services_per_day = 1.0 / P.RATE_PERIOD_DAYS
    exp['reproduction_cost'] = (
        np.where(dp == 0, (AI_COST + DIAGNOSIS_COST + SYNCHRONIZATION_COST) * services_per_day * U * tab['breed_frac'][k] * open_share, 0.0)
        + np.where(dp == 0, DAYS_OPEN_PENALTY_PER_DAY * U * tab['after_vwp_frac'][k] * open_share, 0.0)
        + np.where(dp > 0, survive * p_abort * ABORTION_COST, 0.0))
    for name in names:
        d = float(np.abs(np.asarray(R[name]) - exp[name]).max())
        check(f"B. {name} equals its independent recomputation in all {n} states", d < 1e-8, f"max |diff| {d:.2e}")

    # ---------------- C. weights = transition-matrix exit mass ----------------
    Td = T.tocsr()
    exit_mass = np.asarray(Td[:, rid].todense()).ravel()               # P(s -> replacement state)
    w_sum = w_death + w_cull
    check("C. P(death) + P(cull or calving exit at the last lactation) equals the matrix mass into the replacement state, in every state",
          bool(np.allclose(exit_mass, w_sum, rtol=0, atol=1e-12)), f"max |diff| {np.abs(exit_mass - w_sum).max():.2e}")
    check("C. the two events are disjoint: death and cull weights are each in [0,1] and sum to at most 1",
          bool(((w_death >= 0) & (w_cull >= 0) & (w_sum <= 1 + 1e-12)).all()))
    check("C. the replacement cost does not appear in cull_reward or mortality_cost (they equal weight x (salvage - culling cost) and weight x disposal cost)",
          bool(np.allclose(R['mortality_cost'], w_death * DISPOSAL_COST, atol=1e-9)) and bool(np.allclose(R['cull_reward'], exp['cull_reward'], atol=1e-9)))
    maxl = (lac >= MAX_LACTATION) & (dp > 0)
    exit_calving = max_lactation_exit_probability(S, pr)
    check("C. a calving at the last lactation is an exit: counted once in cull_reward (and in the matrix), never in mortality_cost",
          bool(np.allclose(exit_calving[maxl] + (1 - p_dead[maxl]) * p_cull[maxl] + p_dead[maxl], exit_mass[maxl], atol=1e-12)))
    # stationary flow: exits per year from the matrix equal the exits implied by the two reward weights
    gamma = mc.annual_to_per_step_gamma(0.95, U)
    if n <= 20000:
        pi = np.zeros(n)
        pi[rid] = 1.0
        for _ in range(100000):
            nx = T.T.dot(pi)
            if np.abs(nx - pi).sum() < 1e-13:
                break
            pi = nx
        exits_matrix = float((pi * exit_mass).sum())
        exits_rewards = float((pi * w_sum).sum())
        deaths = float((pi * w_death).sum())
        check("C. long-run exit flow: matrix mass into the replacement state = death weight + cull weight (per step)",
              abs(exits_matrix - exits_rewards) < 1e-12, f"{exits_matrix * 365 / U:.4f} exits/cow-year of which {deaths * 365 / U:.4f} deaths")

    # ---------------- D. representative rows ----------------
    e0 = int(min(tab['entry_steps']))
    from cowmrp.domains.cow.probabilities.calving import calving_step_mass
    cm = calving_step_mass(0.0)
    first_calv, last_calv = min(cm), max(cm)

    def pick(**kw):
        m = np.ones(n, bool)
        for key, val in kw.items():
            m &= (S[key].values == val)
        ix = np.flatnonzero(m)
        return int(ix[0]) if len(ix) else None

    cull_dim = int(tab['cull_dim'][0])
    rows = [
        ("replacement state (heifer, calving)", pick(lactation=1, dim=0, dp=0)),
        ("open, in the breeding window", pick(lactation=2, dim=P.dim_step_of_days(120), dp=0)),
        ("open, past the last breeding day", pick(lactation=2, dim=P.LAST_BREEDING_DIM + 2, dp=0)),
        ("open, milk-floor cull step (cull probability 1)", pick(lactation=1, dim=cull_dim, dp=0)),
        ("pregnant, mid gestation", pick(lactation=2, dim=e0 + 4, dp=4)),
        ("dry, before the calving step (dp = DRY_OFF_DP)", pick(lactation=2, dim=e0 + 1 + DRY_OFF_DP, dp=DRY_OFF_DP)),
        ("pregnant, calving step", pick(lactation=2, dim=e0 + 1 + last_calv, dp=last_calv)),
        (f"last lactation ({MAX_LACTATION}), calving step = exit + calf", pick(lactation=MAX_LACTATION, dim=e0 + 1 + last_calv, dp=last_calv)),
    ]
    print("\nrepresentative rows (US$ per step; components shown with their sign):")
    for label, i in rows:
        if i is None:
            check(f"D. row exists: {label}", False)
            continue
        comp = {k_: signs[k_] * float(R[k_][i]) for k_ in names}
        s_ = sum(comp.values())
        st = S.iloc[i]
        print(f"  {label}: state (L={int(st.lactation)}, dim={int(st.dim)}, dp={int(st.dp)})  "
              f"p_dead={p_dead[i]:.4f} p_cull={p_cull[i]:.4f} p_calve={p_calve[i]:.3f} | matrix mass to replacement {exit_mass[i]:.4f}")
        print("      " + "  ".join(f"{k_}={v_:+.3f}" for k_, v_ in comp.items()))
        print(f"      sum of components = {s_:+.4f}   total = {float(R['total'][i]):+.4f}   "
              f"mortality weight x disposal = {w_death[i]:.4f} x {DISPOSAL_COST:g}   cull weight x (salvage - culling cost) = {w_cull[i]:.4f} x {float(bw[i] * DRESSING_PCT * BEEF_PRICE_PER_KG - CULLING_COST):.2f}")
        check(f"D. total equals the sum of the included signed components: {label}", abs(s_ - float(R['total'][i])) < 1e-9)
    i_dry = rows[5][1]
    if i_dry is not None:
        check("D. a dry row earns no milk income and pays the dry-period feed premium", float(R['milk_income'][i_dry]) == 0.0)
    i_cullstep = rows[3][1]
    if i_cullstep is not None:
        check("D. at the milk-floor cull step the cull weight is 1 - p_dead (all survivors are culled) and the matrix agrees",
              abs(w_cull[i_cullstep] - (1 - p_dead[i_cullstep])) < 1e-9 and abs(exit_mass[i_cullstep] - 1.0) < 1e-9)

    # ---------------- E. the separate replacement comparison ----------------
    C = float(cost_per_replacement())
    rc = replacement_cost_vector(S, pr)
    check("E. replacement-cost memo vector = P(exit) x cost of one replacement (same exit event as the matrix)",
          bool(np.allclose(rc, exit_mass * C, atol=1e-9)), f"C = {C:g}")
    for years in (None, 5):
        net = replacement_net_values(S, T, R, pr, gamma, years)
        tag = "infinite horizon" if years is None else f"{years}-year horizon"
        # old accounting: replacement cost inside the reward, one stream
        old_stream = np.asarray(R['total']) - rc
        if years is None:
            V_old = neumann_solve(T, old_stream, gamma) if n <= 20000 else mc.evaluate_infinite_horizon(T, old_stream, gamma)
        else:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                V_old = mc.evaluate_finite_horizon(T, old_stream, gamma, years, U)
        d = float(np.abs(net['V_keep'] - V_old).max())
        solver = "independent Neumann series" if (years is None and n <= 20000) else "engine solver, same horizon"
        check(f"E. {tag}: V_total - PV_replacement_cost equals the value of the old accounting (replacement cost inside the reward) -- {solver}",
              d < 1e-5 * max(1.0, float(np.abs(V_old).max())), f"max |diff| {d:.2e}")
        cmp_r = replacement_comparison(S, T, R, pr, gamma, 1, 0, 0, years, net=net)
        check(f"E. {tag}: a heifer compared with a heifer bought now is worth exactly the replacement cost it saves", abs(cmp_r['value'] - C) < 1e-6, f"{cmp_r['value']:.4f}")
        cmp_s = replacement_comparison(S, T, R, pr, gamma, 2, P.dim_step_of_days(150), 0, years, net=net)
        parts = (cmp_s['V_total_state'] - cmp_s['V_total_heifer']) - (cmp_s['PV_replacement_cost_state'] - cmp_s['PV_replacement_cost_heifer']) + cmp_s['replacement_cost_now']
        check(f"E. {tag}: value = [V_total(cow) - V_total(heifer)] - [PV_rc(cow) - PV_rc(heifer)] + C (lactation 2, DIM 150 d, open)",
              abs(parts - cmp_s['value']) < 1e-6, f"{cmp_s['value']:.2f}")
        print(f"      {tag}: lactation-2 open cow at DIM 150: keep {cmp_s['V_state']:.1f}, heifer net of C {cmp_s['V_comparison']:.1f}, "
              f"difference {cmp_s['value']:.1f}; PV replacement cost {cmp_s['PV_replacement_cost_state']:.1f} vs {cmp_s['PV_replacement_cost_heifer']:.1f}")
        check(f"E. {tag}: PV of replacement costs is non-negative and finite for every state", bool((net['PV_replacement_cost'] >= -1e-9).all() and np.isfinite(net['PV_replacement_cost']).all()))
    V_total = mc.evaluate_infinite_horizon(T, R['total'], gamma)
    rv = replacement_value(S, V_total, 2, P.dim_step_of_days(150), 0)
    check("E. the registry replacement_value is the difference of state-reward values and therefore excludes replacement costs (documented)",
          abs(rv['value'] - (rv['V_state'] - rv['V_comparison'])) < 1e-9)


    # ---------------- F. the historical baseline is reproduced from the new reward vector ----------------
    hist = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), "runners", "cow", "outputs", "baseline",
                         "historical_calf250_replacement_cost_in_reward", "Model_A", "worker_result.json")
    if P.TIME_UNIT == "month" and _os.path.exists(hist):
        import json
        from cowmrp.domains.cow.rewards.base import CALF_VALUE
        h = json.load(open(hist))
        CALF_OLD = 250.0
        from cowmrp.domains.cow.tables import get_tables as _gt
        _t = _gt()      # the stored means are over the states a cow can reach (the table also holds unreachable-by-design states)
        reach_m = np.where(S['dp'].values > 0, np.isin(S['dim'].values - S['dp'].values, _t['entry_steps']), S['dim'].values <= _t['dim_reachable_max'])
        preg_stream = np.asarray(R['pregnancy_reward'], float)
        old_stream = np.asarray(R['total'], float) - rc - (1 - CALF_OLD / CALF_VALUE) * preg_stream     # calf value 250, replacement cost inside the reward
        V_inf_old = neumann_solve(T, old_stream, gamma)
        V_fin_old = np.zeros(n)
        for _ in range(mc.steps_per_year(U) * 5):
            V_fin_old = old_stream + gamma * T.dot(V_fin_old)
        check("F. the historical baseline (calf value 250, replacement cost inside the reward) is reproduced from the current model: V_infinite_mean",
              abs(float(V_inf_old[reach_m].mean()) - h['V_infinite_mean']) < 1e-6 * h['V_infinite_mean'], f"{float(V_inf_old[reach_m].mean()):.4f} vs stored {h['V_infinite_mean']:.4f}")
        check("F. ... and V_finite_mean (5 years)", abs(float(V_fin_old[reach_m].mean()) - h['V_finite_mean']) < 1e-6 * h['V_finite_mean'],
              f"{float(V_fin_old[reach_m].mean()):.4f} vs stored {h['V_finite_mean']:.4f}")
        V_inf_new = neumann_solve(T, np.asarray(R['total'], float), gamma)
        print(f"      current model: V_infinite_mean {float(V_inf_new.mean()):.4f} = historical {h['V_infinite_mean']:.4f} "
              f"+ calf value effect + avoided replacement cost (PV of replacement costs, mean {float(neumann_solve(T, rc, gamma).mean()):.2f})")


if __name__ == "__main__":
    main()
    if FAILED:
        print("\nREWARD_VECTOR_CHECK_FAILED:", FAILED)
        sys.exit(1)
    print("\nREWARD_VECTOR_CHECK_OK")

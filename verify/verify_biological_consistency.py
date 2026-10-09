"""
verify_biological_consistency.py
================================
Reachability and biological-timing audit of the ACTIVE model at MONTH, WEEK and DAY (one process per resolution).

    python verify/verify_biological_consistency.py          -> prints facts + checks, writes outputs/audit/reachability.json

For each resolution it traces the executed states and transitions (not the exported tables) and establishes:
  * which statuses exist and over which pregnancy-day (DP) steps; the dry period actually modelled (days between dry-off and calving)
  * which ROWS of every lookup table can ever be read by the active resolvers (a populated table row is not necessarily a possible state)
  * impossible-combination checks for every status-dependent mechanism:
      no pregnancy loss, calving, resumed breeding or dry-off for an OPEN cow; no conception for a pregnant cow; conception only inside the
      waiting-period / last-breeding window; the milk-floor rule only for open cows past day 300; milk reduction timing; dry cows give no milk;
      calving only around the gestation length and with certainty by the last pregnant step; the state after calving is (lactation + 1, DIM 0, DP 0);
      a dry cow can only calve, lose the pregnancy, die, be culled, or advance (never skip to open without an event)
Table rows are 'used' when their calendar ages overlap the true ages a reachable step can cover: DIM step k covers ages [kU, (k+2)U]
(entry offset u ~ U(0,U)); pregnant / dry step j covers [(j-1)U, (j+1)U].
"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
UNITS = ("month", "week", "day")
failed = []


def ok(cond, msg):
    print(("[OK] " if cond else "[FAIL] ") + msg)
    if not cond:
        failed.append(msg)


def months_of(lo, hi):
    import numpy as np
    return set(int(m) for m in np.unique(np.arange(max(int(lo), 0), int(hi)) // 30))


def one_unit():
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    import cow.parameters as P
    from mrp import core as mc
    from cow.states import build_cow_states, repro_status
    from cow.probabilities import compute_probabilities
    from cow.transitions import compute_transitions, get_replacement_state_id
    from cow.tables import get_tables
    from cow.curves import curve_kg
    from cow.rewards import compute_rewards
    U = P.UNIT_DAYS
    t = get_tables()
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    M = mc.assemble_matrix(states, rows, cols, vals).tocsr()
    rid = get_replacement_state_id(states)
    lac, dim, dp = (states[k].values for k in ("lactation", "dim", "dp"))
    status = np.array(repro_status(dp))
    out = {"U": U, "dry_off_dp": int(P.DRY_OFF_DP), "dp_max": int(t["dp_max"]), "dry_off_days": P.DRY_OFF_DP_DAYS, "gestation": P.GESTATION_LENGTH_DAYS, "checks": {}}
    chk = out["checks"]

    # ---------- statuses and DP ranges
    for s in ("Open", "Pregnant", "Dry"):
        m = status == s
        out[f"n_{s}"] = int(m.sum())
        out[f"dp_{s}"] = [int(dp[m].min()), int(dp[m].max())]
    out["dry_dp_age_days"] = [int(P.DRY_OFF_DP * U), int(t["dp_max"] * U)]          # mean age during the first and last dry step
    chk["no_state_beyond_dp_max"] = bool(dp.max() <= t["dp_max"])

    # ---------- impossible-combination checks
    op = dp == 0
    pg = ~op
    chk["abortion_zero_for_open_cows"] = bool((probs["p_abort"][op] == 0).all())
    chk["resumed_breeding_zero_for_open_cows"] = bool((probs["p_resume"][op] == 0).all())
    chk["calving_zero_for_open_cows"] = bool((probs["p_calve"][op] == 0).all())
    chk["conception_zero_for_pregnant_cows"] = bool((probs["p_preg"][pg] == 0).all())
    chk["abortion_positive_for_some_pregnant_cows"] = bool((probs["p_abort"][pg] > 0).any())
    # abortion is applied by the transitions only to pregnant states: the Open rows of the transition matrix carry no loss branch
    # (checked below with the target-state test)
    # calving only around the gestation length
    G = P.GESTATION_LENGTH_DAYS
    pc = probs["p_calve"]
    early = pg & (pc > 0) & ((dp - 1) * U >= G)
    late_ok = pg & (dp * U + U <= G)
    chk["calving_only_when_step_can_contain_gestation_length"] = bool((~((pc > 0) & late_ok)).all() and not early.any())
    chk["calving_certain_at_last_pregnant_step"] = bool((pc[dp == t["dp_max"]] == 1).all())
    first_pos = int(dp[pg & (pc > 0)].min())
    out["first_calving_step"] = first_pos
    out["first_calving_age_interval_days"] = [int((first_pos - 1) * U), int((first_pos + 1) * U)]
    # conception window
    VWP, LB = P.VOLUNTARY_WAITING_PERIOD_DAYS, P.LAST_BREEDING_DIM_DAYS
    pp = probs["p_preg"]
    outside = op & (((dim + 2) * U <= VWP) | (dim * U >= LB))
    inside = op & ((dim + 1) * U > VWP) & (dim * U < LB - U)
    chk["conception_zero_outside_waiting_period_and_last_breeding_day"] = bool((pp[outside] == 0).all())
    chk["conception_positive_inside_window"] = bool((pp[inside] > 0).all())
    # milk floor
    pcr = probs["p_cull"]
    from cow.probabilities.culling import p_cull_repro
    cr = p_cull_repro(states, {})
    chk["milk_floor_zero_for_pregnant_and_dry_cows"] = bool((cr[pg] == 0).all())
    cday = np.array(t["cull_day"])
    indep = []
    for L in range(1, P.MAX_LACTATION + 1):
        d = np.arange(LB, 1500) + 0.5
        y = curve_kg(d, L)
        indep.append(float(d[np.where(y <= P.MILK_FLOOR_KG)[0][0]]))
    chk["milk_floor_crossing_day_equals_first_day_after_300_at_or_below_22.7"] = bool(np.allclose(cday, indep))
    out["milk_floor_crossing_days"] = [float(x) for x in cday[:3]]
    chk["milk_floor_zero_before_day_300"] = bool((cr[op & ((dim + 2) * U <= LB)] == 0).all())
    chk["milk_floor_certain_from_last_step"] = bool(all(t["cull_p"][i][t["cull_dim"][i]:].min() == 1 for i in range(P.MAX_LACTATION)))
    # pregnancy milk factor
    pf = np.array(t["preg_factor"])
    chk["milk_multiplier_is_1_for_open_cows"] = bool((pf[:, 0] == 1).all())
    chk["milk_multiplier_zero_once_dry"] = bool((pf[:, P.DRY_OFF_DP:] == 0).all())
    chk["milk_multiplier_non_increasing_while_milked"] = bool((np.diff(pf[0, :P.DRY_OFF_DP]) <= 1e-12).all())
    out["milk_multiplier_first_pregnant_step"] = float(pf[0, 1])
    out["milk_multiplier_last_milked_step"] = float(pf[0, P.DRY_OFF_DP - 1])
    rew = compute_rewards(states, probs)
    mk = np.asarray(rew["milk_income"])
    chk["milk_income_zero_for_dry_cows"] = bool((np.abs(mk[status == "Dry"]) < 1e-12).all())
    chk["milk_income_positive_for_milked_cows"] = bool((mk[status != "Dry"] > 0).all())

    # ---------- transitions: target-state rules
    r, c, v = rows, cols, vals
    r, c, v = np.asarray(r), np.asarray(c), np.asarray(v)
    sl, sd, sp_ = lac[r], dim[r], dp[r]
    tl, td, tp = lac[c], dim[c], dp[c]
    to_rep = c == rid
    live = ~to_rep
    src_open = sp_ == 0
    # Open cows: only to dp=1 (conception) or dp=0 (stay open), always dim+1 (capped), same lactation
    ok_open = ((tp == 0) | (tp == 1)) & (tl == sl)
    chk["open_cows_only_conceive_or_stay_open"] = bool(ok_open[src_open & live].all())
    # pregnant / dry cows: calving -> (lac+1, dim 0, dp 0); loss -> (same lac, dp 0, or dp 1 if re-conceived); remain -> dp+1 (capped)
    calve = live & ~src_open & (td == 0)
    chk["after_calving_state_is_next_lactation_dim0_dp0"] = bool(((tl[calve] == sl[calve] + 1) & (td[calve] == 0) & (tp[calve] == 0)).all()) and bool(calve.any())
    not_calve = live & ~src_open & ~calve
    ok_p = (tl == sl) & ((tp == 0) | (tp == 1) | (tp == np.minimum(sp_ + 1, t["dp_max"])))
    chk["pregnant_cows_only_lose_pregnancy_resume_or_advance_one_step"] = bool(ok_p[not_calve].all())
    dry_src = status[r] == "Dry"
    chk["dry_cows_can_only_calve_lose_pregnancy_advance_die_or_be_culled"] = bool(((td == 0) & (tp == 0) & (tl == sl + 1) | ok_p | to_rep)[dry_src].all())
    # only dry or late-pregnant cows calve; dry-off precedes calving
    chk["calving_transitions_only_from_steps_that_can_contain_gestation_length"] = bool((sp_[calve] >= first_pos).all())
    # max lactation: a cow calving in the last lactation is replaced
    mx = ~src_open & (lac[r] == P.MAX_LACTATION) & to_rep & (probs["p_calve"][r] > 0)
    chk["calving_at_maximum_lactation_exits_to_replacement"] = bool(mx.any())

    # ---------- dry period actually modelled and calving age (chain from a replacement heifer)
    n = len(states)
    Q = M.copy().tolil(); Q[:, rid] = 0; Q = Q.tocsr()
    e = np.zeros(n); e[rid] = 1.0
    x = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)
    surv = (1 - probs["p_dead"]) * (1 - probs["p_cull"])
    dry_days = float(x[status == "Dry"].sum() * U)
    entry_flow = float((x[(dp == P.DRY_OFF_DP - 1)] * surv[dp == P.DRY_OFF_DP - 1] * (1 - probs["p_abort"][dp == P.DRY_OFF_DP - 1]) * (1 - pc[dp == P.DRY_OFF_DP - 1])).sum())
    out["dry_entries_per_cycle"] = entry_flow
    out["dry_days_per_cycle"] = dry_days
    out["dry_days_per_dry_entry"] = dry_days / entry_flow
    calv_flow = x * surv * (1 - probs["p_abort"]) * pc
    out["mean_gestation_booked_at_calving_days"] = float((calv_flow * dp * U).sum() / calv_flow.sum() + U / 2.0)   # step j booked at its END: mean age j*U (+U/2 entry offset)
    # a dry cow is never left in the dry state past the last calving step: probability mass beyond dp_max is zero by construction
    # (dp domain stops at dp_max and p_calve = 1 there), so every dry cow leaves within dp_max - DRY_OFF_DP + 1 steps
    out["max_dry_steps_before_forced_calving"] = int(t["dp_max"] - P.DRY_OFF_DP + 1)

    # ---------- table rows the active resolvers can read
    used = {"mortality": {}, "other_reason_culling": {}, "conception": {"Open": set()}, "pregnancy_loss": {"Pregnant": set()}, "pregnancy_milk_factor": {"Pregnant": set()}}
    for s in ("Open", "Pregnant", "Dry"):
        mset = set()
        if s == "Open":
            for k in np.unique(dim[status == s]):
                mset |= months_of(k * U, (k + 2) * U)
        else:
            for j in np.unique(dp[status == s]):
                mset |= months_of((j - 1) * U, (j + 1) * U)
        used["mortality"][s] = used["other_reason_culling"][s] = sorted(mset)
    # windowed mechanisms: the day rules zero the hazard outside the window, so only the months that overlap the window are read
    for k in np.unique(dim[op & (pp > 0)]):
        used["conception"]["Open"] |= months_of(max(k * U, VWP), min((k + 2) * U, LB))
    for j in np.unique(dp[pg & (probs["p_abort"] > 0)]):
        used["pregnancy_loss"]["Pregnant"] |= months_of(max((j - 1) * U, 0), min((j + 1) * U, 271))       # the loss hazard is 0 from day 271
    for j in np.unique(dp[(status == "Pregnant")]):
        used["pregnancy_milk_factor"]["Pregnant"] |= months_of(max((j - 1) * U, 0), min((j + 1) * U, P.DRY_OFF_DP_DAYS))   # dry-off applied by day: factor 0 from 222 d
    out["used_months"] = {m: {s: sorted(v) for s, v in d.items()} for m, d in used.items()}
    print("RESULT" + json.dumps(out))


def main():
    res = {}
    for u in UNITS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--unit"], env=dict(os.environ, COW_TIME_UNIT=u), cwd=ROOT, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if p.returncode or not line:
            print(p.stderr[-2500:]); sys.exit(1)
        res[u] = json.loads(line[0][6:])
    for u in UNITS:
        r = res[u]
        print(f"\n=== {u.upper()} (U = {r['U']} d) ===")
        print(f"  dry-off at DP step {r['dry_off_dp']} (mean age during the step {r['dry_dp_age_days'][0]} d; target {r['dry_off_days']} d); last pregnant step {r['dp_max']} (mean age {r['dry_dp_age_days'][1]} d)")
        print(f"  states: Open {r['n_Open']}, Pregnant {r['n_Pregnant']} (DP steps {r['dp_Pregnant']}), Dry {r['n_Dry']} (DP steps {r['dp_Dry']})")
        print(f"  first calving step {r['first_calving_step']} (true ages {r['first_calving_age_interval_days']} d); mean gestation booked at calving {r['mean_gestation_booked_at_calving_days']:.1f} d (true {r['gestation']} d)")
        print(f"  dry period actually modelled: {r['dry_days_per_dry_entry']:.1f} days per dry entry (target 55); at most {r['max_dry_steps_before_forced_calving']} dry steps before calving is certain")
        print(f"  milk multiplier: first pregnant step {r['milk_multiplier_first_pregnant_step']:.3f}, last milked step {r['milk_multiplier_last_milked_step']:.3f}, then 0 (dry); milk-floor crossing days (L1-3) {r['milk_floor_crossing_days']}")
        print("  table months read by the active resolvers (mortality / other culling):", {s: (min(v), max(v)) for s, v in r["used_months"]["mortality"].items()}, "| table has months 0-36 for every status")
        print("  conception months:", r["used_months"]["conception"]["Open"][:1] + r["used_months"]["conception"]["Open"][-1:], "| pregnancy loss months:", (min(r["used_months"]["pregnancy_loss"]["Pregnant"]), max(r["used_months"]["pregnancy_loss"]["Pregnant"])),
              "| pregnancy milk factor months:", (min(r["used_months"]["pregnancy_milk_factor"]["Pregnant"]), max(r["used_months"]["pregnancy_milk_factor"]["Pregnant"])))
        for k, v in r["checks"].items():
            ok(v, f"{u}: {k}")
        ok(abs(r["dry_days_per_dry_entry"] - 55) <= 1.5 * r["U"] + 5, f"{u}: modelled dry period ({r['dry_days_per_dry_entry']:.1f} d) is within 55 d +/- 1.5 steps of event booking")
        ok(abs(r["mean_gestation_booked_at_calving_days"] - r["gestation"]) <= r["U"] + 1, f"{u}: calving booked within one step of the {r['gestation']}-day gestation length ({r['mean_gestation_booked_at_calving_days']:.1f} d)")
    # union of used months over resolutions
    union = {}
    for m in res["day"]["used_months"]:
        union[m] = {}
        for s in res["day"]["used_months"][m]:
            union[m][s] = {u: res[u]["used_months"][m][s] for u in UNITS}
    os.makedirs(os.path.join(ROOT, "outputs", "audit"), exist_ok=True)
    json.dump({"used_months": union, "per_unit": {u: {k: v for k, v in res[u].items() if k != "used_months"} for u in UNITS}}, open(os.path.join(ROOT, "outputs", "audit", "reachability.json"), "w"), indent=1)
    print("\nBIOLOGICAL_CONSISTENCY_OK" if not failed else f"\nBIOLOGICAL_CONSISTENCY_FAILED: {len(failed)} check(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if "--unit" in sys.argv:
        one_unit()
    else:
        main()

"""
_plot_data.py
=============
Collects, from the ACTUAL model at ONE time resolution, every array the Plotly validation figures need. Run once per
resolution in its own process (cow.parameters reads COW_TIME_UNIT at import):

    COW_TIME_UNIT=week python verify/_plot_data.py     -> prints "RESULT<json>"

Nothing here re-implements model logic: probabilities come from the registered tables / specs, solutions from the MRP.
Used by verify/plot_validation_plotly.py.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import cow.parameters as P
from mrp import core as mrp_core
from mrp.engine import resolve_method
from cow.states import build_cow_states, repro_status
from cow.probabilities import compute_probabilities
from cow.rewards import compute_rewards
from cow.transitions import compute_transitions, get_replacement_state_id
from cow.value_comparisons import pregnancy_value, replacement_value
from cow.tables import get_tables
from cow.curves import curve_kg
from cow.lookup_tables import get_full_table, table_to_array, N_DIM_MONTHS, N_GEST_MONTHS, STATUSES
import cow.probabilities.mortality as M
import cow.probabilities.culling as C
import cow.probabilities.conception as K
import cow.probabilities.abortion as A
import cow.probabilities.calving as V
from cow.rewards.milk_income import milk_income_formula
from cow.rewards.feed_cost import feed_cost_formula
from cow.rewards.reproduction_cost import reproduction_cost_formula

L10 = list(range(P.MAX_LACTATION))


def collect():
    U = P.UNIT_DAYS
    t = get_tables()
    d = {"unit": P.TIME_UNIT, "U": U, "MAX_DIM": P.MAX_DIM, "dim_max": int(t["dim_max"]), "dp_max": int(t["dp_max"]),
         "dry_off_dp": P.DRY_OFF_DP, "dry_off_dp_days": P.DRY_OFF_DP_DAYS, "gestation_days": P.GESTATION_LENGTH_DAYS,
         "vwp": P.VOLUNTARY_WAITING_PERIOD_DAYS, "last_breeding_days": P.LAST_BREEDING_DIM_DAYS, "floor": P.MILK_FLOOR_KG,
         "max_lactation": P.MAX_LACTATION, "milk_curve_method": P.MILK_CURVE_METHOD}
    d["milk_step_mean"] = t["milk_curve"].tolist()
    d["preg_factor_step"] = t["preg_factor"][0].tolist()
    d["cull_p"] = t["cull_p"].tolist()
    d["cull_day"] = t["cull_day"].tolist()
    # native monthly tables (all lactations) and the model's per-step arrays
    d["native"] = {
        "mortality": table_to_array(get_full_table("mortality_full"), N_DIM_MONTHS, STATUSES).tolist(),
        "culling": table_to_array(get_full_table("culling_age_stage_full"), N_DIM_MONTHS, STATUSES).tolist(),
        "conception": table_to_array(get_full_table("conception_full"), N_DIM_MONTHS)[0].tolist(),
        "abortion": table_to_array(get_full_table("abortion_full"), N_GEST_MONTHS)[0].tolist(),
        "pregnancy_factor": table_to_array(get_full_table("pregnancy_milk_factor_full"), N_GEST_MONTHS, value_col="factor")[0].tolist(),
    }
    d["basis"] = {n: get_full_table(n).groupby("lactation")["basis"].agg(lambda s: sorted(set(s))).to_dict() for n in
                  ("mortality_full", "culling_age_stage_full", "conception_full", "abortion_full")}
    d["step"] = {
        "mortality": [a.tolist() for a in M._mortality_step_arrays()],
        "culling": [a.tolist() for a in C._culling_age_stage_step_arrays()],
        "conception": K._conception_step_arrays().tolist(),
    }
    dpf = pd.DataFrame({"lactation": 1, "dim": 0, "dp": np.arange(0, t["dp_max"] + 1)})
    d["step"]["abortion"] = np.asarray(resolve_method(A.abortion_spec)(dpf, {})).tolist()
    d["calving_mass"] = {str(k): v for k, v in V.calving_step_mass().items()}
    days = np.arange(0, 701, 5, dtype=float) + 0.5
    d["curve_days"] = days.tolist()
    d["curve"] = {m: [curve_kg(days, l, method=m).tolist() for l in range(1, P.MAX_LACTATION + 1)] for m in ("wood", "milkbot")}
    d["m305"] = {m: [float(curve_kg(np.arange(0, 305) + 0.5, l, method=m).sum()) for l in range(1, P.MAX_LACTATION + 1)] for m in ("wood", "milkbot")}
    d["milk_basis"] = P.MILK_CURVE_BASIS

    # full model
    states = build_cow_states()
    probs = compute_probabilities(states)
    rows, cols, vals = compute_transitions(states, probs)
    matrix = mrp_core.assemble_matrix(states, rows, cols, vals)
    rewards = compute_rewards(states, probs)
    gamma = mrp_core.annual_to_per_step_gamma(0.95, U)
    Vv = mrp_core.evaluate_infinite_horizon(matrix, rewards["total"], gamma)
    rid = get_replacement_state_id(states)
    n = len(states)
    M_ = matrix.tocsr()
    Q = M_.copy().tolil()
    Q[:, rid] = 0
    Q = Q.tocsr()
    e = np.zeros(n)
    e[rid] = 1.0
    xv = spl.spsolve((sp.eye(n, format="csc") - Q.tocsc()).T.tocsc(), e)       # expected visits per replacement cycle
    lac, dim, dp = (states[k].values for k in ("lactation", "dim", "dp"))
    surv = (1 - probs["p_dead"]) * (1 - probs["p_cull"])
    d["summary"] = {"V": float(Vv[rid]), "cycle_days": float(xv.sum() * U), "n_states": int(n),
                    "calvings_per_cycle": float((xv * surv * (1 - probs["p_abort"]) * probs["p_calve"]).sum())}

    # exits per lactation (same decomposition as verify/audit_exits.py)
    pr = C.p_cull_repro(states, {})
    pa = C.p_culling_age_stage_full_table(states, {})
    fd, fa, fr = xv * probs["p_dead"], xv * (1 - probs["p_dead"]) * pa * (1 - pr), xv * (1 - probs["p_dead"]) * pr
    fm = xv * surv * (lac >= P.MAX_LACTATION) * (1 - probs["p_abort"]) * probs["p_calve"]
    ex = []
    for L in range(1, P.MAX_LACTATION + 1):
        m = lac == L
        entry = float(xv[(lac == L) & (dim == 0) & (dp == 0)].sum())
        ex.append({"L": L, "entry": entry, "cow_years": float(xv[m].sum() * U / 365.0),
                   "death": float(fd[m].sum() / entry) if entry > 1e-12 else 0.0, "other": float(fa[m].sum() / entry) if entry > 1e-12 else 0.0,
                   "milk_floor": float(fr[m].sum() / entry) if entry > 1e-12 else 0.0, "max_lac": float(fm[m].sum() / entry) if entry > 1e-12 else 0.0})
    d["exits"] = ex

    # net pregnancy by DIM per lactation (a cow entering the lactation, still in the herd): share pregnant / dry among cows alive in it
    Q2 = M_.copy().tolil()
    Q2[:, rid] = 0
    Q2 = Q2.tocsr()
    pregdim = {}
    for L in range(1, P.MAX_LACTATION + 1):
        sel = np.where(lac == L)[0]
        ML = Q2[sel][:, sel].tocsr()
        i0 = np.where((dim[sel] == 0) & (dp[sel] == 0))[0]
        if len(i0) == 0:
            continue
        v = np.zeros(len(sel)); v[i0[0]] = 1.0
        xs, ps, alive = [], [], []
        for step in range(P.MAX_DIM + 1):
            a = v.sum()
            if a < 1e-6:
                break
            xs.append((step + 1) * U)
            ps.append(float(v[dp[sel] > 0].sum() / a))
            alive.append(float(a))
            v = ML.T @ v
        pregdim[str(L)] = {"dim_days": xs, "preg_share": ps, "alive": alive}
    d["pregnant_by_dim"] = pregdim

    # state occupancy (steady-state herd, all cows)
    status = np.array(repro_status(dp))
    xn = xv / xv.sum()

    def density(mask, width=30):
        xs = np.bincount(dim[mask], weights=xn[mask], minlength=int(dim.max()) + 1)
        dens = np.zeros(1500)
        for k, w in enumerate(xs):
            if w:
                a0 = int(round((k + 0.5) * U)); dens[a0:a0 + U] += w / U
        return [float(dens[b * width:(b + 1) * width].sum()) for b in range(0, 1500 // width)]
    d["occ"] = {"all": density(np.ones(n, bool)), "by_status": {s: density(status == s) for s in ("Open", "Pregnant", "Dry")},
                "by_lactation": [float(xn[lac == l].sum()) for l in range(1, P.MAX_LACTATION + 1)]}

    # cohort survival
    pi = np.zeros(n); pi[rid] = 1.0
    horizon = int(round(9 * 365 / U)); stride = max(1, horizon // 120)
    st, sv = [], []
    for step in range(horizon + 1):
        if step % stride == 0:
            st.append(step * U / 365.0); sv.append(float(pi.sum()))
        pi = pi @ Q
    d["cycle_survival"] = {"years": st, "alive": sv}

    # values at matched conception age (~90 d: mean age during the conception step is (c+1)U)
    c0 = max(0, int(round(90.0 / U)) - 1)
    pvx, pvy = [], []
    for dpv in range(1, t["dp_max"] + 1):
        try:
            pvy.append(float(pregnancy_value(states, Vv, lactation=2, dim=c0 + dpv, dp=dpv)["value"])); pvx.append((dpv - 1) * U + U / 2)
        except ValueError:
            pass
    d["pv"] = {"gest_days": pvx, "value": pvy, "conceived_at_dim_days": (c0 + 1) * U}
    rvx, rvy = [], []
    for k in range(0, min(int(t["cull_dim"][1]), t["dim_max"])):
        rvx.append((k + 1) * U); rvy.append(float(replacement_value(states, Vv, lactation=2, dim=k, dp=0)["value"]))
    d["rv"] = {"dim_days": rvx, "value": rvy}

    # rewards per day (open cows, per cow at the START of the step), lactations 1-3
    d["reward_day"] = {}
    for l in (1, 2, 3):
        sel = (lac == l) & (dp == 0)
        sub = states[sel]
        prs = {k: v[sel] for k, v in probs.items()}
        d["reward_day"][str(l)] = {"dim_days": ((sub["dim"].values + 1) * U).tolist(),
                                   "milk_income": (milk_income_formula(sub, prs) / U).tolist(),
                                   "feed_cost": (-feed_cost_formula(sub, prs) / U).tolist(),
                                   "reproduction_cost": (-reproduction_cost_formula(sub, prs) / U).tolist()}
    print("RESULT" + json.dumps(d))


if __name__ == "__main__":
    collect()

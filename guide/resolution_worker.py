"""guide/resolution_worker.py -- one model build at the time resolution set in COW_TIME_UNIT; prints one JSON line.

Used by guide/sync.py to record, from the real model, what each resolution (month, week, day) looks like: the number of states and transitions, the
step length, where a reference cow sits in the state space, and the reward each step of an open second-lactation cow earns. Nothing here changes the model.
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

REF = dict(lactation=2, dim_days=150, dp_days=60)      # the reference cow of the examples: second lactation, 150 days in milk, 60 days pregnant


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        from cowmrp.domains.cow.parameters import TIME_UNIT, UNIT_DAYS, dim_step_of_days, dp_step_of_days
        from cowmrp.model import build_model
        m = build_model("cow")
        y = build_model("youngstock")
    s = m.states
    open2 = s[(s["lactation"] == REF["lactation"]) & (s["dp"] == 0)].sort_values("dim")
    ids = open2["state_id"].to_numpy()
    out = {
        "time_unit": TIME_UNIT, "unit_days": int(UNIT_DAYS), "states": int(len(s)), "transitions": int(m.P.nnz),
        "ys_states": int(len(y.states)), "ys_transitions": int(y.P.nnz),
        "reference": {"dim_step": int(dim_step_of_days(REF["dim_days"])), "dp_step": int(dp_step_of_days(REF["dp_days"]))},
        "curve": {"dim_step": [int(v) for v in open2["dim"]],
                  **{k: [round(float(v), 6) for v in m.rewards[k][ids]] for k in ("milk_income", "feed_cost", "total")}},
    }
    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

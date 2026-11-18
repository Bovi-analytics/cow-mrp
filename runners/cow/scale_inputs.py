"""
runners/cow/scale_inputs.py
Adapter for runners/_shared/scale_benchmark.py: the cow as a value subject. Entity = one cow, start states drawn from the
equilibrium (closed) herd; discount = the cow convention (annual gamma -> per step); rewards = the cow reward vector `total`.
"""
RESOLUTIONS = ("month", "week", "day")
NOTES = ("Entity: one cow. Start states are drawn from the equilibrium herd (stationary distribution of the closed herd). "
         "Value = discounted reward over the horizon (cow convention, annual discount 0.95 by default).")


def inputs(args):
    from cowmrp.domains.cow.parameters import UNIT_DAYS
    from cowmrp.domains.cow.rewards import compute_rewards
    from cowmrp.domains.cow.transitions import get_replacement_state_id
    from runners.cow.farm.build import build_engine
    from runners.cow.farm.generate import steady_state
    states, probs, matrix = build_engine()
    rewards = compute_rewards(states, probs)["total"]
    rid = get_replacement_state_id(states)
    return dict(kind="value", states=states, matrix=matrix, reward=rewards, unit_days=UNIT_DAYS,
                start_distribution=steady_state(states, matrix, rid))

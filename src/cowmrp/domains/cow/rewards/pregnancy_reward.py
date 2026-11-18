"""cow/rewards/pregnancy_reward.py - calf value at calving.

Event-based: paid when the cow actually CALVES, i.e. with the probability of the calving transition in
cow/transitions.py: she must survive death and culling in the step and not abort, then calve
(survive * (1 - p_abort) * p_calve). Using bare p_calve would pay the calf even for cows that
die/are culled/abort in the same step -- a mismatch with the transition the MRP actually takes."""
import numpy as np
from cowmrp.domains.cow.rewards.base import CALF_VALUE


def pregnancy_reward_formula(states, probs):
    survive = (1 - probs['p_dead']) * (1 - probs['p_cull'])
    return survive * (1 - probs['p_abort']) * probs['p_calve'] * CALF_VALUE


pregnancy_reward_spec = {
    'name': 'pregnancy_reward', 'sign': +1,
    'depends_on': {
        'state': [], 'derived_state': [],
        'probability_output': ['p_dead', 'p_cull', 'p_abort', 'p_calve'],
        'reward_output': [], 'parameter': [], 'external_input': [], 'model_output': [], 'time': [],
    },
    'method': 'formula',
    'methods': {'table': None, 'formula': pregnancy_reward_formula, 'model': None},
}

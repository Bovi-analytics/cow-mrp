"""Environmental outputs are available independently of economic valuation (cowmrp.domains.cow.rewards.environment)."""
import numpy as np
import pytest


def test_physical_quantities_exist_and_are_finite_without_any_price(cow):
    _, m = cow
    from cowmrp.domains.cow.rewards.environment import compute_environment
    env = compute_environment(m.states, m.probs)
    for key in ("bw_kg", "milk_yield_kg", "dmi_kg", "nitrogen_intake_g", "enteric_ch4_g", "respiratory_co2_kg"):
        assert key in env and np.isfinite(np.asarray(env[key], float)).all(), key
    assert (np.asarray(env["enteric_ch4_g"], float) >= 0).all()


def test_total_reward_is_the_economic_sum_with_environmental_pricing_off(cow):
    """Default: the monetary environmental rewards are off, so rewards['total'] is exactly the documented economic vector."""
    _, m = cow
    r = m.rewards
    econ = (r["milk_income"] + r["pregnancy_reward"] + r["cull_reward"] - r["mortality_cost"] - r["feed_cost"] - r["reproduction_cost"])
    assert np.allclose(r["total"], econ, rtol=1e-12, atol=1e-9)
    assert np.allclose(r["reward_environment_total"], 0)

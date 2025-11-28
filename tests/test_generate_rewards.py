import numpy as np
from ABM.generate_rewards import generate_rewards

def test_bounds():
    """
    Test the output value of the rewards.
    The reward values should be in bounds,
      bound[0] <= reward <= bound[1], where
      bound[0] is the lowest bound,
      reward is the reward value, and
      bound[1] is the hightest bound.
    """
    n_trials = 256
    init_vals=[4, 6]
    drift_rate=2
    bounds = [0, 9]

    rewards = generate_rewards(n_trials, bounds, init_vals, drift_rate)

    assert np.all(rewards >= min(bounds))
    assert np.all(rewards <= max(bounds))

def test_n_trial():
    """
    The the number of rewards, representing the number of trials.
    The number should be equal to the number of trials,
      rewards.shape[0] == n_trials, where
      rewards.shape[0] is the number of reward for one end state, and
      n_trials is the number of trials.
    """
    n_trials = 256
    bounds = [0, 9]
    init_vals = [4, 6]
    drift_rate = 2

    rewards = generate_rewards(n_trials, bounds, init_vals, drift_rate)
    assert rewards.shape[0] == n_trials

def test_init_vals():
    """
    Test the initial value of the rewards.
    The value should be equal to those set,
      set(rewards[0]) == set(init_vals), where
      set(rewards[0]) is the set of all values in the first trial, and
      set(init_vals) is the set of initial values provided.
    """
    n_trials = 256
    bounds = [0, 9]
    init_vals = [4, 6]
    drift_rate = 2

    rewards = generate_rewards(n_trials, bounds, init_vals, drift_rate)
    assert set(rewards[0]) == set(init_vals)

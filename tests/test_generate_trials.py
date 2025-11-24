import numpy as np
import pandas as pd
import pytest
from ABM.generate_trials import generate_trials

def test_state1_distribution():
    """
    Test the distribution of first state.
    The distribution should be equally reparted,
      len(df_trial[df_trial.state1 == state]) == n_trials / n_state1, where
      df_trial[df_trial.state1 == state] is the number of trial of a certain state, and
      n_trials / n_state1 is the intended repartition of first trial.
    """
    n_trials = 20
    n_state1 = 4
    n_state2 = 2
    bounds=[0, 9]
    init_vals=[4, 6]
    drift_rate=2
    stakes_value = [1, 5]

    df_trial = generate_trials(
        n_trials, n_state1, n_state2, bounds, init_vals, drift_rate, stakes_value
    )

    for state in range(n_state1):
        assert len(df_trial[df_trial.state1 == state]) == n_trials / n_state1

def test_stake_distribution():
    """
    Test the distribution of stake.
    The distribution should be equally reparted,
      len(df_trial[df_trial.stake == stake]) == n_trials / len(stakes_value), where
      len(df_trial[df_trial.stake == stake]) is the number of stake of a certain value, and
      n_trials / len(stakes_value) is the intended repartition stake.
    """
    n_trials = 20
    n_state1 = 4
    n_state2 = 2
    bounds=[0, 9]
    init_vals=[4, 6]
    drift_rate=2
    stakes_value = [1, 5]

    df_trial = generate_trials(
        n_trials, n_state1, n_state2, bounds, init_vals, drift_rate, stakes_value
    )

    for stake in stakes_value:
        assert len(df_trial[df_trial.stake == stake]) == n_trials / len(stakes_value)

def test_n_trial():
    """
    The the number of rewards, representing the number of trials.
    The number should be equal to the number of trials,
      rewards.shape[0] == n_trials, where
      rewards.shape[0] is the number of reward for one end state, and
      n_trials is the number of trials.
    """
    n_trials = 20
    n_state1 = 4
    n_state2 = 2
    bounds=[0, 9]
    init_vals=[4, 6]
    drift_rate=2
    stakes_value = [1, 5]

    df_trial = generate_trials(
        n_trials, n_state1, n_state2, bounds, init_vals, drift_rate, stakes_value
    )

    assert df_trial.shape[0] == n_trials

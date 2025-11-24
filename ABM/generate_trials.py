"""
Utility module providing a single function to create a list of trials based on the two-step task
For a reference for the task, see:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005090
"""

import pandas as pd
import numpy as np
from .generate_rewards import generate_rewards

def generate_trials(n_trials, n_state1, n_state2, bounds, init_vals, drift_rate, stakes_value):
    """
    Generate trials with randomized states, stakes, and drifting rewards.

    Parameters:
        n_trials (int): Number of trials to generate.
        n_state1 (int): Number of state1 (pair of gnomes).
        n_state2 (int): Number of state2 (forest).
        bounds (tuple): Lower and upper bounds for reward.
        init_vals (list): Initial reward values for each state.
        drift_rate (float): Standard deviation of Gaussian noise added per trial.

    Returns:
        pd.DataFrame: DataFrame containing trial information and reward.
    """
    # Initialize DataFrame and arrays for trial components
    df_trials = pd.DataFrame(columns=['n_trial', 'state1', 'stake'])

    # Generate drifting reward
    rewards = generate_rewards(n_trials, bounds, init_vals, drift_rate)

    # Populate DataFrame with trial data
    df_trials['n_trial'] = np.arange(1, n_trials + 1)
    states = np.tile(np.arange(0, n_state1), int(n_trials/n_state1))
    np.random.shuffle(states)
    df_trials['state1'] = states
    stakes = np.tile(stakes_value, int(n_trials/len(stakes_value)))
    np.random.shuffle(stakes)
    df_trials['stake'] = stakes

    # Add rewards for each state
    for state in range(n_state2):
        df_trials[f'rew_{state + 1}'] = rewards[:, state]

    return df_trials

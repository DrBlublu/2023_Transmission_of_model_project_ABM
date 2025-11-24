"""
Utility module providing a single function to create a distribution of rewards based on the two-step task
For a reference for the task, see:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005090
"""

import random
import numpy as np

def generate_rewards(n_trials, bounds, init_vals, drift_rate):
    """
    Generate drifting reward over trials with reflecting bounds.

    Parameters:
        n_trials (int): Number of trials.
        bounds (tuple): Lower and upper bounds for reward.
        init_vals (list): Initial values for each state.
        drift_rate (float): Standard deviation of Gaussian noise added per trial.

    Returns:
        np.ndarray: Array of shape (n_trials, 2) with reward probabilities.
    """
    # Pre-allocate reward probabilities with neutral starting value
    reward_probs = np.ones((n_trials, 2)) * 0.5
    # Initialize first trial with distinct values for each state
    reward_probs[0, :] = random.sample(init_vals, len(init_vals))
    # Apply Gaussian drift across trials
    for trial in range(1, n_trials):
        for state in range(2):
            # Add Gaussian noise
            update = np.random.normal(0, drift_rate)
            reward = reward_probs[trial - 1, state] + update

            # Apply reflecting bounds
            reward = min(reward, max(bounds[1] * 2 - reward, bounds[0]))
            reward = max(reward, min(bounds[0] * 2 - reward, bounds[1]))
            reward_probs[trial, state] = round(reward)

    return reward_probs

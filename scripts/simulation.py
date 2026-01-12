"""
a
"""

from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd
from ABM.generate_trials import generate_trials
from ABM.run_single_trial import run_single_trial
from ABM.init_transition_function import init_transition_function


n_sim = 100
n_trials = 256
n_state1 = 4
n_state2 = 2
bounds = [0, 9]
init_vals = [4, 6]
drift_rate = 2

# Parameters are based on participant parameter estimations (see parameter_extraction.ipynb)
parameters = {
    "alpha": 0.2,
    "alpha_performance": 0.4,
    "beta": 2.1,
    "trace": 0.4,
    "n0": 6.6,
    "n1": 0
}

# Precompute transition functions and weights
l_fixed_w = np.arange(0.0, 1.01, 0.01)
l_transition_function = [init_transition_function(w) for w in l_fixed_w]


def run_simulation(sim_id):
    """
    Run a single simulation of combination of transition function and fixed w.
    """
    df_trial = generate_trials(n_trials, n_state1, n_state2, bounds, init_vals, drift_rate)
    rewards_array = df_trial[["rew_1", "rew_2"]].to_numpy()
    states_array = df_trial["state1"].to_numpy()
    baseline = rewards_array.mean()
    stakes_array = df_trial["stake"].to_numpy()

    sim_results = []
    for transition_function, fixed_w in itertools.product(l_transition_function, l_fixed_w):
        # Reset qvalues and model_perf here
        qvalues = {
            "qmf1": np.zeros((4,2)) + 4.5,
            "qmf2": np.full(2, 4.5),
            "qmb": np.full(2, 4.5),
        }
        model_perf = {
            "based": np.zeros(4) + 4.5,
            "free": np.zeros(4) + 4.5
        }

        trial_rewards = []
        for state1, rewards in zip(states_array, rewards_array):
            reward, qvalues, model_perf = run_single_trial(
                state1, tuple(rewards), parameters, transition_function, qvalues, model_perf, fixed_w
            )
            trial_rewards.append(reward)

        avg_reward = np.mean(trial_rewards)
        perf = avg_reward - baseline
        model_knowledge = (transition_function[:, 0, 0] + transition_function[:, 1, 1]).mean() / 2

        avg_reward_stake = np.mean(trial_rewards*stakes_array)
        baseline_stake = (rewards_array * stakes_array[:, None]).mean()
        perf_stake = avg_reward_stake - baseline_stake

        sim_results.append({
            "sim": sim_id,
            "model_knowledge": model_knowledge,
            "w": fixed_w,
            "perf": perf,
            "perf_stake": perf_stake,
            "baseline_reward": baseline,
            "avg_reward": avg_reward,
            "stake": stakes_array
        })
    return sim_results

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        all_results = pool.map(run_simulation, range(n_sim))

    results = [item for sublist in all_results for item in sublist]

    df = pd.DataFrame(results, columns=["sim", "model_knowledge", "w", "perf", "perf_stake", "baseline_reward", "avg_reward", "stake"])
    df.to_csv("data/simulation_results.csv", index=False)

"""
a
"""

import numpy as np
import pandas as pd
from tqdm import trange
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

parameters = {
    "alpha": 0.2,
    "alpha_performance": 0.3,
    "beta": 1.0,
    "trace": 0.4,
    "n0": 6,
    "n1": 0.2
}

qvalues = {
    "qmf1": np.zeros((4,2)) + 4.5,
    "qmf2": np.full(2, 4.5),
    "qmb": np.full(2, 4.5),
}

model_perf = {
    "based": np.zeros(4) + 4.5,
    "free": np.zeros(4) + 4.5
}

l_transition_function = []
l_fixed_w = []

for i in np.arange(0.0, 1.0, 0.01):
    l_transition_function.append(init_transition_function(i))
    l_fixed_w.append(i)

data = []

for sim in trange(n_sim):
    df_trial = generate_trials(n_trials, n_state1, n_state2, bounds, init_vals, drift_rate)
    for transition_function in l_transition_function:
        for fixed_w in l_fixed_w:
            total_reward = []
            for row in df_trial.itertuples(index=True):
                rewards = (row.rew_1, row.rew_2)
                state1 = row.state1
                reward, qvalues, model_perf = run_single_trial(
                    state1, rewards, parameters, transition_function, qvalues, model_perf, fixed_w
                )
                total_reward.append(reward)

            avg_reward = np.mean(total_reward)
            baseline = df_trial[["rew_1", "rew_2"]].to_numpy().mean()

            perf = avg_reward - baseline
            model_knowledge = (transition_function[:, 0, 0] + transition_function[:, 1, 1]).mean() / 2 * 100

            data.append({
                "sim": sim,
                "model_knowledge": model_knowledge,
                "w": fixed_w,
                "perf": perf
            })

df = pd.DataFrame(data)
df.to_csv("data/results.csv", index=False)

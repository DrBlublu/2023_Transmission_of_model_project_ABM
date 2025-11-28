"""
a
"""

import numpy as np
import pandas as pd
from tqdm import trange
from ABM.generate_trials import generate_trials
from ABM.run_single_trial import run_single_trial

n_sim = 1000
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

transition_function = np.zeros((4, 2, 2)) + 0.5


fixed_w = 0.5


data = []

for sim in trange(n_sim):

    data_sim = []
    df_trial = generate_trials(n_trials, n_state1, n_state2, bounds, init_vals, drift_rate)

    for row in df_trial.itertuples(index=True):
        rewards = (row.rew_1, row.rew_2)
        state1 = row.state1
        reward, qvalues, model_perf = run_single_trial(state1, rewards, parameters, transition_function, qvalues, model_perf, fixed_w)
        data_sim.append(reward)

    data_sim = np.array(data_sim)
    perf = np.mean(data_sim) - np.mean(df_trial[["rew_1", "rew_2"]])
    data.append(perf)

data = np.array(data)
print(np.mean(data))

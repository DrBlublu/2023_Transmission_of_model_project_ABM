# TO DO:

    # transmission
        # parent value + gaussian noise
    # selection algorithm
        # tournament based selection with k = 4
    # learning
        # indirected gaussian noise


# DONE:

    # create population
        # gaussian for model knowledge & w
            # based on participants data from gen 1 (both language & observaiton)
        # n = 100
    # get score
        # import data
        # get score from data


import numpy as np
import pandas as pd
from ABM.get_perf import get_perf
from ABM.init_population import init_population


perf_matrix = pd.read_csv("data/perf_matrix.csv")

param_learning = {
    "p_model_knowledge": 0.01,
    "sd_model_knowledge": 0.2,
    "p_w": 0.01,
    "sd_w": 0.2
}

noise_std = 0.2 # sd of the gaussian

model_knowledge_param = {"mean": 0.04, "sd": 0.15}
w_param = {"mean": 0.5, "sd": 0.14}
pop = init_population(100, model_knowledge_param, w_param)



agent_model_knowledge = 0.5
agent_w = 0.4
perf = get_perf(perf_matrix, agent_model_knowledge, agent_w)


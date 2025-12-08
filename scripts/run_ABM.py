"""
Docstring for scripts.run_ABM
"""

import numpy as np
import pandas as pd
from ABM.init_population import init_population
from ABM.get_perf import get_perf
from ABM.tournament_selection import tournament_selection


perf_matrix = pd.read_csv("data/perf_matrix.csv")

# param_learning = {
#     "p_model_knowledge": 0.01,
#     "sd_model_knowledge": 0.2,
#     "p_w": 0.01,
#     "sd_w": 0.2
# }

# transmission_noise_std = 0.2 # sd of the gaussian


# pop = init_population(100, model_knowledge_param, w_param)



# agent_model_knowledge = 0.5
# agent_w = 0.4
# perf = get_perf(perf_matrix, agent_model_knowledge, agent_w)

n_gen = 3
k = 4
model_knowledge_param = {"mean": 0.04, "sd": 0.15}
w_param = {"mean": 0.5, "sd": 0.14}

learning_params = {"p_model_knowledge": 1.0, "sd_model_knowledge": 0.1, "p_w": 1.0, "sd_w": 0.1}

for gen in range(n_gen):
    if gen == 0:
        pop = init_population(100, model_knowledge_param, w_param)
    perfs = []
    for agent in pop.iterrows():
        perfs.append(get_perf(perf_matrix, agent["model_knowledge"], agent["w"]))
    pop["perf"] = perfs
    children_model_knowledge, children_w, parent_id = tournament_selection(pop, k)
    
    # a = [f"g{gen}n{i}" for i in range(n)]
    # pass

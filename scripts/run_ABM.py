"""
Docstring for scripts.run_ABM
"""

import numpy as np
import pandas as pd
from ABM.init_population import init_population
from ABM.get_perf import get_perf
from ABM.tournament_selection import tournament_selection
from ABM.create_new_gen import create_new_gen
from ABM.parent_transmission import parent_transmission
from ABM.learning import learning


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

N_GEN = 10
N = 100
K = 4
model_knowledge_param = {"mean": 0.04, "sd": 0.15}
w_param = {"mean": 0.5, "sd": 0.14}
TRANSMISSION_NOISE = 0.1

# Preallocate empty DataFrame
# N_TOTAL = N * N_GEN
# data = pd.DataFrame({
#     "id": [None] * N_TOTAL,
#     "gen": np.zeros(N_TOTAL),
#     "model_knowledge": np.full(N_TOTAL, np.nan),
#     "w": np.full(N_TOTAL, np.nan),
#     "parent_id": [None] * N_TOTAL,
#     "parent_model_knowledge": np.full(N_TOTAL, np.nan),
#     "parent_w": np.full(N_TOTAL, np.nan),
#     "parent_perf": np.full(N_TOTAL, np.nan)
# })

data = pd.DataFrame()

param_learning = {"p_model_knowledge": 1.0, "sd_model_knowledge": 0.1, "p_w": 1.0, "sd_w": 0.1}

pop = init_population(N, model_knowledge_param, w_param)
for gen in range(N_GEN):
    if gen != 0:
        children = create_new_gen(parent_values, gen, N)
        children["model_knowledge"], children["w"] = parent_transmission(children, TRANSMISSION_NOISE)
        children["model_knowledge"], children["w"] = learning(children["model_knowledge"], children["w"], param_learning)
        pop = children.copy()
    perfs = []
    for _, agent in pop.iterrows():
        perf = get_perf(perf_matrix, agent["model_knowledge"], agent["w"])
        perfs.append(perf)
    pop["perf"] = perfs
    data = pd.concat([data, pop.copy()], ignore_index=True)
    parent_model_knowledge, parent_w, parent_id = tournament_selection(pop, N)
    parent_values = {
        "id": parent_id,
        "model_knowledge": parent_model_knowledge,
        "w": parent_w,
        "perf": perfs
    }


data.to_csv("results/test_run.csv", index=False)


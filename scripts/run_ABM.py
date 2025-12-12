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

# np.random.seed(123)

perf_matrix = pd.read_csv("data/perf_matrix.csv")

N_SIM = 50
N_GEN = 10
N = 100
K = 4
model_knowledge_param = {"mean": 0.52, "sd": 0.07}
w_param = {"mean": 0.5, "sd": 0.14}
beta_params = {"alpha": 5, "beta": 1}
param_learning = {"p": 1, "sd": 0.4}

data = pd.DataFrame()

for sim in range(N_SIM):
    pop = init_population(N, model_knowledge_param, w_param)
    data_sim = pd.DataFrame()
    for gen in range(N_GEN):
        if gen != 0:
            children = create_new_gen(parent_values, gen, N)
            children["model_knowledge"], children["w"] = parent_transmission(children, beta_params)
            children = learning(children, param_learning)
            pop = children.copy()
        perfs = []
        for _, agent in pop.iterrows():
            perf = get_perf(perf_matrix, agent["model_knowledge"], agent["w"])
            perfs.append(perf)
        pop["perf"] = perfs
        data_sim = pd.concat([data_sim, pop.copy()], ignore_index=True)
        parent_model_knowledge, parent_w, parent_id = tournament_selection(pop, K)
        parent_values = {
            "id": parent_id,
            "model_knowledge": parent_model_knowledge,
            "w": parent_w,
            "perf": perfs
        }
    data_sim["sim"] = sim
    data = pd.concat([data, data_sim.copy()], ignore_index=True)

data.to_csv("data/test_run.csv", index=False)

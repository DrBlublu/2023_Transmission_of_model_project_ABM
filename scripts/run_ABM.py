"""
Docstring for scripts.run_ABM
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool
from ABM.init_population import init_population
from ABM.get_perf import get_perf
from ABM.tournament_selection import tournament_selection
from ABM.create_new_gen import create_new_gen
from ABM.parent_transmission import parent_transmission
from ABM.learning import learning

# np.random.seed(123)

perf_matrix = pd.read_csv("data/perf_matrix.csv")

N_SIM = 100
N_GEN = 10
N = 100
K = 4
model_knowledge_param = {"mean": 0.52, "sd": 0.07}
w_param = {"mean": 0.5, "sd": 0.14}

l_beta = np.arange(1, 5, 1)
l_alpha = np.arange(1, 5, 1)
l_p = np.arange(0, 1, 0.1)


data = pd.DataFrame()

def run_simulation(sim):
    results = []
    for alpha in l_alpha:
        for beta in l_beta:
            beta_params = {"alpha": alpha, "beta": beta}
            for p in l_p:
                param_learning = {"p": p, "sd": 0.2}
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

                        for _, agent in pop.iterrows():
                            results.append({
                                "sim": sim,
                                "gen": gen,
                                "alpha": alpha,
                                "beta": beta,
                                "p": p,
                                "model_knowledge": agent["model_knowledge"],
                                "w": agent["w"],
                                "perf": agent["perf"]
                            })
    return results

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        all_results = pool.map(run_simulation, range(N_SIM))

    # flatten list of lists
    results = [item for sublist in all_results for item in sublist]

    df = pd.DataFrame(results)
    df.to_csv("data/ABM_results.csv", index=False)

"""
Docstring for scripts.run_ABM
"""

from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from ABM.init_population import init_population
from ABM.get_perf import get_perf
from ABM.tournament_selection import tournament_selection
from ABM.create_new_gen import create_new_gen
from ABM.parent_transmission import parent_transmission
from ABM.learning import learning

perf_matrix = pd.read_csv("data/perf_matrix.csv")

N_SIM = 100
N_GEN = 10
N = 100
K = 4
# model_knowledge_param = {"mean": 0.52, "sd": 0.07}
# w_param = {"mean": 0.5, "sd": 0.14}
model_knowledge_param = {"mean": 0.52, "sd": 0.2}
w_param = {"mean": 0.5, "sd": 0.2}

l_beta = np.arange(1, 5.5, 0.5)
l_alpha = np.arange(1, 5.5, 0.5)
l_p = np.arange(0, 1.1, 0.2)
l_sd = np.arange(0, 1.1, 0.2)


param_grid = [
    (beta, alpha, p, sd, sim)
    for beta, alpha, p, sd in itertools.product(l_beta, l_alpha, l_p, l_sd)
    for sim in range(N_SIM)
]

def run_simulation(beta, alpha, p, sd, sim):
    beta_params = {"alpha": alpha, "beta": beta}
    param_learning = {"p": p, "sd": sd}

    pop = init_population(N, model_knowledge_param, w_param)
    data_sim = pd.DataFrame()

    for gen in range(N_GEN):
        if gen != 0:
            children = create_new_gen(parent_values, gen, N)
            children["model_knowledge"], children["w"] = parent_transmission(children, beta_params)
            children = learning(children, param_learning)
            pop = children.copy()

        perfs = [get_perf(perf_matrix, agent["model_knowledge"], agent["w"])
                 for _, agent in pop.iterrows()]
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
    data_sim["alpha"] = alpha
    data_sim["beta"] = beta
    data_sim["p"] = p
    data_sim["sd"] = sd

    return data_sim


def run_simulation_wrapper(args):
    return run_simulation(*args)


if __name__ == "__main__":
    with Pool(processes=12) as pool:
        results = list(
            tqdm(
                pool.imap(run_simulation_wrapper, param_grid),
                total=len(param_grid)
            )
        )

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv("data/ABM_results.csv", index=False)

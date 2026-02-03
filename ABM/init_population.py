"""
Module that initialize a population of N agents with random value for model knowledge
and W
"""

import numpy as np
import pandas as pd

def init_population(N, df_starting_population):
    """
    Initalize a population of size n at gen 0.

    Args:
        n (int): number of agents.
        model_knowledge_param (dict): parameters value for the model_knowledge
          gaussian.
        w_param (dict): parameters value for the w gaussian.
    
    Returns:
        pop (pd.DataFrame): population dataframe with id, model_knowledge and w.
    """
    
    starting_population_sample = df_starting_population.sample(n=N, replace=True).reset_index(drop=True)
    pop = pd.DataFrame({
        "id": [f"g0n{i}" for i in range(N)],
        "gen": np.zeros(N),
        "model_knowledge": round(np.round(starting_population_sample["model_knowledge"] / 0.02) * 0.02, 2), # keep step 0.02 for the perf matrix
        "w": starting_population_sample["w"],
        "perf": starting_population_sample["perf"],
        "parent_id": [f"fake_parent_{i}" for i in range(N)],
        "parent_model_knowledge": np.random.normal(0, 1, N),
        "parent_w": np.random.normal(0, 1, N),
        "parent_perf": np.random.normal(0, 1, N)
    })
    return pop

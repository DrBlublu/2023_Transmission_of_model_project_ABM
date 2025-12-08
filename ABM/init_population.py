"""
Module that initialize a population of N agents with random value for model knowledge
and W
"""

import numpy as np
import pandas as pd

def init_population(n, model_knowledge_param, w_param):
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
    raw_model_knowledge = np.random.normal(
        model_knowledge_param["mean"],
        model_knowledge_param["sd"],
        n
    )
    pop_model_knowledge = 1 / (1 + np.exp(-raw_model_knowledge))
    pop_model_knowledge = pop_model_knowledge.round(2)
    raw_w = np.random.normal(w_param["mean"], w_param["sd"], n)
    pop_w = 1 / (1 + np.exp(-raw_w))  # sigmoid transform
    pop_w = pop_w.round(2)
    pop = pd.DataFrame({
        "id": [f"g0n{i}" for i in range(n)],
        "gen": np.zeros(n),
        "model_knowledge": pop_model_knowledge,
        "w": pop_w,
        "parent_id": [f"fake_parent_{i}" for i in range(n)],
        "parent_model_knowledge": np.random.normal(0, 1, n),
        "parent_w": np.random.normal(0, 1, n),
        "parent_perf": np.random.normal(0, 1, n)
    })
    return pop

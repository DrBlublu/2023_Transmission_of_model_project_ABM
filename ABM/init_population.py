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
    pop_model_knowledge = np.random.normal(
        model_knowledge_param["mean"],
        model_knowledge_param["sd"],
        n
    )
    raw_w = np.random.normal(w_param["mean"], w_param["sd"], n)
    pop_w = 1 / (1 + np.exp(-raw_w))  # sigmoid transform
    pop = pd.DataFrame({
        "id": [f"g0n{i}" for i in range(n)],
        "model_knowledge": pop_model_knowledge,
        "w": pop_w
    })
    return pop

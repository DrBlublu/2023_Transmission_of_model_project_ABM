"""
Module implementing learning in the agents.
"""


import random
import numpy as np
import pandas as pd


def learning(agents: pd.DataFrame, param_learning: dict) -> tuple: 
    """
    Learning function for agents.
    If agents are lucky to learn (based on prob in param_learning), they gain
    value drawn from a gaussian distribution with std based on param_learning.

    Args:
        agents (pd.DataFrame): Population of agent that will try to learn.
        param_learning (dict): Dictionnaire containing probability of learning and
          strandard deviation value for the gaussian learning value.
    
    Returns:
        agents_copy (pd.dataFrame): A copy of the input dataframe.
    """

    agents_copy = agents.copy()

    mask = np.random.rand(len(agents)) <= param_learning["p"]
    learning_values = np.random.normal(0, param_learning["sd"], size=len(agents))
    agents_copy.loc[mask, "model_knowledge"] = (
        round(np.round(np.clip(agents.loc[mask, "model_knowledge"] + learning_values[mask], 0, 1) / 0.02) * 0.02, 2)
    )
    agents_copy.loc[mask, "w"] = (
        np.clip(agents.loc[mask, "w"] + learning_values[mask], 0, 1).round(2)
    )
    return agents_copy

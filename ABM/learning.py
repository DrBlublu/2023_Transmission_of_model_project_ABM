"""
Module implementing learning in the agents.
"""


import random
import numpy as np


def learning(model_knowledge: float, w: float, param_learning: dict) -> tuple: 
    """
    Learning function for agents.
    If agents are lucky to learn (based on prob in param_learning), they gain
    value drawn from a gaussian distribution with std based on param_learning.

    Args:
        model_knowledge (float): Model knowledge value of agent.
        w (float): W value of agent.
        param_learning (dict): Dictionnaire containing probability of learnign and
          strandard deviation value for the gaussian learning value.
    
    Returns:
        model_knowledge (float): Updated model_knowledge value.
        w (float): Updated w value.
    """
    if random.random() <= param_learning["p_model_knowledge"]:
        model_knowledge_change = np.random.normal(0, param_learning["sd_model_knowledge"])
        model_knowledge = np.clip(model_knowledge + model_knowledge_change, 0, 1)
    if random.random() <= param_learning["p_w"]:
        w_change = np.random.normal(0, param_learning["sd_w"])
        w = np.clip(w + w_change, 0, 1)
    return model_knowledge, w

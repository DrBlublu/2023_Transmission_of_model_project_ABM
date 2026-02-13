"""
Module for initializing the transition function used in the simulation.py
"""


import numpy as np


def init_transition_function(model_knowledge: float) -> np.ndarray:
    """
    Initialize a transition matrix with a specific model_knowledge.

    Parameters:
        model_knowledge (float): Model knowledge.

    Returns:
        transition_function (np.ndarray): A transition function with the diagonal
          representing the model knowledge. Format is (4, 2, 2).
    """
    transition_function = np.array([[model_knowledge, 1 - model_knowledge],
                                    [1 - model_knowledge, model_knowledge]])

    transition_function = np.tile(transition_function, (4, 1, 1))
    return transition_function

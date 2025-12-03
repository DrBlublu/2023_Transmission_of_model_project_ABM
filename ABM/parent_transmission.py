"""
Module that add Gaussian noise to the tranmission of information from parents
to children.
"""


import numpy as np
import pandas as pd


def parent_transmission(agents: pd.DataFrame, noise_std: float) -> tuple:
    """
    Adds transmission noise to agent value.

    Args:
        agents (pd.DataFrame): DataFrame containing the children population with
          model knowledge and w value.
        noise_std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        children_model_knowledge (pd.DataFrame): Children model knowledge after transmission.
        children_w (pd.DataFrame): Children W after tranmission.
    """
    noise = np.random.normal(0, noise_std, agents.shape[0])

    children_model_knowledge = np.clip(agents["model_knowledge"] + noise, 0, 1)
    children_w = np.clip(agents["w"] + noise, 0, 1)

    return children_model_knowledge, children_w

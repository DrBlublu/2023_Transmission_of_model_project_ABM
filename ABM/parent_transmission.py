"""
Module that add Gaussian noise to the tranmission of information from parents
to children.
"""


import numpy as np
import pandas as pd


def parent_transmission(agents: pd.DataFrame, beta_params: dict) -> tuple:
    """
    Adds transmission noise to agent value.

    Args:
        agents (pd.DataFrame): DataFrame containing the children population with
          model knowledge and w value.
        beta_params (dict): Dict with parameters for the beta distribution.
    
    Returns:
        children_model_knowledge (pd.DataFrame): Children model knowledge after transmission.
        children_w (pd.DataFrame): Children W after tranmission.
    """
    noise = np.random.beta(
        beta_params["alpha"],
        beta_params["beta"],
        size=len(agents)
    )

    children_model_knowledge = agents["model_knowledge"] * noise
    children_model_knowledge = np.round(children_model_knowledge / 0.02) * 0.02
    children_model_knowledge = children_model_knowledge.round(2)

    children_w = agents["w"] * noise
    children_w = np.round(children_w / 0.02) * 0.02
    children_w = children_w.round(2)

    return children_model_knowledge, children_w

"""
Module used to create a new generation based on a previous generation.
"""


import numpy as np
import pandas as pd


def create_new_gen(parent_values: pd.DataFrame, gen: int, n: int) -> pd.DataFrame:
    """
    Create a new generation of agents.

    Args:
        parent_values (pd.DataFrame): DataFrame containing parent agent values
            with columns ["id", "model_knowledge", "w", "perf"].
        gen (int): Generation number.
        n (int): Number of agents in the new generation.

    Returns:
        pd.DataFrame: DataFrame of the new generation population.
    """

    new_gen = pd.DataFrame({
        "id": [f"g{gen}n{i}" for i in range(n)],
        "gen": np.full(n, gen),
        "model_knowledge": parent_values["model_knowledge"],
        "w": parent_values["w"],
        "parent_id": parent_values["id"],
        "parent_model_knowledge": parent_values["model_knowledge"],
        "parent_w": parent_values["w"],
        "parent_perf": parent_values["perf"]
    })

    return new_gen

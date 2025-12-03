"""
Module that run tournament selection.
"""

import numpy as np
import pandas as pd

def tournament_selection(parents, k=4):
    """
    Perform tournament selection to choose parent rows based on simulated scores.

    Args:
        parents (pd.DataFrame): DataFrame containing 'id', 'perf', 'w', 'model_knowledge' of parents.
        k (int): k parameter from the tournament selection algorithm.

    Returns:
        tuple of lists: (list_of_model_knowledge, list_of_w, list_of_id)
    """
    model_knowledge_list = []
    w_list = []
    id_list = []

    for _ in range(parents.shape[0]):
        # Sample candidate IDs
        candidates = np.random.choice(parents["id"].values, k, replace=True)

        # Get candidate rows and sort by performance
        perf = parents[parents["id"].isin(candidates)].sort_values(
            "perf", ascending=False
        )

        # Select the best candidate row
        best_row = perf.iloc[0]

        # Append values to lists
        model_knowledge_list.append(best_row["model_knowledge"])
        w_list.append(best_row["w"])
        id_list.append(best_row["id"])

    return model_knowledge_list, w_list, id_list

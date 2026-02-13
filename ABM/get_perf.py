"""
Module that get the performance of an agent based on its model knowledge & W.
"""

def get_perf(perf_matrix, agent_model_knowledge, agent_w):
    """
    Get performance of an agent based on its model knowledge & W.
    Performance is drawn from the simulation results.

    Args:
        perf_matrix (pd.DataFrame): dataframe of the average over simulation results.
        agent_model_knowledge (float): agent model knowledge value.
        aget_w (float): agent w value.
    
    Returns:
        perf (float): performance of the agent.
    """

    dist = (
        (perf_matrix["model_knowledge"] - agent_model_knowledge)**2 +
        (perf_matrix["w"] - agent_w)**2
    )

    # Index of closest row
    idx = dist.idxmin()

    return perf_matrix.loc[idx, "perf_stake"]

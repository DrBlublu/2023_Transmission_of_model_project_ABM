"""
Docstring for ABM.get_perf
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

    perf = perf_matrix.loc[
        (perf_matrix["model_knowledge"] == agent_model_knowledge) & (perf_matrix["w"] == agent_w),
        ["perf"]
        ].values[0][0]

    return perf

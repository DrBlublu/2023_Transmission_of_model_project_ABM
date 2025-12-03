"""
Module that change a simulation csv file from "simulation.py" to a 
perf matrix, with average performance perf value of model knowledge & w.
"""

import pandas as pd

def transform_simulation_results(source_filename, output_filename):
    """
    Transform a simulation_result file to a perf_matrix file.
    Average over simulation for value of model knowledge & w.

    Args:
        source_filename (str): name of the simulation.py result file.
        output_filename (str): name of the output file.
    """

    df = pd.read_csv(f"data/{source_filename}.csv")

    df_matrix = (
    df.groupby(["model_knowledge", "w"], as_index=False)["perf"]
      .mean()
      .assign(
          model_knowledge=lambda d: d["model_knowledge"].round(2),
          w=lambda d: d["w"].round(2)
      )
    )

    df_matrix.to_csv(f"data/{output_filename}.csv", index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m ABM.perf_matrix <source_filename> <output_filename>")
    else:
        transform_simulation_results(sys.argv[1], sys.argv[2])

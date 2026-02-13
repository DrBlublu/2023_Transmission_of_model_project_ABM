import numpy as np
import pandas as pd
from ABM.init_population import init_population

df_starting_population = pd.read_csv("data/starting_population.csv")


def test_init_population_shape_and_columns():
    """
    Test population creation.
    DataFrame should have n rows and the correct columns,
        assert pop.shape[0] == n, and
        assert set(pop.columns) == {"model_knowledge", "w"}.
    """
    np.random.seed(42)
    n = 10

    pop = init_population(n, df_starting_population)

    assert pop.shape[0] == n
    assert set(pop.columns) == {"id", "gen", "model_knowledge", "w", "perf", "parent_id", "parent_model_knowledge", "parent_w", "parent_perf"}


def test_init_population_reproducibility():
    """
    Test reproducibility with fixed seed.
    Two calls with the same seed should produce identical DataFrames,
        pd.testing.assert_frame_equal(pop1, pop2).
    """
    np.random.seed(999)
    n = 5
    pop1 = init_population(n, df_starting_population)
    np.random.seed(999)
    pop2 = init_population(n, df_starting_population)

    pd.testing.assert_frame_equal(pop1, pop2)

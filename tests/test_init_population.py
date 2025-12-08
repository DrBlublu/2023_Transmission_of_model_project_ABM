import numpy as np
import pandas as pd
from ABM.init_population import init_population


def test_init_population_shape_and_columns():
    """
    Test population creation.
    DataFrame should have n rows and the correct columns,
        assert pop.shape[0] == n, and
        assert set(pop.columns) == {"model_knowledge", "w"}.
    """
    np.random.seed(42)
    n = 10
    model_knowledge_param = {"mean": 0.5, "sd": 0.1}
    w_param = {"mean": 0.0, "sd": 1.0}

    pop = init_population(n, model_knowledge_param, w_param)

    assert pop.shape[0] == n
    assert set(pop.columns) == {"id", "gen", "model_knowledge", "w", "parent_id", "parent_model_knowledge", "parent_w", "parent_perf"}


def test_init_population_w_in_bounds():
    """
    Test that w values are bounded.
    All w values should lie in [0,1],
        assert np.all(pop["w"] >= 0), and
        assert np.all(pop["w"] <= 1).
    """
    np.random.seed(123)
    n = 20
    model_knowledge_param = {"mean": 1.0, "sd": 0.5}
    w_param = {"mean": 0.0, "sd": 2.0}

    pop = init_population(n, model_knowledge_param, w_param)

    assert np.all(pop["w"] >= 0)
    assert np.all(pop["w"] <= 1)


def test_init_population_reproducibility():
    """
    Test reproducibility with fixed seed.
    Two calls with the same seed should produce identical DataFrames,
        pd.testing.assert_frame_equal(pop1, pop2).
    """
    np.random.seed(999)
    n = 5
    model_knowledge_param = {"mean": 0.0, "sd": 1.0}
    w_param = {"mean": 0.0, "sd": 1.0}

    pop1 = init_population(n, model_knowledge_param, w_param)
    np.random.seed(999)
    pop2 = init_population(n, model_knowledge_param, w_param)

    pd.testing.assert_frame_equal(pop1, pop2)

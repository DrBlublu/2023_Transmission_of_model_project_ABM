import numpy as np
import pandas as pd
import pytest
from ABM.parent_transmission import parent_transmission


def test_parent_transmission_shapes_and_types():
    """
    Test parent_transmission output shapes and types.
    Returned children_model_knowledge and children_w should be numpy arrays of length equal to agents,
        assert len(children_model_knowledge) == len(agents), and
        assert isinstance(children_model_knowledge, np.ndarray).
    """
    np.random.seed(37)
    agents = pd.DataFrame({"model_knowledge": [0.2, 0.8], "w": [0.5, 0.7]})
    children_model_knowledge, children_w = parent_transmission(agents, noise_std=0.1)

    assert len(children_model_knowledge) == len(agents)
    assert len(children_w) == len(agents)
    assert isinstance(children_model_knowledge, pd.Series)
    assert isinstance(children_w, pd.Series)


def test_parent_transmission_zero_noise():
    """
    Test parent_transmission with noise_std=0.
    Outputs should equal inputs exactly,
        np.testing.assert_array_equal(children_model_knowledge, agents["model_knowledge"]), and
        np.testing.assert_array_equal(children_w, agents["w"]).
    """
    np.random.seed(37)
    agents = pd.DataFrame({"model_knowledge": [0.1, 0.9], "w": [0.3, 0.7]})
    children_model_knowledge, children_w = parent_transmission(agents, noise_std=0.0)

    np.testing.assert_array_equal(children_model_knowledge, agents["model_knowledge"])
    np.testing.assert_array_equal(children_w, agents["w"])


def test_parent_transmission_values_clipped():
    """
    Test that outputs are clipped to [0,1].
    All values in children_model_knowledge and children_w should lie between 0 and 1,
        assert np.all(children_model_knowledge >= 0) and np.all(children_model_knowledge <= 1).
    """
    np.random.seed(37)
    agents = pd.DataFrame({"model_knowledge": [0.0, 1.0], "w": [0.0, 1.0]})
    children_model_knowledge, children_w = parent_transmission(agents, noise_std=5.0)

    assert np.all(children_model_knowledge >= 0)
    assert np.all(children_model_knowledge <= 1)
    assert np.all(children_w >= 0)
    assert np.all(children_w <= 1)


def test_parent_transmission_reproducibility():
    """
    Test reproducibility with fixed seed.
    Two calls with the same seed should produce identical outputs,
        np.testing.assert_array_equal(children_model_knowledge1, children_model_knowledge2), and
        np.testing.assert_array_equal(children_w1, children_w2).
    """
    agents = pd.DataFrame({"model_knowledge": [0.4, 0.6], "w": [0.2, 0.8]})

    np.random.seed(37)
    children_model_knowledge1, children_w1 = parent_transmission(agents, noise_std=0.5)

    np.random.seed(37)
    children_model_knowledge2, children_w2 = parent_transmission(agents, noise_std=0.5)

    np.testing.assert_array_equal(children_model_knowledge1, children_model_knowledge2)
    np.testing.assert_array_equal(children_w1, children_w2)

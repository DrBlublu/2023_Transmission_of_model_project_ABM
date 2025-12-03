import numpy as np
import random

from ABM.learning import learning

def test_learning_no_update_when_prob_zero():
    """
    Test learning when probabilities are zero.
    No update should occur, values remain unchanged,
        assert updated_model_knowledge == initial_model_knowledge, and
        assert updated_w == initial_w.
    """
    random.seed(37)
    np.random.seed(37)
    initial_model_knowledge, initial_w = 0.5, 0.5
    params = {"p_model_knowledge": 0.0, "sd_model_knowledge": 1.0,
              "p_w": 0.0, "sd_w": 1.0}

    updated_model_knowledge, updated_w = learning(initial_model_knowledge, initial_w, params)

    assert updated_model_knowledge == initial_model_knowledge
    assert updated_w == initial_w


def test_learning_always_update_when_prob_one():
    """
    Test learning when probabilities are one.
    Updates should always occur, values should change according to Gaussian noise,
        assert updated_model_knowledge != initial_model_knowledge, and
        assert updated_w != initial_w.
    """
    random.seed(123)
    np.random.seed(123)
    initial_model_knowledge, initial_w = 0.5, 0.5
    params = {"p_model_knowledge": 1.0, "sd_model_knowledge": 0.1,
              "p_w": 1.0, "sd_w": 0.1}

    updated_model_knowledge, updated_w = learning(initial_model_knowledge, initial_w, params)

    assert updated_model_knowledge != initial_model_knowledge
    assert updated_w != initial_w
    assert 0 <= updated_model_knowledge <= 1
    assert 0 <= updated_w <= 1


def test_learning_clipping_bounds():
    """
    Test learning clipping behavior.
    Updates should be clipped to [0,1],
        assert np.all((updated_model_knowledge >= 0) & (updated_model_knowledge <= 1)), and
        assert np.all((updated_w >= 0) & (updated_w <= 1)).
    """
    random.seed(999)
    np.random.seed(999)
    initial_model_knowledge, initial_w = 0.0, 1.0
    params = {"p_model_knowledge": 1.0, "sd_model_knowledge": 5.0,
              "p_w": 1.0, "sd_w": 5.0}

    updated_model_knowledge, updated_w = learning(initial_model_knowledge, initial_w, params)

    assert 0 <= updated_model_knowledge <= 1
    assert 0 <= updated_w <= 1


def test_learning_reproducibility_with_seed():
    """
    Test reproducibility with fixed seed.
    Two calls with the same seed should produce identical outputs,
        assert (mk1, w1) == (mk2, w2).
    """
    initial_model_knowledge, initial_w = 0.5, 0.5
    params = {"p_model_knowledge": 1.0, "sd_model_knowledge": 0.1,
              "p_w": 1.0, "sd_w": 0.1}

    random.seed(2025)
    np.random.seed(2025)
    mk1, w1 = learning(initial_model_knowledge, initial_w, params)

    random.seed(2025)
    np.random.seed(2025)
    mk2, w2 = learning(initial_model_knowledge, initial_w, params)

    assert mk1 == mk2
    assert w1 == w2

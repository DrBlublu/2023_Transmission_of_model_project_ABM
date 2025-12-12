import numpy as np
import random

import pandas as pd
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
    agents = pd.DataFrame({"model_knowledge": [0.2, 0.8], "w": [0.5, 0.7]})
    params = {"p": 0.0, "sd": 1.0,}

    agents_bis = learning(agents, params)

    assert (agents["model_knowledge"] == agents_bis["model_knowledge"]).all()
    assert (agents["w"] == agents_bis["w"]).all()


def test_learning_always_update_when_prob_one():
    """
    Test learning when probabilities are one.
    Updates should always occur, values should change according to Gaussian noise,
        assert updated_model_knowledge != initial_model_knowledge, and
        assert updated_w != initial_w.
    """
    random.seed(123)
    np.random.seed(123)
    agents = pd.DataFrame({"model_knowledge": [0.2, 0.8], "w": [0.5, 0.7]})
    params = {"p": 1.0, "sd": 1.0,}

    agents_bis = learning(agents, params)

    assert (agents["model_knowledge"] != agents_bis["model_knowledge"]).all()
    assert (agents["w"] != agents_bis["w"]).all()
    assert 0 <= (agents_bis["model_knowledge"]).all() <= 1
    assert 0 <= (agents_bis["w"]).all() <= 1


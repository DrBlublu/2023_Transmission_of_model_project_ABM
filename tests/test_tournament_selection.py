import numpy as np
import pandas as pd
import pytest
from ABM.tournament_selection import tournament_selection


def test_tournament_selection_output_lengths_and_types():
    """
    Test tournament_selection output lengths and types.
    Returned lists should have length equal to number of parents,
        assert len(knowledge_list) == len(parents), and
        assert isinstance(knowledge_list, list).
    """
    np.random.seed(37)
    parents = pd.DataFrame({
        "id": [1, 2, 3],
        "perf": [0.1, 0.9, 0.5],
        "w": [0.2, 0.4, 0.6],
        "model_knowledge": [0.3, 0.7, 0.9]
    })
    model_knowledge_list, w_list, id_list = tournament_selection(parents, k=2)

    assert len(model_knowledge_list) == len(parents)
    assert len(w_list) == len(parents)
    assert len(id_list) == len(parents)
    assert isinstance(model_knowledge_list, list)
    assert isinstance(w_list, list)
    assert isinstance(id_list, list)


def test_tournament_selection_values_in_parents():
    """
    Test that selected values come from parents DataFrame.
    All ids in id_list should be in parents['id'],
        assert set(id_list).issubset(set(parents['id'])).
    """
    np.random.seed(37)
    parents = pd.DataFrame({
        "id": [10, 20, 30],
        "perf": [1.0, 2.0, 3.0],
        "w": [0.1, 0.2, 0.3],
        "model_knowledge": [0.5, 0.6, 0.7]
    })
    model_knowledge_list, w_list, id_list = tournament_selection(parents, k=2)

    assert set(id_list).issubset(set(parents["id"]))
    assert set(w_list).issubset(set(parents["w"]))
    assert set(model_knowledge_list).issubset(set(parents["model_knowledge"]))


def test_tournament_selection_reproducibility():
    """
    Test reproducibility with fixed seed.
    Two calls with the same seed should produce identical outputs,
        assert knowledge_list1 == knowledge_list2, and
        assert id_list1 == id_list2.
    """
    parents = pd.DataFrame({
        "id": [1, 2],
        "perf": [0.5, 0.8],
        "w": [0.2, 0.4],
        "model_knowledge": [0.3, 0.7]
    })

    np.random.seed(37)
    model_knowledge_list1, w_list1, id_list1 = tournament_selection(parents, k=2)

    np.random.seed(37)
    model_knowledge_list2, w_list2, id_list2 = tournament_selection(parents, k=2)

    assert model_knowledge_list1 == model_knowledge_list2
    assert w_list1 == w_list2
    assert id_list1 == id_list2

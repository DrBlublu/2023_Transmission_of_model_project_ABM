import numpy as np
from ABM.init_transition_function import init_transition_function

def test_init_transition_function_shape_and_values():
    """
    Test transition function initialization.
    Transition function should have shape (4, 2, 2), with diagonal entries equal to model_knowledge
    and off-diagonal entries equal to 1 - model_knowledge,
        assert transition_function.shape == (4, 2, 2), and
        np.testing.assert_almost_equal(transition_function[0,0,0], model_knowledge, decimal=6).
    """
    model_knowledge = 0.7
    transition_function = init_transition_function(model_knowledge)

    assert transition_function.shape == (4, 2, 2)
    np.testing.assert_almost_equal(transition_function[0, 0, 0], model_knowledge, decimal=6)
    np.testing.assert_almost_equal(transition_function[0, 0, 1], 1 - model_knowledge, decimal=6)
    np.testing.assert_almost_equal(transition_function[0, 1, 0], 1 - model_knowledge, decimal=6)
    np.testing.assert_almost_equal(transition_function[0, 1, 1], model_knowledge, decimal=6)


def test_init_transition_function_model_knowledge_one():
    """
    Test transition function when model_knowledge=1.0.
    Transition function should be identity matrices repeated 4 times,
        np.testing.assert_array_equal(transition_function[0], np.eye(2)).
    """
    transition_function = init_transition_function(1.0)
    expected = np.eye(2)

    for i in range(4):
        np.testing.assert_array_equal(transition_function[i], expected)


def test_init_transition_function_model_knowledge_zero():
    """
    Test transition function when model_knowledge=0.0.
    Transition function should be flipped identity matrices repeated 4 times,
        np.testing.assert_array_equal(transition_function[0], np.array([[0,1],[1,0]])).
    """
    transition_function = init_transition_function(0.0)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])

    for i in range(4):
        np.testing.assert_array_equal(transition_function[i], expected)

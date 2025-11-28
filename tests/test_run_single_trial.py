"""
Test module that test all functions of run_single_trials.
"""


import numpy as np
from ABM.run_single_trial import compute_q_values, compute_arbitration_weight, compute_prediction_error, compute_reward_and_pe2, update_q_values, update_model_performance

def test_compute_q_values():
    """
    Test the q values.
    The probabilities should sum to one,
      np.testing.assert_almost_equal(np.sum(p_mb), 1.0, decimal=6), where
      np.sum(p_mb) is the sum probabilities.
    """
    transition_function = np.zeros((4, 2, 2)) + 0.5
    qvalues = {
        "qmf1": np.zeros((4,2)) + 4.5,
        "qmf2": np.full(2, 4.5),
        "qmb": np.full(2, 4.5),
    }
    parameters = {"beta": 5.0}
    state1 = 0

    p_mb, p_mf = compute_q_values(transition_function, state1, qvalues, parameters)

    np.testing.assert_almost_equal(np.sum(p_mb), 1.0, decimal=6)
    np.testing.assert_almost_equal(np.sum(p_mf), 1.0, decimal=6)


def test_arbitration_weight_properties():
    """
    Test the arbitration weight.
    The probabilities should sum to 1,
        np.testing.assert_almost_equal(np.sum(p), 1.0, decimal=6), where
        np.sum(p) is the sum probabilities.
    The probabilites should be in [0, 1],
        assert np.all(p >= 0), and
        assert np.all(p <= 1).
    The probabilities should match the fixed W,
        expected = fixed_w * p_mb + (1 - fixed_w) * p_mf, and
        np.testing.assert_almost_equal(p, expected, decimal=6), where
        expected is the expected weight under w = 0.8.
    """
    parameters = {"n0": 5.0, "n1": 0.9}
    model_perf = {"based": np.zeros(4) + 4.5, "free": np.zeros(4) + 4.5}
    state1 = 2
    p_mb = np.array([0.9, 0.1])
    p_mf = np.array([0.2, 0.8])
    fixed_w = 0.8

    p = compute_arbitration_weight(parameters, model_perf, state1, p_mb, p_mf, fixed_w)

    np.testing.assert_almost_equal(np.sum(p), 1.0, decimal=6)
    assert np.all(p >= 0)
    assert np.all(p <= 1)

    expected = fixed_w * p_mb + (1 - fixed_w) * p_mf
    np.testing.assert_almost_equal(p, expected, decimal=6)


def test_arbitration_weight_pure_model_based():
    """
    Test the arbitration weight if pure model based.
    When fixed_w=1, output should equal p_mb,
        np.testing.assert_almost_equal(p, p_mb, decimal=6), where
        p is the actual probability and p_mb is the model based probability.
    """
    parameters = {"n0": 5.0, "n1": 0.9}
    model_perf = {"based": np.zeros(4) + 4.5, "free": np.zeros(4) + 4.5}
    state1 = 2
    p_mb = np.array([0.9, 0.1])
    p_mf = np.array([0.2, 0.8])

    p = compute_arbitration_weight(parameters, model_perf, state1, p_mb, p_mf, fixed_w=1.0)
    np.testing.assert_almost_equal(p, p_mb, decimal=6)


def test_arbitration_weight_pure_model_free():
    """
    Test the arbitration weight if pure model free.
    When fixed_w=1, output should equal p_mf,
        np.testing.assert_almost_equal(p, p_mf, decimal=6), where
        p is the actual probability and p_mf is the model based probability.
    """
    parameters = {"n0": 5.0, "n1": 0.9}
    model_perf = {"based": np.zeros(4) + 4.5, "free": np.zeros(4) + 4.5}
    state1 = 2
    p_mb = np.array([0.9, 0.1])
    p_mf = np.array([0.2, 0.8])

    p = compute_arbitration_weight(parameters, model_perf, state1, p_mb, p_mf, fixed_w=0.0)
    np.testing.assert_almost_equal(p, p_mf, decimal=6)


def test_prediction_error_positive():
    """
    Test the prediction error.
    Prediction error should increase qmf1 when qmf2 > qmf1,
        np.testing.assert_almost_equal(pe, expected, decimal=6), where
        pe is the prediction error, and
        expected = 1.0 + 0.5 * (3.0 - 1.0).
    """
    qvalues = {
        "qmf1": np.array([[1.0, 2.0], [0.5, 1.5]]),
        "qmf2": np.array([3.0, 1.0])
    }
    parameters = {"alpha": 0.5}
    state1, state2, choice = 0, 0, 0  # qmf1=1.0, qmf2=3.0

    pe = compute_prediction_error(qvalues, state2, state1, choice, parameters)

    # Expected: old_value + alpha * (qmf2 - old_value)
    expected = 1.0 + 0.5 * (3.0 - 1.0)
    np.testing.assert_almost_equal(pe, expected, decimal=6)


def test_prediction_error_negative():
    """
    Test the prediction error.
    Prediction error should decrease qmf1 when qmf2 < qmf1,
        np.testing.assert_almost_equal(pe, expected, decimal=6), where
        pe is the prediction error, and
        expected = 2.0 + 0.25 * (0.5 - 2.0).
    """
    qvalues = {
        "qmf1": np.array([[2.0, 1.0], [0.5, 1.5]]),
        "qmf2": np.array([0.5, 1.0])
    }
    parameters = {"alpha": 0.25}
    state1, state2, choice = 0, 0, 0  # qmf1=2.0, qmf2=0.5

    pe = compute_prediction_error(qvalues, state2, state1, choice, parameters)

    # Expected: old_value + alpha * (qmf2 - old_value)
    expected = 2.0 + 0.25 * (0.5 - 2.0)
    np.testing.assert_almost_equal(pe, expected, decimal=6)


def test_prediction_error_alpha_zero():
    """
    Test the prediction error.
    Prediction error should be old qmf1 when alpha=0,
        np.testing.assert_almost_equal(pe, expected, decimal=6), where
        pe is the prediction error, and
        expected = qvalues["qmf1"][state1][choice] = 2.0.
    """
    qvalues = {
        "qmf1": np.array([[1.0, 2.0], [0.5, 1.5]]),
        "qmf2": np.array([3.0, 1.0])
    }
    parameters = {"alpha": 0.0}
    state1, state2, choice = 0, 0, 1  # qmf1=2.0, qmf2=3.0

    pe = compute_prediction_error(qvalues, state2, state1, choice, parameters)

    expected = qvalues["qmf1"][state1][choice]  # no update
    np.testing.assert_almost_equal(pe, expected, decimal=6)


def test_prediction_error_alpha_one():
    """
    Test the prediction error.
    Prediction error should be qmf2 when alpha=1,
        np.testing.assert_almost_equal(pe, expected, decimal=6), where
        pe is the prediction error, and
        expected = qvalues["qmf2"][state2] = 3.0.
    """
    qvalues = {
        "qmf1": np.array([[1.0, 2.0], [0.5, 1.5]]),
        "qmf2": np.array([3.0, 1.0])
    }
    parameters = {"alpha": 1.0}
    state1, state2, choice = 0, 0, 1  # qmf1=2.0, qmf2=3.0

    pe = compute_prediction_error(qvalues, state2, state1, choice, parameters)

    expected = qvalues["qmf2"][state2]  # full update
    np.testing.assert_almost_equal(pe, expected, decimal=6)


def test_reward_and_pe2_basic():
    """
    Test the reward and prediction error.
    Reward should equal rewards[choice], and prediction error should be reward - qmf2[choice],
        np.testing.assert_almost_equal(pe_2, expected, decimal=6), where
        reward = rewards[choice] = 2.0, and
        expected = rewards[choice] - qvalues["qmf2"][choice].
    """
    rewards = np.array([1.0, 2.0])
    qvalues = {"qmf2": np.array([0.5, 2.5])}
    choice = 1

    reward, pe_2 = compute_reward_and_pe2(choice, rewards, qvalues)

    expected = rewards[choice] - qvalues["qmf2"][choice]
    assert reward == rewards[choice]
    np.testing.assert_almost_equal(pe_2, expected, decimal=6)


def test_reward_and_pe2_positive_error():
    """
    Test the prediction error.
    Prediction error should be positive when reward > qmf2,
        assert pe_2 > 0, where
        reward = rewards[choice] = 5.0, and
        qmf2[choice] = 2.0.
    """
    rewards = np.array([5.0, 1.0])
    qvalues = {"qmf2": np.array([2.0, 1.0])}
    choice = 0

    reward, pe_2 = compute_reward_and_pe2(choice, rewards, qvalues)

    assert reward == rewards[choice]
    assert pe_2 > 0


def test_reward_and_pe2_negative_error():
    """
    Test the prediction error.
    Prediction error should be negative when reward < qmf2,
        assert pe_2 < 0, where
        reward = rewards[choice] = 4.0, and
        qmf2[choice] = 5.0.
    """
    rewards = np.array([1.0, 4.0])
    qvalues = {"qmf2": np.array([2.0, 5.0])}
    choice = 1

    reward, pe_2 = compute_reward_and_pe2(choice, rewards, qvalues)

    assert reward == rewards[choice]
    assert pe_2 < 0


def test_reward_and_pe2_zero_error():
    """
    Test the prediction error.
    Prediction error should be zero when reward == qmf2,
        assert pe_2 == 0.0, where
        reward = rewards[choice] = 3.0, and
        qmf2[choice] = 3.0.
    """
    rewards = np.array([3.0])
    qvalues = {"qmf2": np.array([3.0])}
    choice = 0

    reward, pe_2 = compute_reward_and_pe2(choice, rewards, qvalues)

    assert reward == rewards[choice]
    assert pe_2 == 0.0


def test_update_q_values_qmf2_update():
    """
    Test Q-value update for qmf2.
    qmf2[state2] should be updated as old_value + alpha * error,
        np.testing.assert_almost_equal(updated, expected, decimal=6), where
        old_value = 1.0, parameters["alpha"] = 0.5, pe_2 = 2.0, and
        expected = 2.0.
    """
    qvalues = {"qmf1": np.array([[1.0, 2.0]]), "qmf2": np.array([1.0, 3.0])}
    parameters = {"alpha": 0.5, "trace": 1.0}
    state1, state2, choice, pe_2 = 0, 0, 1, 2.0

    updated = update_q_values(qvalues, state1, state2, choice, pe_2, parameters)

    expected = 1.0 + parameters["alpha"] * pe_2
    np.testing.assert_almost_equal(updated["qmf2"][state2], expected, decimal=6)


def test_update_q_values_qmf1_update_with_trace():
    """
    Test Q-value update for qmf1 with trace.
    qmf1[state1, choice] should be updated as old_value + alpha * trace * error,
        np.testing.assert_almost_equal(updated, expected, decimal=6), where
        old_value = 2.0, parameters["alpha"] = 0.5, trace = 0.5, pe_2 = 2.0, and
        expected = 2.5.
    """
    qvalues = {"qmf1": np.array([[1.0, 2.0]]), "qmf2": np.array([1.0, 3.0])}
    parameters = {"alpha": 0.5, "trace": 0.5}
    state1, state2, choice, pe_2 = 0, 1, 1, 2.0

    updated = update_q_values(qvalues, state1, state2, choice, pe_2, parameters)

    expected = 2.0 + parameters["alpha"] * parameters["trace"] * pe_2
    np.testing.assert_almost_equal(updated["qmf1"][state1, choice], expected, decimal=6)


def test_update_q_values_alpha_zero():
    """
    Test Q-value update when alpha=0.
    No update should occur, values remain unchanged,
        np.testing.assert_almost_equal(updated, old_value, decimal=6).
    """
    qvalues = {"qmf1": np.array([[1.0, 2.0]]), "qmf2": np.array([1.0, 3.0])}
    parameters = {"alpha": 0.0, "trace": 1.0}
    state1, state2, choice, pe_2 = 0, 0, 1, 5.0

    updated = update_q_values(qvalues, state1, state2, choice, pe_2, parameters)

    np.testing.assert_almost_equal(updated["qmf2"][state2], 1.0, decimal=6)
    np.testing.assert_almost_equal(updated["qmf1"][state1, choice], 2.0, decimal=6)


def test_update_model_performance_basic():
    """
    Test model performance update.
    Both 'based' and 'free' should be updated according to the weighted prediction errors,
        np.testing.assert_almost_equal(updated["based"][state1], expected_based, decimal=6), and
        np.testing.assert_almost_equal(updated["free"][state1], expected_free, decimal=6), where
        expected_based = 1.0 + 0.5 * ((0.6/0.5) * (2.0 - 1.0)) = 1.6, and
        expected_free  = 1.0 + 0.5 * ((0.4/0.5) * (2.0 - 1.0)) = 1.4.
    """
    model_perf = {"based": np.array([1.0]), "free": np.array([1.0])}
    parameters = {"alpha_performance": 0.5}
    state1, choice = 0, 0
    reward = 2.0
    p = np.array([0.5])
    p_mb = np.array([0.6])
    p_mf = np.array([0.4])

    updated = update_model_performance(model_perf, parameters, state1, choice, reward, p, p_mb, p_mf)

    expected_based = 1.0 + 0.5 * ((0.6 / 0.5) * (2.0 - 1.0))
    expected_free  = 1.0 + 0.5 * ((0.4 / 0.5) * (2.0 - 1.0))

    np.testing.assert_almost_equal(updated["based"][state1], expected_based, decimal=6)
    np.testing.assert_almost_equal(updated["free"][state1], expected_free, decimal=6)


def test_update_model_performance_alpha_zero():
    """
    Test model performance update when alpha_performance=0.
    No update should occur, values remain unchanged,
        np.testing.assert_almost_equal(updated["based"][state1], old_based, decimal=6), and
        np.testing.assert_almost_equal(updated["free"][state1], old_free, decimal=6).
    """
    model_perf = {"based": np.array([2.0]), "free": np.array([3.0])}
    parameters = {"alpha_performance": 0.0}
    state1, choice = 0, 0
    reward = 5.0
    p = np.array([0.5])
    p_mb = np.array([0.6])
    p_mf = np.array([0.4])

    updated = update_model_performance(model_perf, parameters, state1, choice, reward, p, p_mb, p_mf)

    np.testing.assert_almost_equal(updated["based"][state1], 2.0, decimal=6)
    np.testing.assert_almost_equal(updated["free"][state1], 3.0, decimal=6)


def test_update_model_performance_equal_reward():
    """
    Test model performance update when reward equals current performance.
    Prediction errors should be zero, so no update occurs,
        np.testing.assert_almost_equal(updated["based"][state1], old_based, decimal=6), and
        np.testing.assert_almost_equal(updated["free"][state1], old_free, decimal=6).
    """
    model_perf = {"based": np.array([4.0]), "free": np.array([4.0])}
    parameters = {"alpha_performance": 0.5}
    state1, choice = 0, 0
    reward = 4.0
    p = np.array([0.5])
    p_mb = np.array([0.5])
    p_mf = np.array([0.5])

    updated = update_model_performance(model_perf, parameters, state1, choice, reward, p, p_mb, p_mf)

    np.testing.assert_almost_equal(updated["based"][state1], 4.0, decimal=6)
    np.testing.assert_almost_equal(updated["free"][state1], 4.0, decimal=6)

"""
Module that run a single trial of arbitration reinforcement learning.
"""


from typing import Tuple, Dict
import numpy as np
from .utilities import softmax, td_rule


def compute_q_values(
        transition_function: np.ndarray,
        state1: int,
        qvalues: Dict[str, np.ndarray],
        parameters: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute model-based and model-free action probabilities.

    Args:
        transition_function (np.ndarray): Transition matrix mapping state1 to possible next states.
        state1 (int): Current state 1 index.
        qvalues (dict): Dictionary containing Q-values ("qmf1", "qmf2").
        parameters (dict): Dictionary containing model parameters (e.g., beta).

    Returns:
        p_mb (np.ndarray): Softmax probabilities from model-based Q-values.
        p_mf (np.ndarray): Softmax probabilities from model-free Q-values.
    """
    qmb = np.dot(transition_function[state1], qvalues["qmf2"])
    p_mb = softmax(qmb, parameters["beta"])
    p_mf = softmax(qvalues["qmf1"][state1], parameters["beta"])
    return p_mb, p_mf


def compute_arbitration_weight(
    parameters: Dict[str, float],
    model_perf: Dict[str, np.ndarray],
    state1: int,
    p_mb: np.ndarray,
    p_mf: np.ndarray,
    fixed_w: float = None
) -> np.ndarray:
    """
    Compute arbitration weight between model-based and model-free systems.

    Args:
        parameters (dict): Dictionary containing model parameters (e.g., beta).
        model_perf (dict): Dictionary containing performance estimates ("based", "free").
        state1 (int): Current state 1 index.
        p_mb (np.ndarray): Softmax probabilities from model-based Q-values.
        p_mf (np.ndarray): Softmax probabilities from model-free Q-values.
        fixed_w (float): A fix value of model basedness.

    Returns:
        p (np.ndarray): Weighted choice probabilities.
    """
    if fixed_w is not None:
        w = fixed_w
    else:
        w = 1 / (1 + np.exp(-parameters["n0"] * (model_perf["based"][state1] - model_perf["free"][state1]) + parameters["n1"]))
    p = w * p_mb + (1 - w) * p_mf
    return p


def make_choice(p: np.ndarray) -> int:
    """
    Make a choice based on arbitration probabilities.

    Args:
        p (np.ndarray): Weighted choice probabilities.

    Returns:
        choice (int): Choice index.
    """
    choice = int(np.random.uniform(0,1) > p[0])
    return choice


def compute_prediction_error(
    qvalues: Dict[str, np.ndarray],
    state2: int,
    state1: int,
    choice: int,
    parameters: Dict[str, float]
) -> float:
    """
    Compute prediction error for state1.

    Args:
        qvalues (dict): Dictionary containing Q-values ("qmf1", "qmf2").
        state2 (int): Current state 2 index.
        state1 (int): Current state 1 index.
        choice (int): Choice index.
        parameters (dict): Dictionary containing model parameters (e.g., beta).

    Returns:
        pe_1 (float): Updated prediction error for state 1.
    """
    pe_1 = qvalues["qmf2"][state2] - qvalues["qmf1"][state1, choice]
    pe_1 = td_rule(parameters["alpha"], qvalues["qmf1"][state1, choice], pe_1)
    return pe_1


def compute_reward_and_pe2(
    choice: int,
    rewards: np.ndarray,
    qvalues: Dict[str, np.ndarray]
) -> Tuple[float, float]:
    """
    Compute reward and second-level prediction error.

    Args:
        choice (int): Choice index.
        rewards (np.ndarray): Reward values for each state.
        qvalues (dict): Dictionary containing Q-values ("qmf1", "qmf2").

    Returns:
        reward (float): Obtained reward.
        pe_2 (float): Prediction error for state 2.
    """
    reward = rewards[choice]
    pe_2 = reward - qvalues["qmf2"][choice]
    return reward, pe_2


def update_q_values(
    qvalues: Dict[str, np.ndarray],
    state1: int,
    state2: int,
    choice: int,
    pe_2: float,
    parameters: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Update Q-values for both model-free and model-based systems.

    Args:
        qvalues (dict): Dictionary containing Q-values ("qmf1", "qmf2").
        state1 (int): Current state 1 index.
        state2 (int): Current state 2 index.
        choice (int): Choice index.
        pe_2 (float): Prediction error for state 2.
        parameters (dict): Dictionary containing model parameters (e.g., beta).

    Returns:
        qvalues (dict): Updated qvalues dictionary.
    """
    # Update qmf2
    qvalues["qmf2"][state2] = td_rule(
        parameters["alpha"],
        qvalues["qmf2"][state2],
        pe_2
    )
    # Update qmf1
    qvalues["qmf1"][state1, choice] = td_rule(
        parameters["alpha"],
        qvalues["qmf1"][state1, choice],
        pe_2,
        parameters["trace"]
    )
    return qvalues


def update_model_performance(
    model_perf: Dict[str, np.ndarray],
    parameters: Dict[str, float],
    state1: int,
    choice: int,
    reward: float,
    p: np.ndarray,
    p_mb: np.ndarray,
    p_mf: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Update model performance values based on prediction errors.

    Args:
        model_perf (dict): Dictionary containing performance estimates ("based", "free").
        parameters (dict): Dictionary containing model parameters (e.g., beta).
        state1 (int): Current state 1 index.
        choice (int): Choice index.
        reward (float): Obtained reward.
        p (np.ndarray): Weighted choice probabilities.
        p_mb (np.ndarray): Softmax probabilities from model-based Q-values.
        p_mf (np.ndarray): Softmax probabilities from model-free Q-values.

    Returns:
        model_perf (dict): Updated model_perf dictionary.
    """
    ppe_mb = (p_mb[choice] / p[choice]) * (reward - model_perf["based"][state1])
    ppe_mf = (p_mf[choice] / p[choice]) * (reward - model_perf["free"][state1])

    model_perf["based"][state1] += parameters["alpha_performance"] * ppe_mb
    model_perf["free"][state1] += parameters["alpha_performance"] * ppe_mf

    return model_perf



def run_single_trial(state1, rewards, parameters, transition_function, qvalues, model_perf, fixed_w = None):
    """
    Run a single trial of the arbitration model.

    Args:
        state1 (int): Current state 1 index.
        rewards (np.ndarray): Reward values for each state.
        parameters (dict): Dictionary containing model parameters (e.g., beta).
        transition_function (np.ndarray): Transition matrix mapping state1 to possible next states.
        qvalues (dict): Dictionary containing Q-values ("qmf1", "qmf2").
        model_perf (dict): Dictionary containing performance estimates ("based", "free").
    
    Returns:
        reward (float): Obtained reward.
        qvalues (dict): Updated qvalues dictionary.
        model_perf (dict): Updated model_perf dictionary.
    """

    # Compute probabilities
    p_mb, p_mf = compute_q_values(transition_function, state1, qvalues, parameters)
    p = compute_arbitration_weight(parameters, model_perf, state1, p_mb, p_mf, fixed_w)

    # Make choice and compute reward
    choice = make_choice(p)
    state2 = choice
    reward, pe_2 = compute_reward_and_pe2(choice, rewards, qvalues)

    # Compute state 1 prediction error
    pe_1 = compute_prediction_error(qvalues, state2, state1, choice, parameters)
    qvalues["qmf1"][state1, choice] = pe_1

    # Update Q-values
    qvalues = update_q_values(qvalues, state1, state2, choice, pe_2, parameters)

    # Update model performance
    model_perf = update_model_performance(
        model_perf,
        parameters,
        state1,
        choice,
        reward,
        p,
        p_mb,
        p_mf
    )

    return reward, qvalues, model_perf

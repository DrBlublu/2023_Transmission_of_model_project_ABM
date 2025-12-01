"""
a
"""

import numpy as np
from ABM.utilities import softmax, td_rule


def mb_mf_subject_arbitration(df, transition_function, n0=6, n1=-0.2,
                              alpha=0.2, alpha_performance=0.3, beta=1.0, trace=0.4):
    """
    Simulate subject arbitration between model-based and model-free decision-making.

    Parameters:
        df (pd.DataFrame): Input data containing trials, rewards, stakes, and states.
        transition_function (np.ndarray): Transition probabilities for model-based computation.
        n0 (float, optional): Arbitration sensitivity parameter (default=6).
        n1 (float, optional): Arbitration bias parameter (default=-0.2).
        alpha (float, optional): Learning rate for Q-value updates (default=0.2).
        alpha_performance (float, optional): Learning rate for performance tracking (default=0.3).
        beta (float, optional): Softmax inverse temperature (default=1.0).
        trace (float, optional): Eligibility trace parameter (default=0.4).

    Returns:
        float: Task performance, defined as the difference between mean scores and mean rewards.
    """

    # Number of trials
    n_trials = df['n_trial'].max()

    # Initialize arrays
    choices = np.zeros(n_trials)
    scores = np.zeros(n_trials)

    # Extract rewards, stakes, and states
    rewards = df[['rew_1', 'rew_2']].values
    stakes = df['stake'].values
    states1 = df['state1'].values

    # Initialize Q-values
    qmf1 = np.full((4, 2), 4.5)  # model-free Q-values (state-action)
    qmf2 = np.full(2, 4.5)       # model-free Q-values (second stage)
    qmb = np.full(2, 4.5)        # model-based Q-values

    # Performance tracking
    mb_performance = np.full(4, 4.5)
    mf_performance = np.full(4, 4.5)

    for i in range(n_trials):
        stake = stakes[i]
        s1 = states1[i]

        # Compute Q-values
        qmb = np.dot(transition_function[s1], qmf2)
        p_mb = softmax(qmb, beta)
        p_mf = softmax(qmf1[s1], beta)

        # Arbitration weight
        w = 1 / (1 + np.exp(-n0 * (mb_performance[s1] - mf_performance[s1]) + n1))
        p = w * p_mb + (1 - w) * p_mf

        # Make choice
        choice = int(np.random.rand() > p[0])
        choices[i] = choice
        s2 = choice

        # Prediction errors
        pe_1 = qmf2[s2] - qmf1[s1, choice]
        qmf1[s1, choice] = td_rule(alpha, qmf1[s1, choice], pe_1)

        # Reward and score
        reward = rewards[i, s2] * stake
        scores[i] = reward
        pe_2 = reward - qmf2[s2]

        # Update Q-values
        qmf2[s2] = td_rule(alpha, qmf2[s2], pe_2)
        qmf1[s1, choice] = td_rule(alpha, qmf1[s1, choice], pe_2, trace)

        # Performance prediction errors
        ppe_mb = (p_mb[choice] / p[choice]) * (reward - mb_performance[s1])
        ppe_mf = (p_mf[choice] / p[choice]) * (reward - mf_performance[s1])

        mb_performance[s1] += alpha_performance * ppe_mb
        mf_performance[s1] += alpha_performance * ppe_mf

    # Compute task performance
    task_performance = np.mean(scores) - np.mean(rewards)

    return task_performance

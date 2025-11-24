"""
Reinforcement learning utility functions.
"""

import numpy as np

def softmax(Q, beta):
    p =  np.exp(beta*Q) / np.sum(np.exp(beta*Q))
    return p

def td_rule(alpha, Q_chosen, PE, trace=1):
    Q_chosen_updated = Q_chosen + alpha * PE * trace
    return Q_chosen_updated

def logistic_function(x, n0, n1):
    return 1 / (1 + (np.exp(-n0 * x + n1)))

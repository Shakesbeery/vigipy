import numpy as np
from scipy.special import gdtr


# Calculation of CI lower bound
def quantiles(threshold, Q, a1, b1, a2, b2):
    """
    Calculate CI lower bound using algorithms from DuMouchel's paper
    "Bayesian Data Mining in Large Frequency Tables..." (1999)

    """
    is_scalar = np.ndim(Q) == 0
    length = 1 if is_scalar else len(Q)
    m = np.repeat(-100000.0, length)
    M = np.repeat(100000.0, length)
    x = np.repeat(1.0, length)
    cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
    while np.max(np.round(cost * 1e4)) != 0:
        S = np.sign(cost)
        xnew = (1 + S) / 2 * ((x + m) / 2) + (1 - S) / 2 * ((M + x) / 2)
        M = (1 + S) / 2 * x + (1 - S) / 2 * M
        m = (1 + S) / 2 * m + (1 - S) / 2 * x
        x = xnew
        cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
    if is_scalar:
        return float(x[0])
    return x


def f_cost_quantiles(p, threshold, Q, a1, b1, a2, b2):
    one = Q * gdtr(p, a1, b1)
    two = (1 - Q) * gdtr(p, a2, b2)
    if np.any(np.isnan(one)):
        one = 0.0
    if np.any(np.isnan(two)):
        two = 0.0
    return one + two - threshold

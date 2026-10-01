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
    max_mean = np.maximum(1000.0, 10.0 * np.maximum(a1 / np.maximum(b1, 1e-6), a2 / np.maximum(b2, 1e-6)))
    M = np.asarray(np.broadcast_to(max_mean, length), dtype=np.float64).copy()
    m = np.zeros(length, dtype=np.float64)
    x = np.repeat(1.0, length)
    cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
    iteration = 0
    max_iter = 100
    while np.max(np.round(cost * 1e4)) != 0 and iteration < max_iter:
        S = np.sign(cost)
        xnew = (1 + S) / 2 * ((x + m) / 2) + (1 - S) / 2 * ((M + x) / 2)
        M = (1 + S) / 2 * x + (1 - S) / 2 * M
        m = (1 + S) / 2 * m + (1 - S) / 2 * x
        x = xnew
        cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
        iteration += 1
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

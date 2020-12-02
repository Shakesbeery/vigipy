import numpy as np
'''
A series of short functions used to handle difference in d/p/q functions.
d - The density function
p - The distribution function
q - The quantile function
'''


def R_D__0(log_p):
    return -float('Inf') if log_p else 0.0


def R_D__1(log_p):
    return 0. if log_p else 1.0


def R_D_exp(x, log_p):
    return x if log_p else np.exp(x)


def R_DT_0(lower_tail, log_p):
    return R_D__0(log_p) if lower_tail else R_D__1(log_p)


def R_DT_1(lower_tail, log_p):
    return R_D__1(log_p) if lower_tail else R_D__0(log_p)

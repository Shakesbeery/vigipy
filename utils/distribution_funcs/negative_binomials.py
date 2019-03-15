import math
import numpy
from .nbinom_utils import pbeta, stirlerr, bd0
from .dpq_handling import R_D__0, R_D__1, R_D_exp, R_DT_0, R_DT_1

M_LN_2PI = math.log(2*math.pi)


def dnbinom(x, size, prob, give_log=False):
    '''
    The negative binomial distribution returning the density

    @param x: An array of quantiles

    @param size: target for number of successful trials,
                 or dispersion parameter

    @param prob: Probability of success in each trial

    @param give_log: Probabilities are given as log(p)

    '''
    if (math.isnan(x) or math.isnan(size) or math.isnan(prob)):
        return x + size + prob

    if prob <= 0 or prob > 1 or size < 0:
        return numpy.nan
    if x < 0 or x == float('Inf'):
        return R_D__0(give_log)

    if x == 0 and size == 0:
        return R_D__1(give_log)

    ans = dbinom_raw(size, x+size, prob, 1-prob, give_log)
    p = float(size)/(size+x)
    return (numpy.log(p)+ans) if give_log else (p*ans)


def dbinom_raw(x, n, p, q, give_log):
    if (p == 0):
        return R_D__1(give_log) if x == 0 else R_D__0(give_log)
    if (q == 0):
        return R_D__1(give_log) if x == n else R_D__0(give_log)

    if (x == 0):
        if n == 0:
            return R_D__1(give_log)
        lc = -bd0(n, n*q) - n*p if p < 0.1 else n*numpy.log(q)
        return R_D_exp(lc, give_log)

    if x == n:
        lc = -bd0(n, n*p) - n*q if p < 0.1 else n*numpy.log(p)
        return R_D_exp(lc, give_log)

    if x < 0 or x > n:
        return R_D__0(give_log)

    lc = (stirlerr(n) - stirlerr(x) - stirlerr(n-x) -
          bd0(x, n*p) - bd0(n-x, n*q))

    x = float(x)
    lf = M_LN_2PI + numpy.log(x) + numpy.log1p(-x/n)
    return R_D_exp(lc - 0.5*lf, give_log)


def pnbinom(x, size, prob, lower_tail=True, log_p=False):
    '''
    The negative binomial function returning the distribution

    @param x: A vector of quantiles

    @param size: target for number of successful trials,
                 or dispersion parameter

    @param prob: Probability of success in each trial

    @param lower_tail: probabilities are P[X ? x], otherwise, P[X > x]

    @param log_p: Probabilities are given as log(p)

    '''
    if math.isnan(x) or math.isnan(size) or math.isnan(prob):
        return x + size + prob
    if size == float('Inf') or prob == float('Inf'):
        return numpy.nan
    if size < 0 or prob <= 0 or prob > 1:
        return numpy.nan

    if size == 0:
        return R_DT_1(lower_tail, log_p) if x >= 0 else R_DT_0(lower_tail,
                                                               log_p)

    if x < 0:
        return R_DT_0(lower_tail, log_p)
    if x == float('Inf'):
        return R_DT_1(lower_tail, log_p)
    x = math.floor(x + 1e-7)
    return pbeta(prob, size, x + 1, lower_tail, log_p)

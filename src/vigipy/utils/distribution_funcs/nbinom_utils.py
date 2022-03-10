import sys
import math
import numpy
from .lgammafn import lgammafn
from .bratio import bratio
from .dpq_handling import R_DT_0, R_DT_1

M_LN2 = 0.69314718055994530942
M_LN_SQRT_2PI = math.log(math.sqrt(math.pi * 2))


def pbeta_raw(x, a, b, lower_tail, log_p=False):
    if a == 0 or b == 0 or a == float("Inf") or b == float("Inf"):
        if a == 0 and b == 0:
            return -M_LN2 if log_p else 0.5
        if a == 0 or a / b == 0:
            return R_DT_1(lower_tail, log_p)
        if b == 0 or b / a == 0:
            return R_DT_0(lower_tail, log_p)
        if x < 0.5:
            return R_DT_0(lower_tail, log_p)
        else:
            return R_DT_1(lower_tail, log_p)

    x1 = 0.5 - x + 0.5
    w, wc, ierr = bratio(a, b, x, x1, log_p)
    if ierr != 0 and ierr != 11 and ierr != 14:
        print(
            """pbeta_raw({0}, a={1}, b={2}, ..) ->
                 bratio() gave error code {3}""".format(
                x, a, b, ierr
            )
        )
        sys.exit()
    return w if lower_tail else wc


def pbeta(x, a, b, lower_tail, log_p):
    """
    The beta distribution function returning the same

    """
    if math.isnan(x) or math.isnan(a) or math.isnan(b):
        return x + a + b

    if a < 0 or b < 0:
        return numpy.nan

    if x <= 0:
        return R_DT_0(lower_tail, log_p)
    if x >= 1:
        return R_DT_1(lower_tail, log_p)

    return pbeta_raw(x, a, b, lower_tail, log_p)


def stirlerr(n):
    """
    Computes the log of the error term in Stirling's formula.
    For n > 15, uses the series 1/12n - 1/360n^3 + ...
    For n <=15, integers or half-integers, uses stored values.
    For other n < 15, uses lgamma directly

    Adapted from R code written by Catherine Loader

    """
    S0 = 1.0 / 12.0
    S1 = 1.0 / 360.0
    S2 = 1.0 / 1260.0
    S3 = 1.0 / 1680.0
    S4 = 1.0 / 1188.0

    sferr_halves = [
        0.0,  # /* n=0 - wrong, place holder only */
        0.1534264097200273452913848,  # /* 0.5 */
        0.0810614667953272582196702,  # /* 1.0 */
        0.0548141210519176538961390,  # /* 1.5 */
        0.0413406959554092940938221,  # /* 2.0 */
        0.03316287351993628748511048,  # /* 2.5 */
        0.02767792568499833914878929,  # /* 3.0 */
        0.02374616365629749597132920,  # /* 3.5 */
        0.02079067210376509311152277,  # /* 4.0 */
        0.01848845053267318523077934,  # /* 4.5 */
        0.01664469118982119216319487,  # /* 5.0 */
        0.01513497322191737887351255,  # /* 5.5 */
        0.01387612882307074799874573,  # /* 6.0 */
        0.01281046524292022692424986,  # /* 6.5 */
        0.01189670994589177009505572,  # /* 7.0 */
        0.01110455975820691732662991,  # /* 7.5 */
        0.010411265261972096497478567,  # /* 8.0 */
        0.009799416126158803298389475,  # /* 8.5 */
        0.009255462182712732917728637,  # /* 9.0 */
        0.008768700134139385462952823,  # /* 9.5 */
        0.008330563433362871256469318,  # /* 10.0 */
        0.007934114564314020547248100,  # /* 10.5 */
        0.007573675487951840794972024,  # /* 11.0 */
        0.007244554301320383179543912,  # /* 11.5 */
        0.006942840107209529865664152,  # /* 12.0 */
        0.006665247032707682442354394,  # /* 12.5 */
        0.006408994188004207068439631,  # /* 13.0 */
        0.006171712263039457647532867,  # /* 13.5 */
        0.005951370112758847735624416,  # /* 14.0 */
        0.005746216513010115682023589,  # /* 14.5 */
        0.005554733551962801371038690,  # /* 15.0 */
    ]

    if n <= 15.0:
        nn = n + n
        if nn == int(nn):
            return sferr_halves[int(nn)]
        return lgammafn(n + 1.0) - (n + 0.5) * numpy.log(n) + n - M_LN_SQRT_2PI

    nn = n * n
    if n > 500:
        return (S0 - S1 / nn) / n
    if n > 80:
        return (S0 - (S1 - S2 / nn) / nn) / n
    if n > 35:
        return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    # 15 < n <= 35 : */
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n


def bd0(x, np):
    """
    Evaluates the "deviance part"
    bd0(x,M) :=  M * D0(x/M) = M*[ x/M * log(x/M) + 1 - (x/M) ] =
            =  x * log(x/M) + M - x
    where M = E[X] = n*p (or = lambda), for	  x, M > 0

    in a manner that should be stable (with small relative error)
    for all x and M=np. In particular for x/np close to 1, direct
    evaluation fails, and evaluation is based on the Taylor series
    of log((1+v)/(1-v)) with v = (x-np)/(x+np).

        Adapted from R code written by Catherine Loader

    """
    if x == float("Inf") or np == float("Inf") or np == 0.0:
        return numpy.nan

    if math.fabs(x - np) < 0.1 * (x + np):
        v = (x - np) / (x + np)
        s = (x - np) * v
        if math.fabs(s) < sys.float_info.min:
            return s
        ej = 2 * x * v
        v = v * v
        for j in range(1000):
            ej = ej * v
            s1 = s + ej / ((j << 1) + 1)
            if s1 == s:
                return s1
            s = s1

    return x * numpy.log(x / np) + np - x

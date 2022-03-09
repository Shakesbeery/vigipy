import sys
import math
from .lgammacor import lgammacor
from .bratio import Rf_d1mach as d1mach

M_LN_SQRT_2PI = math.log(math.sqrt(math.pi * 2))
M_LN_SQRT_PId2 = math.log(math.sqrt(math.pi / 2))


def lgammafn_sign(x, sgn):
    """
    See description of lgammafn() below

    """
    xmax = 0.0
    dxrel = 0.0

    if xmax == 0:
        xmax = d1mach(2) / math.log(d1mach(2))
        dxrel = math.sqrt(d1mach(4))

    if sgn is not None:
        sgn = 1

    if math.isnan(x):
        return x

    if sgn is not None and x < 0 and math.fmod(math.floor(-x), 2.0) == 0:
        sgn = -1

    if x <= 0 and x == math.trunc(x):
        return float("Inf")

    y = math.fabs(x)

    if y < 1e-306:
        return -math.log(y)
    if y <= 10:
        return math.log(math.fabs(math.gamma(x)))

    if y > xmax:
        return float("Inf")

    if x > 0:
        if x > 1e17:
            return x * (math.log(x) - 1.0)
        elif x > 4934720.0:
            return M_LN_SQRT_2PI + (x - 0.5) * math.log(x) - x
        else:
            return M_LN_SQRT_2PI + (x - 0.5) * math.log(x) - x + lgammacor(x)
    sinpiy = math.fabs(math.sin(math.pi * y))

    ans = M_LN_SQRT_PId2 + (x - 0.5) * math.log(y) - x - math.log(sinpiy) - lgammacor(y)

    if math.fabs((x - math.trunc(x - 0.5)) * ans / x) < dxrel:
        print("The answer is less than half precision...")
        sys.exit()

    return ans


def lgammafn(x):
    """
    The function lgammafn computes log|gamma(x)|.  The function
    lgammafn_sign in addition assigns the sign of the gamma function
    to the address in the second argument if this is not NULL.

    Adapted from RMath code written by Ross Ihaka

    """
    return lgammafn_sign(x, None)

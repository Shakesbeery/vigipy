import sys
import math
from .dpq_handling import R_D__0, R_D__1


def Rf_d1mach(i):
    '''
    Function that returns constants defined in R

    '''
    if i == 1:
        return sys.float_info.min
    elif i == 2:
        return sys.float_info.max
    elif i == 3:
        return 0.5*sys.float_info.epsilon
    elif i == 4:
        return sys.float_info.epsilon
    elif i == 5:
        return math.log10(2)
    else:
        return 0.0


def bratio(a, b, x, y, log_p):
    '''
/* -----------------------------------------------------------------------
 *        Evaluation of the Incomplete Beta function I_x(a,b)
 *             --------------------
 *     It is assumed that a and b are nonnegative, and that x <= 1
 *     and y = 1 - x.  Bratio assigns w and w1 the values
 *          w  = I_x(a,b)
 *          w1 = 1 - I_x(a,b)
 *     ierr is a variable that reports the status of the results.
 *     If no input errors are detected then ierr is set to 0 and
 *     w and w1 are computed. otherwise, if an error is detected,
 *     then w and w1 are assigned the value 0 and ierr is set to
 *     one of the following values ...
 *    ierr = 1  if a or b is negative
 *    ierr = 2  if a = b = 0
 *    ierr = 3  if x < 0 or x > 1
 *    ierr = 4  if y < 0 or y > 1
 *    ierr = 5  if x + y != 1
 *    ierr = 6  if x = a = 0
 *    ierr = 7  if y = b = 0
 *    ierr = 8  (not used currently)
 *    ierr = 9  NaN in a, b, x, or y
 *    ierr = 10     (not used currently)
 *    ierr = 11  bgrat() error code 1 [+ warning in bgrat()]
 *    ierr = 12  bgrat() error code 2   (no warning here)
 *    ierr = 13  bgrat() error code 3   (no warning here)
 *    ierr = 14  bgrat() error code 4 [+ WARNING in bgrat()]
 * --------------------
 *     Written by Alfred H. Morris, Jr.
 *    Naval Surface Warfare Center
 *    Dahlgren, Virginia
 *     Revised ... Nov 1991
* ----------------------------------------------------------------------- */
'''
# /*  eps is a machine dependent constant: the smallest
# *      floating point number for which   1. + eps > 1.
# * NOTE: for almost all purposes it is replaced by 1e-15
# (~= 4.5 times larger) below */
    eps = 2. * Rf_d1mach(3)  # /* == DBL_EPSILON (in R, Rmath) */

# /* ----------------------------------------------------------------------- */
    w = R_D__0(log_p)
    w1 = R_D__0(log_p)

# ifdef IEEE_754
    # // safeguard, preventing infinite loops further down
    if math.isnan(x) or math.isnan(y) or math.isnan(a) or math.isnan(b):
        return w, w1, 9
# endif
    if (a < 0. or b < 0.):
        return w, w1, 1
    if (a == 0. and b == 0.):
        return w, w1, 2
    if (x < 0. or x > 1.):
        return w, w1, 3
    if (y < 0. or y > 1.):
        return w, w1, 4

    # /* check that  'y == 1 - x' : */
    z = x + y - 0.5 - 0.5

    if math.fabs(z) > eps * 3.:
        return w, w1, 5

    ierr = 0
    if (x == 0.):
        if a == 0:
            return w, w1, 6
    if (y == 0.):
        if b == 0:
            return w, w1, 7
    if a == 0.:
        return R_D__1(log_p), R_D__0(log_p), ierr
    if b == 0.:
        return R_D__0(log_p), R_D__1(log_p), ierr

    eps = max(eps, 1e-15)
    a_lt_b = a < b
    if (b if a_lt_b else a) < eps * .001:
        # /* procedure for a and b < 0.001 * eps */
        # // L230:  -- result *independent* of x (!)
        # // *w  = a/(a+b)  and  w1 = b/(a+b) :
        if(log_p):
            if(a_lt_b):
                w = math.log1p(-a/(a+b))  # // notably if a << b
                w1 = math.log(a/(a+b))
        else:  # // b <= a
            w = math.log(b/(a+b))
            w1 = math.log1p(-b/(a+b))
    else:
        w = b/(a + b)
        w1 = a/(a + b)

    return w, w1, ierr

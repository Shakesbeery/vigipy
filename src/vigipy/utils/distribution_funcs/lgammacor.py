import sys
import numpy as np


def lgammacor(x):
    '''
    Compute the log gamma correction factor for x >= 10 so that
    log(gamma(x)) = .5*log(2*pi) + (x-.5)*log(x) -x + lgammacor(x)
    [ lgammacor(x) is called	Del(x)	in other contexts (e.g. dcdflib)]

    Adapted from the RMath function written by Ross Ihaka

    '''
    algmcs = [
                .1666389480451863247205729650822e+0,
                -.1384948176067563840732986059135e-4,
                .9810825646924729426157171547487e-8,
                -.1809129475572494194263306266719e-10,
                .6221098041892605227126015543416e-13,
                -.3399615005417721944303330599666e-15,
                .2683181998482698748957538846666e-17,
                -.2868042435334643284144622399999e-19,
                .3962837061046434803679306666666e-21,
                -.6831888753985766870111999999999e-23,
                .1429227355942498147573333333333e-24,
                -.3547598158101070547199999999999e-26,
                .1025680058010470912000000000000e-27,
                -.3401102254316748799999999999999e-29,
                .1276642195630062933333333333333e-30
               ]

    nalgm = 5
    xbig = 94906265.62425156
    xmax = 3.745194030963158e306

    if (x < 10):
        return np.nan
    elif (x >= xmax):
        print("Underflow...")
        sys.exit()
    elif (x < xbig):
        tmp = 10 / x
        return chebyshev_eval(tmp * tmp * 2 - 1, algmcs, nalgm) / x
    return 1 / (x * 12)


def chebyshev_eval(x, a, n):
    '''
    evaluates the n-term Chebyshev series
    a at x

    '''
    if n < 1 or n > 1000:
        return np.nan

    if x < -1.1 or x > 1.1:
        return np.nan

    twox = x * 2
    b2 = 0
    b1 = 0
    b0 = 0
    for i in range(n):
        b2 = b1
        b1 = b0
        b0 = twox * b1 - b2 + a[n - i]
    return (b0 - b2) * 0.5

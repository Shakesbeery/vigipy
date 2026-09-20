import warnings
from typing import NamedTuple, Optional

import numpy as np
from scipy.special import gamma
from scipy.stats import norm, rankdata
from scipy.optimize import minimize_scalar


class LBEResult(NamedTuple):
    """Result of the Local Bayes Estimation procedure."""

    fdr: Optional[float]
    pi0: float
    icpi0: list
    ci_level: float
    a: Optional[float]
    sdbound: float
    qvalues: Optional[np.ndarray]
    pvalues: np.ndarray
    significant: Optional[np.ndarray]
    n_significant: Optional[int]


def lbe(
    pvals,
    a=None,
    lb=0.05,
    ci_level=0.95,
    qvalues=True,
    fdr_level=0.05,
    n_significant=None,
):

    if min(pvals) < 0 or max(pvals) > 1:
        raise ValueError("ERROR: p-values not in valid range.")

    else:
        m = len(pvals)
        fdr = None
        if a is not None and a < 1:
            a = None
            sdbound = np.sqrt(1 / (3 * m))
            pi0 = min(1, np.mean(pvals) * 2)
            icpi0 = [0, pi0 - norm.ppf((1 - ci_level), 0, sdbound)]

        else:
            if a is None:
                a = lbe_a(m, lb)
            sdbound = np.sqrt((1 / (gamma(a + 1)) ** 2) * ((gamma(2 * a + 1) - (gamma(a + 1)) ** 2) / m))
            pi0 = min(1, np.mean((-np.log(1 - pvals)) ** a) / gamma(a + 1))
            icpi0 = [0, min(1, pi0 - norm.ppf((1 - ci_level), 0, sdbound))]

        if qvalues:
            sort_pval = np.sort(pvals)
            rank_pval = (rankdata(pvals, method="ordinal") - 1).astype(np.intp)
            ranks = np.arange(1, m + 1)
            raw_q = (pi0 * m * sort_pval) / ranks
            qval = np.minimum.accumulate(raw_q[::-1])[::-1]
            qval = np.clip(qval, 0, 1)
            mat = np.column_stack((rank_pval, qval, sort_pval))

            if n_significant is not None:
                fdr_level = mat[n_significant, 1]
                fdr = fdr_level
            else:
                n_significant = (mat[:, 1] <= fdr_level).sum()
                try:
                    fdr = max(np.amax(mat[mat[:, 1] <= fdr_level, 1]), 0)
                except ValueError:
                    warnings.warn("No data matches the specified FDR threshold. Setting FDR to 0.")

        if sdbound > 0.5:
            warnings.warn(
                f"l = {sdbound}. A smaller value is recommended for a (or l)."
            )

        if qvalues:
            significant = qval[rank_pval] <= fdr_level
            return LBEResult(
                fdr=fdr,
                pi0=pi0,
                icpi0=icpi0,
                ci_level=ci_level,
                a=a,
                sdbound=sdbound,
                qvalues=qval[rank_pval],
                pvalues=pvals,
                significant=significant,
                n_significant=significant.sum(),
            )
        else:
            return LBEResult(
                fdr=None,
                pi0=pi0,
                icpi0=icpi0,
                ci_level=ci_level,
                a=a,
                sdbound=sdbound,
                qvalues=None,
                pvalues=pvals,
                significant=None,
                n_significant=None,
            )


def lbe_a(m, l):
    res = minimize_scalar(asearch, bounds=(0.3, 25), args=(m, l), method="bounded")
    return max(1.0, float(res.x))


def asearch(a, m, l):
    return np.abs(np.sqrt(1 / (gamma(a + 1)) ** 2 * ((gamma(2 * a + 1) - (gamma(a + 1)) ** 2) / m)) - l)

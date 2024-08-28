import numpy as np
from scipy.special import gamma
from scipy.stats import norm, rankdata
from scipy.optimize import minimize


def lbe(pvals, a=None, lb=0.05, ci_level=0.95, qvalues=True, fdr_level=0.05, n_significant=None):

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
            qval = np.empty((m,))
            sort_pval = np.sort(pvals)
            rank_pval = (rankdata(pvals) - 1).astype(np.int16)
            qval[m - 1] = (pi0 * m * sort_pval[m - 1]) / m
            for i in range(2, m + 1):
                qval[m - i] = min((pi0 * m * sort_pval[m - i]) / (m - i), qval[m - i + 1])
            mat = np.column_stack((rank_pval, qval, sort_pval))

            if n_significant is not None:
                fdr_level = mat[n_significant, 1]
                fdr = fdr_level
            else:
                n_significant = (mat[:, 1] <= fdr_level).sum()
                try:
                    fdr = max(np.amax(mat[mat[:, 1] <= fdr_level, 1]), 0)
                except ValueError:
                    print("No data matches the specified FDR threshold. Setting FDR to 0.")

        if sdbound > 0.5:
            print(
                """WARNING: l = {0}. A smaller value is
                    recommended for a (or l).""".format(
                    sdbound
                )
            )

        if qvalues:
            significant = qval[rank_pval] <= fdr_level
            r = [
                fdr,
                pi0,
                icpi0,
                ci_level,
                a,
                sdbound,
                qval[rank_pval],
                pvals,
                significant,
                (significant[significant == True]).sum(),
            ]
        else:
            r = [None, pi0, icpi0, ci_level, a, sdbound, None, pvals, None, None]

    return r


def lbe_a(m, l):
    aopt = max(1, minimize(asearch, [1], bounds=[(0.3, 25)], args=(m, l), method="CG").x)
    return aopt


def asearch(a, m, l):
    return np.abs(np.sqrt(1 / (gamma(a + 1)) ** 2 * ((gamma(2 * a + 1) - (gamma(a + 1)) ** 2) / m)) - l)

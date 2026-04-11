import numpy as np
from scipy.stats import fisher_exact, hypergeom

from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
)


def rfet(
    container,
    relative_risk=1,
    min_events=1,
    decision_metric="fdr",
    decision_thres=0.05,
    mid_pval=False,
    expected_method="mantel-haentzel",
    method_alpha=1,
):
    """
    Calculate the Reporting Fisher's Exact Test.

    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        relative_risk (int/float): The relative risk value

        min_events: The min number of AE reports to be considered a signal

        decision_metric (str): The metric used for detecting signals:
                            {fdr = false detection rate,
                            signals = number of signals,
                            rank = ranking statistic}

        decision_thres (float): The min thres value for the decision_metric

        expected_method: The method of calculating the expected counts for
                        the disproportionality analysis.

        method_alpha: If the expected_method is negative-binomial, this
                    parameter is the alpha parameter of the distribution.

    """
    d = extract_contingency_data(container, min_events, expected_method, method_alpha)
    n11, n10, n01, n00 = d["n11"], d["n10"], d["n01"], d["n00"]

    log_rfet = np.log(n11 * n00 / (n10 * n01))

    pval_fish_uni = np.empty(d["num_cell"])
    for p in range(d["num_cell"]):
        table = [[n11[p], n10[p]], [n01[p], n00[p]]]
        pval_fish_uni[p] = fisher_exact(table, alternative="greater")[1]

    if mid_pval:
        for p in range(d["num_cell"]):
            pval_fish_uni[p] -= 0.5 * hypergeom.pmf(
                n11[p], n11[p] + n10[p], n11[p] + n01[p], n10[p] + n00[p]
            )

    pval_uni = np.clip(pval_fish_uni, 0, 1)
    RankStat = pval_uni

    FDR = compute_fdr(pval_uni, d["num_cell"])
    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, "p_value", d["num_cell"]
    )

    return build_freq_result(
        d["DATA"], n11, d["expected"], RankStat,
        np.exp(log_rfet), "RFET",
        d["n1j"], d["ni1"], FDR, "p_value", num_signals,
    )

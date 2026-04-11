import numpy as np
from scipy.stats import norm

from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
)


def ror(
    container,
    relative_risk=1,
    min_events=1,
    decision_metric="fdr",
    decision_thres=0.05,
    ranking_statistic="p_value",
    expected_method="mantel-haentzel",
    method_alpha=1,
):
    """
    Calculate the reporting odds ratio.

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

        ranking_statistic (str): How to rank signals:
                            {'p_value' = posterior prob of the null hypothesis,
                            'CI' = 95% CI lower boundary}

        expected_method: The method of calculating the expected counts for
                        the disproportionality analysis.

        method_alpha: If the expected_method is negative-binomial, this
                    parameter is the alpha parameter of the distribution.

    """
    d = extract_contingency_data(container, min_events, expected_method, method_alpha)

    log_ror = np.log(d["n11"] * d["n00"] / (d["n10"] * d["n01"]))
    var_log_ror = 1.0 / d["n11"] + 1.0 / d["n10"] + 1.0 / d["n01"] + 1.0 / d["n00"]
    pval_uni = 1 - norm.cdf(log_ror, np.log(relative_risk), np.sqrt(var_log_ror))
    pval_uni = np.clip(pval_uni, 0, 1)

    FDR = compute_fdr(pval_uni, d["num_cell"])
    if ranking_statistic == "CI":
        FDR = np.empty((len(d["n11"]),))

    LB = norm.ppf(0.025, log_ror, np.sqrt(var_log_ror))
    RankStat = pval_uni if ranking_statistic == "p_value" else LB

    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, d["num_cell"]
    )

    return build_freq_result(
        d["DATA"], d["n11"], d["expected"], RankStat,
        np.exp(log_ror), "ROR",
        d["n1j"], d["ni1"], FDR, ranking_statistic, num_signals,
    )

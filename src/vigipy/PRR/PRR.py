import numpy as np
from scipy.stats import norm

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.types import DecisionMetric, RankingStatistic, ExpectedMethod
from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
)


def prr(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 1,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    ranking_statistic: RankingStatistic = "p_value",
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    fdr_threshold: float = 0.05,
) -> AnalysisResult:
    """Calculate the proportional reporting ratio."""
    d = extract_contingency_data(container, min_events, expected_method, method_alpha)

    log_prr = np.log((d["n11"] / (d["n11"] + d["n10"])) / (d["n01"] / (d["n01"] + d["n00"])))
    var_log_prr = 1 / d["n11"] - 1 / (d["n11"] + d["n10"]) + 1 / d["n01"] - 1 / (d["n01"] + d["n00"])
    pval_uni = 1 - norm.cdf(log_prr, np.log(relative_risk), np.sqrt(var_log_prr))
    pval_uni = np.clip(pval_uni, 0, 1)

    FDR = compute_fdr(pval_uni, d["num_cell"], fdr_threshold)
    if ranking_statistic == "CI":
        FDR = np.empty((len(d["n11"]),))

    LB = norm.ppf(0.025, log_prr, np.sqrt(var_log_prr))
    RankStat = pval_uni if ranking_statistic == "p_value" else LB

    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, d["num_cell"]
    )

    return build_freq_result(
        d["DATA"], d["n11"], d["expected"], RankStat,
        np.exp(log_prr), "PRR",
        d["n1j"], d["ni1"], FDR, ranking_statistic, num_signals,
    )

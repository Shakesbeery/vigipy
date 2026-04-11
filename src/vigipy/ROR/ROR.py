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


def ror(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 1,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    ranking_statistic: RankingStatistic = "p_value",
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
) -> AnalysisResult:
    """Calculate the reporting odds ratio."""
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

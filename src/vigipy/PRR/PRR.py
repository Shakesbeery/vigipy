import numpy as np
from scipy.stats import norm

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.types import DecisionMetric, FreqRankingStatistic, ExpectedMethod
from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
    build_params,
)


def prr(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 1,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    ranking_statistic: FreqRankingStatistic = "p_value",
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    fdr_threshold: float = 0.05,
    continuity_correction: bool = True,
) -> AnalysisResult:
    """Calculate the Proportional Reporting Ratio (PRR) for pharmacovigilance signal detection.

    Computes the PRR and associated variance under a log-normal approximation, deriving
    one-sided p-values and local Bayes false discovery rates (FDR).

    Parameters:
        container: A DataContainer holding event counts and marginal totals.
        relative_risk: Null hypothesis threshold for relative risk (default: 1.0).
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for identifying signals ('fdr', 'rank', or 'signals').
        decision_thres: Significance threshold applied to the decision metric.
        ranking_statistic: Metric used to rank candidate signals ('p_value' or 'CI').
        expected_method: Method for calculating expected counts ('mantel-haentzel',
            'poisson', or 'negative-binomial').
        method_alpha: Dispersion parameter when using the negative binomial expected method.
        fdr_threshold: Target FDR level for local Bayes estimation.
        continuity_correction: If True, applies Haldane-Anscombe continuity correction (+0.5)
            to 2x2 tables with zero-count cells to prevent division by zero and variance inflation.

    Returns:
        AnalysisResult containing detected signals, all evaluated pairs, signal count,
        and model parameters.
    """
    d = extract_contingency_data(
        container, min_events, expected_method, method_alpha,
        continuity_correction=continuity_correction,
    )

    log_prr = np.log((d["n11"] / (d["n11"] + d["n10"])) / (d["n01"] / (d["n01"] + d["n00"])))
    var_log_prr = 1 / d["n11"] - 1 / (d["n11"] + d["n10"]) + 1 / d["n01"] - 1 / (d["n01"] + d["n00"])
    pval_uni = 1 - norm.cdf(log_prr, np.log(relative_risk), np.sqrt(var_log_prr))
    pval_uni = np.clip(pval_uni, 0, 1)

    FDR = compute_fdr(pval_uni, d["num_cell"], fdr_threshold)

    LB = norm.ppf(0.025, log_prr, np.sqrt(var_log_prr))
    RankStat = pval_uni if ranking_statistic == "p_value" else LB

    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, d["num_cell"]
    )

    params = build_params("prr", {
        "relative_risk": relative_risk, "min_events": min_events,
        "decision_metric": decision_metric, "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic, "expected_method": expected_method,
        "method_alpha": method_alpha, "fdr_threshold": fdr_threshold,
        "continuity_correction": continuity_correction,
    })

    return build_freq_result(
        d["DATA"], d.get("n11_raw", d["n11"]), d["expected"], RankStat,
        np.exp(log_prr), "PRR",
        d["n1j"], d["ni1"], FDR, ranking_statistic, num_signals, params,
    )

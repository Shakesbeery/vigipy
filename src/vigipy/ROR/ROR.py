import numpy as np

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.types import DecisionMetric, FreqRankingStatistic, ExpectedMethod
from ..utils.common import (
    extract_contingency_data,
    compute_ratio_inference,
    build_freq_result,
    build_params,
)


def ror(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 3,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    ranking_statistic: FreqRankingStatistic = "p_value",
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    fdr_threshold: float = 0.05,
    continuity_correction: bool = True,
) -> AnalysisResult:
    """Calculate the Reporting Odds Ratio (ROR) for pharmacovigilance signal detection.

    Clinical Intuition:
        ROR mirrors the classic case-control epidemiological odds ratio: it compares the
        odds of reporting a specific adverse event for a given drug against the odds for all
        other drugs. An ROR of 2.5 indicates that the odds of this reaction being reported
        are 2.5-fold higher when exposed to the suspect drug. Woolf's log-odds standard
        error provides stable confidence intervals across medium-to-large spontaneous cohorts.

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

    log_ror = np.log(d["n11"] * d["n00"] / (d["n10"] * d["n01"]))
    var_log_ror = 1.0 / d["n11"] + 1.0 / d["n10"] + 1.0 / d["n01"] + 1.0 / d["n00"]

    rank_stat, ci_lower, ci_upper, fdr, num_signals = compute_ratio_inference(
        log_ror, var_log_ror, relative_risk, d["num_cell"], fdr_threshold,
        ranking_statistic, decision_metric, decision_thres,
    )

    params = build_params("ror", {
        "relative_risk": relative_risk, "min_events": min_events,
        "decision_metric": decision_metric, "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic, "expected_method": expected_method,
        "method_alpha": method_alpha, "fdr_threshold": fdr_threshold,
        "continuity_correction": continuity_correction,
    })

    return build_freq_result(
        d["DATA"], d.get("n11_raw", d["n11"]), d["expected"], rank_stat,
        np.exp(log_ror), "ROR",
        d["n1j"], d["ni1"], fdr, ranking_statistic, num_signals, params,
        ci_lower=ci_lower, ci_upper=ci_upper,
    )

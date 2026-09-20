import warnings

import numpy as np
from scipy.stats import hypergeom

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.types import DecisionMetric, ExpectedMethod
from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
    build_params,
)


def rfet(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 1,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    mid_pval: bool = False,
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    fdr_threshold: float = 0.05,
) -> AnalysisResult:
    """Calculate the Reporting Fisher's Exact Test (RFET) for pharmacovigilance signal detection.

    Computes exact hypergeometric p-values for 2x2 contingency tables using SciPy's
    hypergeometric survival function, optionally applying Lancaster's mid-p correction.

    Parameters:
        container: A DataContainer holding event counts and marginal totals.
        relative_risk: Deprecated parameter retained for signature compatibility.
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for identifying signals ('fdr', 'rank', or 'signals').
        decision_thres: Significance threshold applied to the decision metric.
        mid_pval: Whether to apply Lancaster mid-p adjustment (subtracts 0.5 * P(X = n11)).
        expected_method: Method for calculating expected counts ('mantel-haentzel',
            'poisson', or 'negative-binomial').
        method_alpha: Dispersion parameter when using the negative binomial expected method.
        fdr_threshold: Target FDR level for local Bayes estimation.

    Returns:
        AnalysisResult containing detected signals, all evaluated pairs, signal count,
        and model parameters.
    """
    if relative_risk != 1:
        warnings.warn(
            "relative_risk is unused in rfet() and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
    d = extract_contingency_data(container, min_events, expected_method, method_alpha)
    n11, n10, n01, n00 = d["n11"], d["n10"], d["n01"], d["n00"]
    n11_raw = d.get("n11_raw", n11)

    log_rfet = np.log(n11 * n00 / (n10 * n01))

    # One-sided Fisher's exact test: P(X >= n11)
    pval_fish_uni = hypergeom.sf(n11_raw - 1, d["N"], d["n1j"], d["ni1"])

    if mid_pval:
        # Lancaster mid-p correction
        pval_fish_uni -= 0.5 * hypergeom.pmf(n11_raw, d["N"], d["n1j"], d["ni1"])

    pval_uni = np.clip(pval_fish_uni, 0, 1)
    RankStat = pval_uni

    FDR = compute_fdr(pval_uni, d["num_cell"], fdr_threshold)
    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, "p_value", d["num_cell"]
    )

    params = build_params("rfet", {
        "min_events": min_events, "decision_metric": decision_metric,
        "decision_thres": decision_thres, "mid_pval": mid_pval,
        "expected_method": expected_method, "method_alpha": method_alpha,
        "fdr_threshold": fdr_threshold,
    })

    return build_freq_result(
        d["DATA"], n11_raw, d["expected"], RankStat,
        np.exp(log_rfet), "RFET",
        d["n1j"], d["ni1"], FDR, "p_value", num_signals, params,
    )

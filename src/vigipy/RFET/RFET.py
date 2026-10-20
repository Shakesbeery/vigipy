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
    min_events: int = 3,
    decision_metric: DecisionMetric = "fdr",
    decision_thres: float = 0.05,
    mid_pval: bool = True,
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    fdr_threshold: float = 0.05,
) -> AnalysisResult:
    """Calculate the Reporting Fisher's Exact Test (RFET) for pharmacovigilance signal detection.

    Clinical Intuition:
        When cell counts are small (e.g., fewer than 5 reports), large-sample asymptotic
        approximations (like normal Wald z-scores for PRR and ROR) break down, producing
        unstable p-values and inflated type I errors.

        The Reporting Fisher's Exact Test (RFET) calculates exact hypergeometric probabilities
        conditioned on the observed margins, making it the gold standard for rare, small-count
        adverse drug reactions. When `mid_pval=True`, Lancaster's mid-p correction is applied,
        which subtracts half the point probability of the observed count to reduce the
        inherent conservatism of classical Fisher's exact tests while strictly controlling
        false positive rates.

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
    # Guard against fractional counts produced by exponential time decay in LongitudinalModel
    N_val = d["N"]
    n1j_val = d["n1j"]
    ni1_val = d["ni1"]
    if not (
        np.all(np.equal(np.mod(n11_raw, 1.0), 0.0))
        and np.all(np.equal(np.mod(N_val, 1.0), 0.0))
        and np.all(np.equal(np.mod(n1j_val, 1.0), 0.0))
        and np.all(np.equal(np.mod(ni1_val, 1.0), 0.0))
    ):
        k_hg = np.maximum(1.0, np.rint(n11_raw))
        n_hg = np.maximum(k_hg, np.rint(n1j_val))
        N_draw = np.maximum(k_hg, np.rint(ni1_val))
        M_pop = np.maximum(n_hg + N_draw - k_hg, np.rint(N_val))
    else:
        k_hg, M_pop, n_hg, N_draw = n11_raw, N_val, n1j_val, ni1_val

    pval_fish_uni = hypergeom.sf(k_hg - 1, M_pop, n_hg, N_draw)

    if mid_pval:
        # Lancaster mid-p correction
        pval_fish_uni -= 0.5 * hypergeom.pmf(k_hg, M_pop, n_hg, N_draw)

    pval_uni = np.clip(np.nan_to_num(pval_fish_uni, nan=1.0), 0, 1)
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

"""Shared utility functions for disproportionality analysis methods."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .lbe import lbe
from .Container import AnalysisResult, DataContainer
from .expectations import calculate_expected
from .types import DecisionMetric, RankingStatistic, ExpectedMethod

# Small constant added to denominators to prevent division by zero
DIVISION_EPSILON = 1e-7


def extract_contingency_data(
    container: DataContainer,
    min_events: int,
    expected_method: ExpectedMethod,
    method_alpha: float,
) -> dict[str, Any]:
    """Extract and prepare the 2x2 contingency table components from a container.

    Returns a dict with keys: DATA, N, n11, n1j, ni1, num_cell, expected,
    n10, n01, n00.
    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)

    n10 = n1j - n11
    n01 = ni1 - n11 + DIVISION_EPSILON
    n00 = N - (n11 + n10 + n01)

    return {
        "DATA": DATA,
        "N": N,
        "n11": n11,
        "n1j": n1j,
        "ni1": ni1,
        "num_cell": num_cell,
        "expected": expected,
        "n10": n10,
        "n01": n01,
        "n00": n00,
    }


def compute_fdr(
    pval_uni: np.ndarray, num_cell: int, fdr_threshold: float = 0.05
) -> np.ndarray:
    """Compute FDR using local Bayes estimation (LBE)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = lbe(2 * np.minimum(pval_uni, 1 - pval_uni), fdr_level=fdr_threshold)
    pi_c = results.pi0

    half_mask_sum = (pval_uni <= 0.5).sum()
    fdr = pi_c * np.sort(pval_uni[pval_uni <= 0.5]) / (
        np.arange(1, half_mask_sum + 1) / num_cell
    )

    fdr = np.concatenate(
        (
            fdr,
            (
                pi_c / (2 * np.arange(half_mask_sum, num_cell) / num_cell)
                + 1
                - half_mask_sum / np.arange(half_mask_sum, num_cell)
            ),
        ),
        axis=None,
    )

    return np.minimum(fdr, np.ones((len(fdr),)))


def determine_num_signals(
    FDR: np.ndarray,
    RankStat: np.ndarray,
    decision_metric: DecisionMetric,
    decision_thres: float,
    ranking_statistic: RankingStatistic,
    num_cell: int,
) -> int:
    """Determine the number of signals based on the decision rule."""
    if decision_metric == "fdr":
        return int((FDR <= decision_thres).sum())
    elif decision_metric == "signals":
        return int(min((RankStat <= decision_thres).sum(), num_cell))
    elif decision_metric == "rank":
        if ranking_statistic == "p_value":
            return int((RankStat <= decision_thres).sum())
        else:
            return int((RankStat >= decision_thres).sum())
    return 0


def compute_bayesian_metrics(
    posterior_probability: np.ndarray, num_cell: int, ranking_statistic: RankingStatistic
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute FDR, FNR, sensitivity, and specificity from posterior probabilities.

    Used by Bayesian methods (BCPNN, GPS).
    """
    post_cumsum = np.cumsum(posterior_probability)
    post_1_cumsum = np.cumsum(1 - posterior_probability)
    post_1_sum = (1 - posterior_probability).sum()
    post_range = np.arange(1, len(posterior_probability) + 1)

    FDR = post_cumsum / post_range
    Se = np.cumsum(1 - posterior_probability) / post_1_sum

    if ranking_statistic == "p_value":
        FNR = np.array(post_1_cumsum) / ((num_cell - post_range) + DIVISION_EPSILON)
        Sp = np.array(post_cumsum) / (num_cell - post_1_sum)
    else:
        FNR = post_1_cumsum[::-1] / ((num_cell - post_range) + DIVISION_EPSILON)
        Sp = post_cumsum[::-1] / (num_cell - post_1_sum)

    return FDR, FNR, Se, Sp


def build_freq_result(
    DATA: pd.DataFrame,
    n11: np.ndarray,
    expected: np.ndarray,
    RankStat: np.ndarray,
    stat_values: np.ndarray,
    stat_column_name: str,
    n1j: np.ndarray,
    ni1: np.ndarray,
    FDR: np.ndarray,
    ranking_statistic: RankingStatistic,
    num_signals: int,
) -> AnalysisResult:
    """Build the AnalysisResult for frequentist methods (PRR, ROR, RFET)."""
    all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "Count": n11,
            "Expected Count": expected,
            "p_value": RankStat,
            stat_column_name: stat_values,
            "product margin": n1j,
            "event margin": ni1,
            "fdr": FDR,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["p_value"])

    if ranking_statistic == "CI":
        all_signals = all_signals.rename(
            columns={"p_value": "lower_bound_CI(95%)"}
        ).sort_values(by=["lower_bound_CI(95%)"])

    return AnalysisResult(
        all_signals=all_signals,
        signals=all_signals.iloc[0:num_signals],
        num_signals=num_signals,
    )

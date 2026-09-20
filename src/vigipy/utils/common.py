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


def build_params(method_name: str, input_params: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build a standardized params dict for AnalysisResult.

    All methods produce the same structure:
        {"method": "prr", "input_params": {...}, ...extra_keys}
    """
    result = {"method": method_name, "input_params": input_params}
    result.update(extra)
    return result


def extract_contingency_data(
    container: DataContainer,
    min_events: int,
    expected_method: ExpectedMethod,
    method_alpha: float,
    continuity_correction: bool = True,
) -> dict[str, Any]:
    """Extract and prepare the 2x2 contingency table components from a container.

    Applies Haldane-Anscombe continuity correction (+0.5 to all 4 cells) to any
    contingency tables containing zero-count cells to prevent division by zero and
    infinite variances.

    Returns a dict with keys: DATA, N, n11, n1j, ni1, num_cell, expected,
    n10, n01, n00, n11_raw.
    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11_raw = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11_raw)
    expected = calculate_expected(N, n1j, ni1, n11_raw, expected_method, method_alpha)

    n10_raw = n1j - n11_raw
    n01_raw = ni1 - n11_raw
    n00_raw = N - (n11_raw + n10_raw + n01_raw)

    zero_cells = (n11_raw <= 0) | (n10_raw <= 0) | (n01_raw <= 0) | (n00_raw <= 0)
    if continuity_correction and np.any(zero_cells):
        # Haldane-Anscombe continuity correction: add 0.5 to all 4 cells of tables with zeros
        adj = 0.5 * zero_cells.astype(np.float64)
        n11 = n11_raw + adj
        n10 = n10_raw + adj
        n01 = n01_raw + adj
        n00 = n00_raw + adj
    else:
        n11 = n11_raw
        n10 = n10_raw
        n01 = n01_raw + DIVISION_EPSILON
        n00 = n00_raw

    return {
        "DATA": DATA,
        "N": N,
        "n11": n11,
        "n11_raw": n11_raw,
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
    if num_cell == 0 or len(pval_uni) == 0:
        return np.empty(0, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = lbe(2 * np.minimum(pval_uni, 1 - pval_uni), fdr_level=fdr_threshold)
    pi_c = results.pi0

    sort_order = np.argsort(pval_uni)
    sorted_p = pval_uni[sort_order]

    half_mask_sum = int((sorted_p <= 0.5).sum())
    fdr_parts = []

    if half_mask_sum > 0:
        ranks_low = np.arange(1, half_mask_sum + 1) / num_cell
        fdr_low = pi_c * sorted_p[:half_mask_sum] / ranks_low
        fdr_parts.append(fdr_low)

    if half_mask_sum < num_cell:
        ranks_high = np.arange(half_mask_sum + 1, num_cell + 1) / num_cell
        fdr_high = (
            pi_c / (2 * ranks_high)
            + 1
            - (half_mask_sum / num_cell) / ranks_high
        )
        fdr_parts.append(fdr_high)

    sorted_fdr = np.concatenate(fdr_parts) if fdr_parts else np.empty(0, dtype=np.float64)
    # Enforce step-up monotonicity (backward cumulative minimum)
    if len(sorted_fdr) > 0:
        sorted_fdr = np.minimum.accumulate(sorted_fdr[::-1])[::-1]
    sorted_fdr = np.clip(sorted_fdr, 0.0, 1.0)

    # Restore original row order
    fdr = np.empty_like(sorted_fdr)
    fdr[sort_order] = sorted_fdr
    return fdr


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
    posterior_probability: np.ndarray,
    num_cell: int,
    ranking_statistic: RankingStatistic = "p_value",
    rank_stat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute FDR, FNR, sensitivity, and specificity from posterior probabilities.

    Used by Bayesian methods (BCPNN, GPS).
    Calculates cumulative metrics along the ranked sequence of candidate signals.
    """
    if num_cell == 0 or len(posterior_probability) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    if rank_stat is not None:
        if ranking_statistic == "p_value":
            sort_order = np.argsort(rank_stat)
        else:
            sort_order = np.argsort(-rank_stat)
    else:
        sort_order = np.argsort(posterior_probability)

    p_sorted = posterior_probability[sort_order]
    ranks = np.arange(1, num_cell + 1)

    # Bayesian FDR (cumulative mean posterior null probability, monotonic step-down)
    fdr_raw = np.cumsum(p_sorted) / ranks
    fdr_sorted = np.maximum.accumulate(fdr_raw)

    # Cumulative sensitivity
    signal_mass = 1.0 - p_sorted
    total_signal = signal_mass.sum()
    if total_signal > 0:
        se_sorted = np.cumsum(signal_mass) / total_signal
    else:
        se_sorted = np.zeros(num_cell, dtype=np.float64)

    # False negative rate among unselected hypotheses
    unselected_signal = total_signal - np.cumsum(signal_mass)
    remaining_cells = num_cell - ranks
    fnr_sorted = unselected_signal / (remaining_cells + DIVISION_EPSILON)

    # Specificity
    total_null = p_sorted.sum()
    unselected_null = total_null - np.cumsum(p_sorted)
    if total_null > 0:
        sp_sorted = unselected_null / total_null
    else:
        sp_sorted = np.ones(num_cell, dtype=np.float64)

    # Restore original row order
    FDR = np.empty_like(fdr_sorted)
    FNR = np.empty_like(fnr_sorted)
    Se = np.empty_like(se_sorted)
    Sp = np.empty_like(sp_sorted)

    FDR[sort_order] = fdr_sorted
    FNR[sort_order] = fnr_sorted
    Se[sort_order] = se_sorted
    Sp[sort_order] = sp_sorted

    return np.clip(FDR, 0, 1), np.clip(FNR, 0, 1), np.clip(Se, 0, 1), np.clip(Sp, 0, 1)


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
    params: dict[str, Any] | None = None,
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
        params=params,
    )

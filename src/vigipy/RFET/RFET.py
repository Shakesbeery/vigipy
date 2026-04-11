import numpy as np
from scipy.stats import fisher_exact, hypergeom

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.types import DecisionMetric, ExpectedMethod
from ..utils.common import (
    extract_contingency_data,
    compute_fdr,
    determine_num_signals,
    build_freq_result,
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
) -> AnalysisResult:
    """Calculate the Reporting Fisher's Exact Test."""
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

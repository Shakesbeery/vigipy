import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy.stats import norm

from ..utils.Container import AnalysisResult, DataContainer
from ..utils import calculate_expected
from ..utils.common import compute_bayesian_metrics, determine_num_signals, build_params
from ..utils.types import DecisionMetric, BCPNNRankingStatistic, ExpectedMethod


def bcpnn(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 1,
    decision_metric: DecisionMetric = "rank",
    decision_thres: float = 0.05,
    ranking_statistic: BCPNNRankingStatistic = "quantile",
    MC: bool = False,
    num_MC: int = 10000,
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
) -> AnalysisResult:
    """Bayesian Confidence Propagation Neural Network (BCPNN) signal detection.

    Estimates the Information Component (IC) measuring dependency between a product
    and an adverse event. Supports both closed-form analytical approximations (via
    digamma/polygamma functions) and numerical Dirichlet Monte Carlo sampling.

    Parameters:
        container: A DataContainer holding event counts and marginal totals.
        relative_risk: Null hypothesis threshold for relative risk (default: 1.0).
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for identifying signals ('rank', 'fdr', or 'signals').
        decision_thres: Significance threshold applied to the decision metric.
        ranking_statistic: Metric used to rank candidate signals ('quantile' for IC_025 or 'p_value').
        MC: If True, uses Monte Carlo Dirichlet simulation instead of analytical formulas.
        num_MC: Number of Monte Carlo draws per contingency table when MC=True.
        expected_method: Method for calculating expected counts ('mantel-haentzel',
            'poisson', or 'negative-binomial').
        method_alpha: Dispersion parameter when using the negative binomial expected method.

    Returns:
        AnalysisResult containing detected signals, all evaluated pairs, signal count,
        and model parameters.
    """
    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "decision_metric": decision_metric,
        "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic,
        "MC": MC,
        "num_MC": num_MC,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
    }

    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA.loc[DATA.events >= min_events]

    n11 = DATA["events"].to_numpy(dtype=np.float64)
    n1j = DATA["product_aes"].to_numpy(dtype=np.float64)
    ni1 = DATA["count_across_brands"].to_numpy(dtype=np.float64)
    E = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)
    num_cell = len(n11)

    if not MC:
        p1 = 1 + n1j
        p2 = 1 + N - n1j
        q1 = 1 + ni1
        q2 = 1 + N - ni1
        r1 = 1 + n11
        r2b = N - n11 - 1 + (2 + N) ** 2 / (q1 * p1)
        # Calculate the Information Criterion
        digamma_term = (
            digamma(r1) - digamma(r1 + r2b) - (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2))
        )
        IC = np.asarray((np.log(2) ** -1) * digamma_term, dtype=np.float64)
        IC_variance = np.asarray(
            (np.log(2) ** -2)
            * (
                polygamma(1, r1)
                - polygamma(1, r1 + r2b)
                + (polygamma(1, p1) - polygamma(1, p1 + p2) + polygamma(1, q1) - polygamma(1, q1 + q2))
            ),
            dtype=np.float64,
        )
        rr_threshold = np.log2(relative_risk) if relative_risk > 0 else -np.inf
        posterior_prob = norm.cdf(rr_threshold, IC, np.sqrt(IC_variance))
        lower_bound = norm.ppf(0.025, IC, np.sqrt(IC_variance))
    else:
        num_MC = float(num_MC)
        # Priors for the contingency table
        q1j = (n1j + 0.5) / (N + 1)
        qi1 = (ni1 + 0.5) / (N + 1)
        qi0 = (N - ni1 + 0.5) / (N + 1)
        q0j = (N - n1j + 0.5) / (N + 1)

        a_ = 0.5 / (q1j * qi1)

        a11 = q1j * qi1 * a_
        a10 = q1j * qi0 * a_
        a01 = q0j * qi1 * a_
        a00 = q0j * qi0 * a_

        g11 = a11 + n11
        g10 = a10 + n10
        g01 = a01 + n01
        g00 = a00 + n00

        posterior_prob = []
        lower_bound = []
        log2_scale = 1.0 / np.log(2)
        rr_threshold = np.log2(relative_risk) if relative_risk > 0 else -np.inf
        for m in range(num_cell):
            alpha = [g11[m], g10[m], g01[m], g00[m]]
            p = np.random.dirichlet(alpha, int(num_MC))
            p11 = p[:, 0]
            p1_ = p11 + p[:, 1]
            p_1 = p11 + p[:, 2]
            ic_monte = log2_scale * np.log(p11 / (p1_ * p_1))
            posterior_prob.append(float(np.mean(ic_monte < rr_threshold)))
            lower_bound.append(float(np.percentile(ic_monte, 2.5)))
        posterior_prob = np.asarray(posterior_prob, dtype=np.float64)
        lower_bound = np.asarray(lower_bound, dtype=np.float64)

    if ranking_statistic == "p_value":
        RankStat = posterior_prob
    else:
        RankStat = lower_bound

    FDR, FNR, Se, Sp = compute_bayesian_metrics(posterior_prob, num_cell, ranking_statistic, RankStat)
    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, num_cell
    )

    name = DATA["product_name"]
    ae = DATA["ae_name"]
    count = n11

    # SIGNALS RESULTS and presentation
    if ranking_statistic == "p_value":
        all_signals = pd.DataFrame(
            {
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": E,
                "p_value": RankStat,
                "count/expected": (count / E),
                "product margin": n1j,
                "event margin": ni1,
                "fdr": FDR,
                "FNR": FNR,
                "Se": Se,
                "Sp": Sp,
            }
        ).sort_values(by=[ranking_statistic])
    else:
        all_signals = pd.DataFrame(
            {
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": E,
                "quantile": RankStat,
                "count/expected": (count / E),
                "product margin": n1j,
                "event margin": ni1,
                "fdr": FDR,
                "FNR": FNR,
                "Se": Se,
                "Sp": Sp,
            }
        ).sort_values(by=[ranking_statistic], ascending=False)

    all_signals.index = np.arange(len(all_signals.index))
    signals = all_signals.iloc[0:num_signals]

    return AnalysisResult(
        all_signals=all_signals,
        signals=signals,
        num_signals=num_signals,
        params=build_params("bcpnn", input_params),
    )

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize

from ..utils.Container import AnalysisResult, DataContainer
from ..utils import calculate_expected
from ..utils.common import (
    compute_bayesian_metrics,
    determine_num_signals,
    build_params,
    build_bayesian_result,
)
from ..utils.types import DecisionMetric, GPSRankingStatistic, ExpectedMethod
from ..utils.distribution_funcs.quantile_funcs import quantiles as _quantiles_scalar

quantiles = np.vectorize(_quantiles_scalar, otypes=[np.float64])

EPS = np.finfo(np.float32).eps
BOUNDED_METHODS = {
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
}


def _optimize_gps_priors(
    container: DataContainer,
    priors: np.ndarray,
    truncate: bool,
    truncate_thres: float,
    n11: np.ndarray,
    expected: np.ndarray,
    N: int,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))

    if minimization_options is None:
        minimization_options = {}

    if not truncate:
        data_cont = container.contingency
        n1__mat = data_cont.sum(axis=1)
        n_1_mat = data_cont.sum(axis=0)
        rep = len(n_1_mat)
        n1__c = np.tile(n1__mat.values, reps=rep)
        rep = len(n1__mat)
        n_1_c = np.repeat(n_1_mat.values, repeats=rep)
        E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
        n11_c_temp = []
        for col in data_cont:
            n11_c_temp.extend(list(data_cont[col]))
        n11_c = np.asarray(n11_c_temp)

        gammaln_n11_1_c = gammaln(n11_c + 1.0)
        p_out = minimize(
            non_truncated_likelihood,
            x0=priors,
            args=(n11_c, E_c, gammaln_n11_1_c),
            options={"maxiter": 500},
            method=minimization_method,
            bounds=minimization_bounds,
            **minimization_options,
        )
    else:
        trunc = truncate_thres - 1
        n11_trunc = n11[n11 >= truncate_thres]
        E_trunc = expected[n11 >= truncate_thres]
        gammaln_n11_1 = gammaln(n11_trunc + 1.0)
        p_out = minimize(
            truncated_likelihood,
            x0=priors,
            args=(
                n11_trunc,
                E_trunc,
                trunc,
                gammaln_n11_1,
            ),
            options={"maxiter": 500},
            method=minimization_method,
            bounds=minimization_bounds,
            **minimization_options,
        )

    priors_opt = p_out.x
    if np.any(priors_opt < 0) or priors_opt[4] > 1:
        warnings.warn(
            f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors_opt}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
        )
    return priors_opt, p_out.message


def gps(
    container: DataContainer,
    relative_risk: float = 1,
    min_events: int = 3,
    decision_metric: DecisionMetric = "rank",
    decision_thres: float = 0.05,
    ranking_statistic: GPSRankingStatistic = "log2",
    truncate: bool = True,
    truncate_thres: float = 1,
    prior_init: dict[str, float] | None = None,
    prior_param: list[float] | None = None,
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1,
    minimization_method: str = "Nelder-Mead",
    minimization_bounds: tuple[tuple[float, float], ...] = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1)),
    minimization_options: dict | None = None,
) -> AnalysisResult:
    """Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS).

    Clinical Intuition:
        GPS models the true relative risk distribution across all drug-event pairs as a mixture
        of two Gamma distributions: a dominant background null component (capturing non-signals
        centered near RR=1.0) and an elevated risk component. By shrinking observed counts toward
        this empirical Bayesian mixture, GPS stabilizes high-variance small counts (preventing
        spurious alerts from 1 or 2 isolated reports) while producing robust empirical Bayes
        geometric mean (EBGM) estimates and conservative 5th percentile lower bounds (EB05).
        It is the foundational methodology behind the FDA's Empirica Signal / MGPS system.

    Parameters:
        container: A container object holding the input data, including event counts (`events`),
            product-event pairs (`product_aes`), and across-brand counts (`count_across_brands`).
        relative_risk: The threshold for relative risk used in posterior probability calculations.
        min_events: Minimum number of events required for a pair to be retained.
        decision_metric: Decision rule for signal detection ('rank', 'fdr', or 'signals').
        decision_thres: Threshold used in the decision rule to filter significant signals.
        ranking_statistic: Ranking statistic to order results ('log2', 'p_value', or 'quantile').
        truncate: Whether to truncate likelihoods below a threshold for numerical stability.
        truncate_thres: Truncation threshold for likelihood values if truncate is True.
        prior_init: Initial values for the prior distributions (alpha1, beta1, alpha2, beta2, w).
        prior_param: Manually provided prior distribution parameters. If None, estimates priors via ML.
        expected_method: Method for calculating expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter used in the expected value calculation method.
        minimization_method: Optimization algorithm for estimating prior parameters (default: 'Nelder-Mead').
        minimization_bounds: Bounds on prior parameters during optimization.
        minimization_options: Options passed directly to scipy.optimize.minimize.

    Returns:
        AnalysisResult containing detected signals, all evaluated pairs, signal count,
        and model parameters.
    """
    if prior_init is None:
        prior_init = {
            "alpha1": 0.2041,
            "beta1": 0.05816,
            "alpha2": 1.415,
            "beta2": 1.838,
            "w": 0.0969,
        }
    elif isinstance(prior_init, (list, tuple, np.ndarray)):
        prior_init = {
            "alpha1": float(prior_init[0]),
            "beta1": float(prior_init[1]),
            "alpha2": float(prior_init[2]),
            "beta2": float(prior_init[3]),
            "w": float(prior_init[4]),
        }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "decision_metric": decision_metric,
        "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic,
        "truncate": truncate,
        "truncate_thres": truncate_thres,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
        "minimization_method": minimization_method,
    }

    priors = np.asarray(
        [
            prior_init["alpha1"],
            prior_init["beta1"],
            prior_init["alpha2"],
            prior_init["beta2"],
            prior_init["w"],
        ]
    )
    DATA = container.data
    N = container.N

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    code_convergence = "User-provided priors"

    if prior_param is None:
        priors, code_convergence = _optimize_gps_priors(
            container=container,
            priors=priors,
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11,
            expected=expected,
            N=N,
            minimization_method=minimization_method,
            minimization_bounds=minimization_bounds,
            minimization_options=minimization_options,
        )
    else:
        priors = np.asarray(prior_param, dtype=np.float64)

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]
        expected = expected[n11 >= min_events]
        n1j = n1j[n11 >= min_events]
        ni1 = ni1[n11 >= min_events]
        n11 = n11[n11 >= min_events]

    num_cell = len(n11)

    # Posterior probability of the null hypothesis
    _p_post1 = np.clip(priors[1] / (priors[1] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    _p_post2 = np.clip(priors[3] / (priors[3] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gammaln_n11_1_post = gammaln(n11 + 1.0)
    r1, r2 = priors[0], priors[2]
    log_qdb1 = gammaln(n11 + r1) - gammaln_n11_1_post - gammaln(r1) + r1 * np.log(_p_post1) + n11 * np.log(1.0 - _p_post1)
    log_qdb2 = gammaln(n11 + r2) - gammaln_n11_1_post - gammaln(r2) + r2 * np.log(_p_post2) + n11 * np.log(1.0 - _p_post2)
    qdb1 = np.exp(log_qdb1)
    qdb2 = np.exp(log_qdb2)

    _qn_denom = priors[4] * qdb1 + (1 - priors[4]) * qdb2
    Qn = np.where(_qn_denom > 0, priors[4] * qdb1 / _qn_denom, priors[4])

    gd1 = gdtr(relative_risk, priors[0] + n11, np.maximum(priors[1] + expected, 1e-10))
    gd2 = gdtr(relative_risk, priors[2] + n11, np.maximum(priors[3] + expected, 1e-10))
    posterior_probability = Qn * gd1 + (1 - Qn) * gd2

    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(np.maximum(priors[1] + expected, 1e-10))
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(np.maximum(priors[3] + expected, 1e-10))
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)
    ebgm = np.power(2.0, np.asarray(EBlog2, dtype=np.float64))

    # Calculation of the Lower Bound (EB05) and Upper Bound (EB95)
    LB = quantiles(
        0.05,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )
    UB = quantiles(
        0.95,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )

    # Assignment based on ranking statistic
    if ranking_statistic == "p_value":
        RankStat = posterior_probability
    elif ranking_statistic == "quantile":
        RankStat = LB
    elif ranking_statistic == "log2":
        RankStat = np.asarray(EBlog2, dtype=np.float64)
    else:
        RankStat = np.asarray(EBlog2, dtype=np.float64)

    FDR, FNR, FOR, Se, Sp = compute_bayesian_metrics(posterior_probability, num_cell, ranking_statistic, RankStat)
    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, num_cell
    )

    params = build_params(
        "gps", input_params,
        prior_init=prior_init, prior_param=priors, convergence=code_convergence,
    )

    extra_cols = {
        "EBGM": ebgm,
        "LowerBound": LB,
        "UpperBound": UB,
    }

    return build_bayesian_result(
        DATA,
        count=n11,
        expected=expected,
        ranking_statistic=ranking_statistic,
        rank_stat=RankStat,
        posterior_probability=posterior_probability,
        n1j=n1j,
        ni1=ni1,
        FDR=FDR,
        FNR=FNR,
        FOR=FOR,
        Se=Se,
        Sp=Sp,
        num_signals=num_signals,
        params=params,
        extra_cols=extra_cols,
    )


def non_truncated_likelihood(p, n11, E, gammaln_n11_1=None):
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term = (w * dnb1 + (1.0 - w) * dnb2) + 1e-7
    return np.sum(-np.log(term))


def truncated_likelihood(p, n11, E, truncate, gammaln_n11_1=None):
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term1 = w * dnb1 + (1.0 - w) * dnb2

    if truncate == 0:
        pnb1 = np.exp(r1 * np.log(p_nb1))
        pnb2 = np.exp(r2 * np.log(p_nb2))
    else:
        pnb1 = betainc(r1, truncate + 1, p_nb1)
        pnb2 = betainc(r2, truncate + 1, p_nb2)
    term2 = 1.0 - (w * pnb1 + (1.0 - w) * pnb2)

    return np.sum(-np.log(np.maximum(term1, 1e-300)) + np.log(np.maximum(term2, 1e-7)))

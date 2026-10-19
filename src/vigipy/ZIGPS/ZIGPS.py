"""Zero-Inflated Gamma-Poisson Shrinker (ZIGPS) for Disproportionality Analysis.

References:
    - Hu et al. (2015), "Zero-inflated Poisson models for disproportionality analysis in pharmacovigilance."
    - Zhang et al. (2021), "Zero-Inflated Gamma Poisson Shrinker for adverse event signal detection."
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.special import digamma, gdtr, gammaln
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
from ..utils.distribution_funcs.quantile_funcs import quantiles

EPS = float(np.finfo(np.float32).eps)
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


def _zigps_neg_log_likelihood(
    params: np.ndarray,
    n_pos: np.ndarray,
    E_pos: np.ndarray,
    gammaln_n_pos_1: np.ndarray,
    E_zero: np.ndarray,
) -> float:
    """Negative log-likelihood of the Zero-Inflated 2-Gamma-Poisson mixture over full contingency table.

    Parameters:
        params: [alpha1, beta1, alpha2, beta2, w, pi0]
        n_pos: Observed counts for cells with n > 0
        E_pos: Expected counts for cells with n > 0
        gammaln_n_pos_1: Precomputed gammaln(n_pos + 1)
        E_zero: Expected counts for cells with n == 0
    """
    r1, b1, r2, b2, w, pi0 = (
        max(float(params[0]), EPS),
        max(float(params[1]), EPS),
        max(float(params[2]), EPS),
        max(float(params[3]), EPS),
        float(np.clip(params[4], 1e-6, 1.0 - 1e-6)),
        float(np.clip(params[5], 0.0, 1.0 - 1e-6)),
    )

    nll = 0.0

    # Contribution from positive cells (n > 0): P(C = n) = (1 - pi0) * [w * NB1(n) + (1 - w) * NB2(n)]
    if len(n_pos) > 0:
        p1_pos = np.clip(b1 / (b1 + E_pos + 1e-10), 1e-10, 1.0 - 1e-10)
        p2_pos = np.clip(b2 / (b2 + E_pos + 1e-10), 1e-10, 1.0 - 1e-10)

        log_nb1 = (
            gammaln(n_pos + r1)
            - gammaln_n_pos_1
            - gammaln(r1)
            + r1 * np.log(p1_pos)
            + n_pos * np.log(1.0 - p1_pos)
        )
        log_nb2 = (
            gammaln(n_pos + r2)
            - gammaln_n_pos_1
            - gammaln(r2)
            + r2 * np.log(p2_pos)
            + n_pos * np.log(1.0 - p2_pos)
        )
        max_log = np.maximum(log_nb1, log_nb2)
        mix_pos = w * np.exp(log_nb1 - max_log) + (1.0 - w) * np.exp(log_nb2 - max_log)
        log_prob_pos = np.log(max(1.0 - pi0, 1e-12)) + max_log + np.log(np.maximum(mix_pos, 1e-300))
        nll -= float(np.sum(log_prob_pos))

    # Contribution from zero cells (n == 0): P(C = 0) = pi0 + (1 - pi0) * [w * p1^r1 + (1 - w) * p2^r2]
    if len(E_zero) > 0:
        p1_zero = np.clip(b1 / (b1 + E_zero + 1e-10), 1e-10, 1.0 - 1e-10)
        p2_zero = np.clip(b2 / (b2 + E_zero + 1e-10), 1e-10, 1.0 - 1e-10)
        nb1_zero = np.exp(r1 * np.log(p1_zero))
        nb2_zero = np.exp(r2 * np.log(p2_zero))
        prob_zero = pi0 + (1.0 - pi0) * (w * nb1_zero + (1.0 - w) * nb2_zero)
        nll -= float(np.sum(np.log(np.maximum(prob_zero, 1e-300))))

    return nll


def _optimize_zigps_priors(
    container: DataContainer,
    priors: np.ndarray,
    n11: np.ndarray,
    expected: np.ndarray,
    N: float,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes ZIGPS hyperprior parameters [alpha1, beta1, alpha2, beta2, w, pi0]."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = (
            (EPS, 20.0),
            (EPS, 10.0),
            (EPS, 20.0),
            (EPS, 10.0),
            (1e-4, 1.0 - 1e-4),
            (0.0, 0.99),
        )

    if minimization_options is None:
        minimization_options = {}

    if container.contingency is not None:
        data_cont = container.contingency
        n1_vec = np.asarray(data_cont.sum(axis=1).values, dtype=np.float64)
        n_1_vec = np.asarray(data_cont.sum(axis=0).values, dtype=np.float64)
        E_full = (np.outer(n1_vec, n_1_vec) / max(float(N), 1.0)).ravel(order="F")
        n_full = np.asarray(data_cont.values, dtype=np.float64).ravel(order="F")
        pos_mask = n_full > 0
        n_pos = n_full[pos_mask]
        E_pos = E_full[pos_mask]
        E_zero = E_full[~pos_mask]
    else:
        pos_mask = n11 > 0
        n_pos = n11[pos_mask]
        E_pos = expected[pos_mask]
        E_zero = expected[~pos_mask]

    # Adapt initial pi0 to empirical zero fraction if default
    total_cells = len(n_pos) + len(E_zero)
    if total_cells > 0 and len(E_zero) > 0:
        emp_zero_frac = len(E_zero) / float(total_cells)
        priors = priors.copy()
        priors[5] = float(np.clip(0.5 * emp_zero_frac, 0.01, 0.90))

    gammaln_n_pos_1 = gammaln(n_pos + 1.0)
    p_out = minimize(
        _zigps_neg_log_likelihood,
        x0=priors,
        args=(n_pos, E_pos, gammaln_n_pos_1, E_zero),
        options={"maxiter": 500},
        method=minimization_method,
        bounds=minimization_bounds,
        **minimization_options,
    )
    priors_opt = np.asarray(p_out.x, dtype=np.float64)
    priors_opt[0] = max(priors_opt[0], EPS)
    priors_opt[1] = max(priors_opt[1], EPS)
    priors_opt[2] = max(priors_opt[2], EPS)
    priors_opt[3] = max(priors_opt[3], EPS)
    priors_opt[4] = float(np.clip(priors_opt[4], 1e-5, 1.0 - 1e-5))
    priors_opt[5] = float(np.clip(priors_opt[5], 0.0, 0.999))
    return priors_opt, str(p_out.message)


def zigps(
    container: DataContainer,
    relative_risk: float = 1.0,
    min_events: int = 3,
    decision_metric: DecisionMetric = "rank",
    decision_thres: float = 0.05,
    ranking_statistic: GPSRankingStatistic = "log2",
    prior_init: dict[str, float] | list[float] | None = None,
    prior_param: list[float] | None = None,
    expected_method: ExpectedMethod = "mantel-haentzel",
    method_alpha: float = 1.0,
    minimization_method: str = "L-BFGS-B",
    minimization_bounds: tuple[tuple[float, float], ...] | None = None,
    minimization_options: dict | None = None,
) -> AnalysisResult:
    """Zero-Inflated Gamma-Poisson Shrinker (ZIGPS) for pharmacovigilance signal detection.

    Clinical Intuition:
        Spontaneous reporting contingency tables are overwhelmingly sparse (>90% zeros),
        mixing random sampling zeros with structural zeros (drugs and events that are
        biologically unrelated or never co-administered). Standard MGPS/GPS either discards
        zero cells via truncation or allows excess zeros to pull the null Gamma component
        toward zero, which subsequently under-shrinks noisy low-count cells (n=1, 2).

        ZIGPS explicitly introduces a point-mass spike at zero (pi0 * delta_0) alongside the
        two-component Gamma mixture:
            pi(lambda) = pi0 * delta_0(lambda) + (1 - pi0) * [w * Gamma(a1, b1) + (1 - w) * Gamma(a2, b2)]
        By absorbing structural zeros into pi0 across the full contingency table, ZIGPS
        recovers a cleaner, better-calibrated background Gamma distribution and reduces
        false-positive shrinkage artifacts on sparse drug-event pairs.
    """
    if prior_init is None:
        prior_init_dict = {
            "alpha1": 0.25,
            "beta1": 0.10,
            "alpha2": 1.60,
            "beta2": 1.80,
            "w": 0.15,
            "pi0": 0.25,
        }
    elif isinstance(prior_init, (list, tuple, np.ndarray)):
        prior_init_dict = {
            "alpha1": float(prior_init[0]),
            "beta1": float(prior_init[1]),
            "alpha2": float(prior_init[2]),
            "beta2": float(prior_init[3]),
            "w": float(prior_init[4]),
            "pi0": float(prior_init[5]) if len(prior_init) > 5 else 0.25,
        }
    else:
        prior_init_dict = {
            "alpha1": float(prior_init.get("alpha1", 0.25)),
            "beta1": float(prior_init.get("beta1", 0.10)),
            "alpha2": float(prior_init.get("alpha2", 1.60)),
            "beta2": float(prior_init.get("beta2", 1.80)),
            "w": float(prior_init.get("w", 0.15)),
            "pi0": float(prior_init.get("pi0", 0.25)),
        }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "decision_metric": decision_metric,
        "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
        "minimization_method": minimization_method,
    }

    priors = np.asarray(
        [
            prior_init_dict["alpha1"],
            prior_init_dict["beta1"],
            prior_init_dict["alpha2"],
            prior_init_dict["beta2"],
            prior_init_dict["w"],
            prior_init_dict["pi0"],
        ],
        dtype=np.float64,
    )

    DATA = container.data
    N = container.N

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    code_convergence = "User-provided priors"

    if prior_param is None:
        priors, code_convergence = _optimize_zigps_priors(
            container=container,
            priors=priors,
            n11=n11,
            expected=expected,
            N=N,
            minimization_method=minimization_method,
            minimization_bounds=minimization_bounds,
            minimization_options=minimization_options,
        )
    else:
        priors = np.asarray(prior_param, dtype=np.float64)
        if len(priors) == 5:
            priors = np.append(priors, 0.25)

    if min_events > 1:
        keep_mask = n11 >= min_events
        DATA = DATA[DATA.events >= min_events]
        expected = expected[keep_mask]
        n1j = n1j[keep_mask]
        ni1 = ni1[keep_mask]
        n11 = n11[keep_mask]

    num_cell = len(n11)
    r1, b1, r2, b2, w, pi0 = priors[0], priors[1], priors[2], priors[3], priors[4], priors[5]

    _p_post1 = np.clip(b1 / (b1 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    _p_post2 = np.clip(b2 / (b2 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gammaln_n11_1_post = gammaln(n11 + 1.0)

    log_qdb1 = (
        gammaln(n11 + r1)
        - gammaln_n11_1_post
        - gammaln(r1)
        + r1 * np.log(_p_post1)
        + n11 * np.log(1.0 - _p_post1)
    )
    log_qdb2 = (
        gammaln(n11 + r2)
        - gammaln_n11_1_post
        - gammaln(r2)
        + r2 * np.log(_p_post2)
        + n11 * np.log(1.0 - _p_post2)
    )
    max_log_q = np.maximum(log_qdb1, log_qdb2)
    qdb1_scaled = np.exp(log_qdb1 - max_log_q)
    qdb2_scaled = np.exp(log_qdb2 - max_log_q)

    # For n > 0, point mass at lambda=0 has zero likelihood (Poisson(n > 0 | 0) = 0).
    # For fractional or zero counts (e.g., time-decayed counts near 0), incorporate spike weight:
    spike_lik_scaled = np.where(n11 <= 0.0, pi0 * np.exp(-max_log_q), 0.0)
    slab_weight = 1.0 - pi0
    denom_scaled = spike_lik_scaled + slab_weight * (w * qdb1_scaled + (1.0 - w) * qdb2_scaled)

    # Posterior probability of structural zero Q0 and slab component weight Qn
    Q0 = np.where(denom_scaled > 0, spike_lik_scaled / denom_scaled, 0.0)
    slab_denom = w * qdb1_scaled + (1.0 - w) * qdb2_scaled
    Qn = np.where(slab_denom > 0, w * qdb1_scaled / slab_denom, w)

    gd1 = gdtr(relative_risk, r1 + n11, np.maximum(b1 + expected, 1e-10))
    gd2 = gdtr(relative_risk, r2 + n11, np.maximum(b2 + expected, 1e-10))
    # Under the spike at 0, P(lambda <= relative_risk) = 1.0 for relative_risk >= 0
    posterior_probability = Q0 + (1.0 - Q0) * (Qn * gd1 + (1.0 - Qn) * gd2)

    dg1 = digamma(r1 + n11) - np.log(np.maximum(b1 + expected, 1e-10))
    dg2 = digamma(r2 + n11) - np.log(np.maximum(b2 + expected, 1e-10))
    # Slab geometric mean damped by (1 - Q0)
    EBlog2 = ((1.0 - Q0) / np.log(2.0)) * (Qn * dg1 + (1.0 - Qn) * dg2)
    ebgm = np.power(2.0, np.asarray(EBlog2, dtype=np.float64))

    LB = (1.0 - Q0) * quantiles(
        0.05,
        Qn,
        r1 + n11,
        b1 + expected,
        r2 + n11,
        b2 + expected,
    )
    UB = (1.0 - Q0) * quantiles(
        0.95,
        Qn,
        r1 + n11,
        b1 + expected,
        r2 + n11,
        b2 + expected,
    )

    if ranking_statistic == "p_value":
        RankStat = posterior_probability
    elif ranking_statistic == "quantile":
        RankStat = LB
    else:
        RankStat = np.asarray(EBlog2, dtype=np.float64)

    FDR, FNR, FOR, Se, Sp = compute_bayesian_metrics(
        posterior_probability, num_cell, ranking_statistic, RankStat
    )
    num_signals = determine_num_signals(
        FDR, RankStat, decision_metric, decision_thres, ranking_statistic, num_cell
    )

    params = build_params(
        "zigps",
        input_params,
        prior_init=prior_init_dict,
        prior_param=priors,
        structural_zero_pi0=float(pi0),
        convergence=code_convergence,
    )

    extra_cols = {
        "EBGM": ebgm,
        "LowerBound": LB,
        "UpperBound": UB,
        "StructuralZeroProb": Q0,
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

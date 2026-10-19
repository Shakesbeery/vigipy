"""Configuration dataclasses for vigipy disproportionality analysis methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .utils.types import (
    DecisionMetric,
    ExpectedMethod,
    FreqRankingStatistic,
    BCPNNRankingStatistic,
    GPSRankingStatistic,
)

EPS = float(np.finfo(np.float32).eps)
DEFAULT_GPS_BOUNDS: tuple[tuple[float, float], ...] = (
    (EPS, 20.0),
    (EPS, 10.0),
    (EPS, 20.0),
    (EPS, 10.0),
    (0.0, 1.0),
)


@dataclass(frozen=True)
class PRRConfig:
    """Configuration for Proportional Reporting Ratio analysis.

    Parameters:
        relative_risk: Threshold for relative risk (null value, typically 1.0).
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for filtering ('fdr', 'rank', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric.
        ranking_statistic: Statistic used to rank candidate signals ('p_value' or 'CI').
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
        fdr_threshold: Target FDR level for local Bayes estimation.
        continuity_correction: Apply Haldane-Anscombe correction (+0.5) to contingency tables with zero cells.
    """

    method: str = "prr"
    relative_risk: float = 1
    min_events: int = 3
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    ranking_statistic: FreqRankingStatistic = "p_value"
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    fdr_threshold: float = 0.05
    continuity_correction: bool = True


@dataclass(frozen=True)
class RORConfig:
    """Configuration for Reporting Odds Ratio analysis.

    Parameters:
        relative_risk: Threshold for relative risk (null value, typically 1.0).
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for filtering ('fdr', 'rank', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric.
        ranking_statistic: Statistic used to rank candidate signals ('p_value' or 'CI').
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
        fdr_threshold: Target FDR level for local Bayes estimation.
        continuity_correction: Apply Haldane-Anscombe correction (+0.5) to contingency tables with zero cells.
    """

    method: str = "ror"
    relative_risk: float = 1
    min_events: int = 3
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    ranking_statistic: FreqRankingStatistic = "p_value"
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    fdr_threshold: float = 0.05
    continuity_correction: bool = True


@dataclass(frozen=True)
class RFETConfig:
    """Configuration for Reporting Fisher's Exact Test analysis.

    RFET uses hypergeometric testing with p-value ranking.

    Parameters:
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for filtering ('fdr', 'rank', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric.
        mid_pval: Whether to apply Lancaster mid-p correction.
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
        fdr_threshold: Target FDR level for local Bayes estimation.
    """

    method: str = "rfet"
    min_events: int = 3
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    mid_pval: bool = True
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    fdr_threshold: float = 0.05


@dataclass(frozen=True)
class BCPNNConfig:
    """Configuration for Bayesian Confidence Propagation Neural Network.

    Parameters:
        relative_risk: Null hypothesis relative risk threshold.
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for filtering ('rank', 'fdr', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric.
        ranking_statistic: Statistic used to rank candidate signals ('quantile' or 'p_value').
        MC: Whether to use Monte Carlo Dirichlet sampling instead of analytical equations.
        num_MC: Number of Monte Carlo draws when MC=True.
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
    """

    method: str = "bcpnn"
    relative_risk: float = 1
    min_events: int = 3
    decision_metric: DecisionMetric = "rank"
    decision_thres: float = 0.0
    ranking_statistic: BCPNNRankingStatistic = "quantile"
    MC: bool = False
    num_MC: int = 10000
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1


@dataclass(frozen=True)
class GPSConfig:
    """Configuration for Multi-item Gamma Poisson Shrinkage.

    Parameters:
        relative_risk: Null hypothesis relative risk threshold.
        min_events: Minimum observed count required for an event to be retained.
        decision_metric: Decision rule for filtering ('rank', 'fdr', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric.
        ranking_statistic: Statistic used to rank candidate signals ('log2', 'p_value', or 'quantile').
        truncate: Whether to use truncated Poisson likelihood for numerical stability on sparse tables.
        truncate_thres: Truncation threshold when truncate=True.
        prior_init: Initial parameter dictionary for the bivariate Poisson mixture priors.
        prior_param: Pre-fitted prior parameters vector (alpha1, beta1, alpha2, beta2, w).
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
        minimization_method: SciPy optimization algorithm for fitting hyperpriors (e.g. 'Nelder-Mead', 'L-BFGS-B').
        minimization_bounds: Bounds for optimization variables.
        minimization_options: Solver-specific options dictionary passed to scipy.optimize.minimize.
    """

    method: str = "gps"
    relative_risk: float = 1
    min_events: int = 3
    decision_metric: DecisionMetric = "rank"
    decision_thres: float = 0.05
    ranking_statistic: GPSRankingStatistic = "log2"
    truncate: bool = True
    truncate_thres: float = 1
    prior_init: dict | None = None
    prior_param: list | None = None
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    minimization_method: str = "Nelder-Mead"
    minimization_bounds: tuple[tuple[float, float], ...] | None = DEFAULT_GPS_BOUNDS
    minimization_options: dict | None = None


@dataclass(frozen=True)
class LASSOConfig:
    """Configuration for LASSO regression signal detection.

    Parameters:
        lasso_thresh: Minimum non-zero coefficient magnitude required for signal reporting.
        alpha: Regularization strength parameter.
        min_events: Minimum event occurrences to evaluate.
        num_bootstrap: Number of bootstrap iterations for coefficient confidence intervals.
        ci: Percentile confidence interval (e.g. 95).
        use_lars: Whether to use Least Angle Regression (LassoLars).
        use_IC: Whether to select penalty via Information Criterion (LassoLarsIC).
        IC_criterion: Information criterion choice ('aic' or 'bic').
        lasso_kwargs: Additional arguments forwarded to the scikit-learn estimator.
        use_glm: Whether to use Statsmodels L1-penalized Negative Binomial GLM instead of linear LASSO.
        nb_alpha: Dispersion parameter for Negative Binomial GLM.
        lasso_alpha: ElasticNet penalty parameter for GLM fitting.
        relaxed: When True and family='logistic', applies two-stage Relaxed LASSO: L1 screening
            followed by unpenalized refit on active features. Produces debiased coefficients and
            more accurate aROR estimates. Default True.
        n_jobs: Number of parallel jobs for fitting across adverse events. Use -1 for all CPUs,
            1 for sequential (default). Only used when family='logistic'.
    """

    method: str = "lasso"
    lasso_thresh: float = 0
    alpha: float = 0.5
    min_events: int = 3
    num_bootstrap: int = 10
    ci: int = 95
    use_lars: bool = False
    use_IC: bool = False
    IC_criterion: str = "bic"
    lasso_kwargs: dict | None = None
    use_glm: bool = False
    nb_alpha: float = 1
    lasso_alpha: float = 1e-9
    family: str = "logistic"
    C: float = 1.0
    decision_metric: str = "lower_bound"
    use_cv: bool = False
    cv: int = 3
    use_bootstrap: bool = False
    relaxed: bool = True
    n_jobs: int = 1


@dataclass(frozen=True)
class SCOREConfig:
    """Configuration for SCORE-DA (Syndromic Cellwise Outlier & Residual Estimation).

    Parameters:
        latent_rank: Number of latent factors for background indication & drug class absorption.
        syndromic_weight: Graph Laplacian coupling penalty (lambda_2 >= 0).
        sparsity_param: L1 sparsity penalty (lambda_1 >= 0) on the excess signal rate.
        fdr_threshold: Target False Discovery Rate (q-value) cutoff for signal detection.
        deflate_iterations: Number of iterative deflation passes to remove masking/blockbuster bias.
        min_events: Minimum observed event count required to qualify as a signal.
        max_iter: Maximum number of FISTA iterations per drug.
        tol: Convergence tolerance for FISTA.
        n_jobs: Number of CPU worker processes.
        seed: Random seed for reproducibility.
    """

    method: str = "score"
    latent_rank: int = 5
    syndromic_weight: float = 0.5
    sparsity_param: float = 1.0
    fdr_threshold: float = 0.05
    deflate_iterations: int = 2
    min_events: int = 1
    max_iter: int = 50
    tol: float = 1e-4
    n_jobs: int = 1
    seed: int = 42


@dataclass(frozen=True)
class SCOREDDIConfig:
    """Configuration for SCORE-DDI (Drug-Drug & Multi-Drug Interaction Discovery).

    Parameters:
        interaction_model: Null model for expected count under no interaction ('multiplicative' or 'additive').
        syndromic_weight: Graph Laplacian coupling penalty (lambda_2 >= 0) across adverse events.
        sparsity_param: L1 sparsity penalty (lambda_1 >= 0) on the synergistic excess rate.
        fdr_threshold: Target False Discovery Rate (q-value) cutoff for interaction signal detection.
        min_events: Minimum observed co-occurrence count required to qualify as an interaction signal.
        max_iter: Maximum number of FISTA iterations per drug pair.
        tol: Convergence tolerance for FISTA.
        n_jobs: Number of CPU worker processes (-1 for all available cores).
        seed: Random seed for reproducibility.
    """

    method: str = "score_ddi"
    interaction_model: str = "multiplicative"
    syndromic_weight: float = 0.5
    sparsity_param: float = 1.0
    fdr_threshold: float = 0.05
    min_events: int = 1
    max_iter: int = 50
    tol: float = 1e-4
    n_jobs: int = 1
    seed: int = 42


@dataclass(frozen=True)
class ZIGPSConfig:
    """Configuration for Zero-Inflated Gamma-Poisson Shrinker (ZIGPS).

    Parameters:
        relative_risk: Null hypothesis relative risk threshold (default: 1.0).
        min_events: Minimum observed count required for a pair to be retained (default: 3).
        decision_metric: Decision rule for filtering ('rank', 'fdr', or 'signals').
        decision_thres: Cutoff threshold applied to the decision metric (default: 0.05).
        ranking_statistic: Statistic used to rank candidate signals ('log2', 'p_value', or 'quantile').
        prior_init: Initial parameters for the zero-inflated 2-Gamma mixture (alpha1, beta1, alpha2, beta2, w, pi0).
        prior_param: Pre-fitted prior parameters vector.
        expected_method: Calculation method for expected counts ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative binomial expected count model.
        minimization_method: SciPy optimization algorithm for fitting hyperpriors (default: 'L-BFGS-B').
        minimization_bounds: Bounds for optimization variables.
        minimization_options: Solver-specific options passed to scipy.optimize.minimize.
    """

    method: str = "zigps"
    relative_risk: float = 1.0
    min_events: int = 3
    decision_metric: DecisionMetric = "rank"
    decision_thres: float = 0.05
    ranking_statistic: GPSRankingStatistic = "log2"
    prior_init: dict | None = None
    prior_param: list | None = None
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1.0
    minimization_method: str = "L-BFGS-B"
    minimization_bounds: tuple[tuple[float, float], ...] | None = None
    minimization_options: dict | None = None


@dataclass(frozen=True)
class MaxSPRTConfig:
    """Configuration for Kulldorff's Maximized Sequential Probability Ratio Test (MaxSPRT).

    Parameters:
        alpha: Overall sequential Type I error spending level (default: 0.05).
        min_events: Minimum observed count required to trigger a signal (default: 3).
        upper_limit: Expected count surveillance horizon U for continuous boundary calibration (default: 50.0).
        model: Likelihood model ('poisson' or 'binomial').
        expected_method: Method for expected count calculation ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative-binomial expected method.
        critical_value: Optional manual LLR critical value override.
    """

    method: str = "maxsprt"
    alpha: float = 0.05
    min_events: int = 3
    upper_limit: float = 50.0
    model: str = "poisson"
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1.0
    critical_value: float | None = None


@dataclass(frozen=True)
class CUSUMConfig:
    """Configuration for Page's Poisson Cumulative Sum (CUSUM) Surveillance.

    Parameters:
        target_rr: Target alternative relative risk R_1 > 1 to detect (default: 2.0).
        arl0: Target in-control Average Run Length ARL_0 under the null (default: 500.0).
        min_events: Minimum observed count required to trigger a signal (default: 3).
        expected_method: Method for expected count calculation ('mantel-haentzel', 'poisson', 'negative-binomial').
        method_alpha: Dispersion parameter for negative-binomial expected method.
        control_limit: Optional manual CUSUM control limit h override.
    """

    method: str = "cusum"
    target_rr: float = 2.0
    arl0: float = 500.0
    min_events: int = 3
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1.0
    control_limit: float | None = None


@dataclass(frozen=True)
class AEGISConfig:
    """Configuration for AEGIS (Anytime Empirical Graph-Informed Shrinkage).

    Parameters:
        min_events: Minimum observed count required to flag a signal (default: 3).
        fdr_threshold: Target FDR / e-BH threshold for signal detection (default: 0.05).
        lb_threshold: Minimum 5th-percentile Context-Gated Empirical Bayes lower bound (default: 1.0).
        decision_metric: Decision rule ('combined', 'lb', 'fdr', or 'e_value').
        null_rr: Null relative risk threshold for posterior probability calculation (default: 1.0).
        huber_delta: Huber-Poisson outlier clipping threshold for Sinkhorn unmasking (default: 2.0).
        sinkhorn_iters: Number of Sinkhorn-Huber unmasking iterations (default: 5).
        latent_rank: Rank of SVD residual baseline absorption for drug-class confounding (default: 2).
        min_npmi: Minimum positive NPMI edge weight to retain in the syndrome graph (default: 0.05).
        syndromic_gain: Logistic gate sensitivity to syndromic coherence (default: 1.50).
        syndromic_coupling: Conjugate Poisson-Gamma neighborhood precision pooling weight (default: 0.60).
        velocity_gain: Logistic gate sensitivity to Dual-EWMA surge velocity (default: 0.85).
        fast_window_fraction: Effective fraction of recent reports weighted in Dual-EWMA velocity (default: 0.25).
        seed: Random seed for reproducibility (default: 42).
    """

    method: str = "aegis"
    min_events: int = 3
    fdr_threshold: float = 0.05
    lb_threshold: float = 1.0
    decision_metric: str = "combined"
    null_rr: float = 1.0
    huber_delta: float = 2.0
    sinkhorn_iters: int = 5
    latent_rank: int = 2
    min_npmi: float = 0.05
    syndromic_gain: float = 1.50
    syndromic_coupling: float = 0.60
    velocity_gain: float = 0.85
    fast_window_fraction: float = 0.25
    seed: int = 42


MethodConfig = Union[
    PRRConfig,
    RORConfig,
    RFETConfig,
    BCPNNConfig,
    GPSConfig,
    ZIGPSConfig,
    MaxSPRTConfig,
    CUSUMConfig,
    LASSOConfig,
    SCOREConfig,
    SCOREDDIConfig,
    AEGISConfig,
]



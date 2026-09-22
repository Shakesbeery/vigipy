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

    method: str = field(default="prr", init=False)
    relative_risk: float = 1
    min_events: int = 1
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

    method: str = field(default="ror", init=False)
    relative_risk: float = 1
    min_events: int = 1
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

    method: str = field(default="rfet", init=False)
    min_events: int = 1
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    mid_pval: bool = False
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

    method: str = field(default="bcpnn", init=False)
    relative_risk: float = 1
    min_events: int = 1
    decision_metric: DecisionMetric = "rank"
    decision_thres: float = 0.05
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

    method: str = field(default="gps", init=False)
    relative_risk: float = 1
    min_events: int = 1
    decision_metric: DecisionMetric = "rank"
    decision_thres: float = 0.05
    ranking_statistic: GPSRankingStatistic = "log2"
    truncate: bool = False
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
    """

    method: str = field(default="lasso", init=False)
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


MethodConfig = Union[PRRConfig, RORConfig, RFETConfig, BCPNNConfig, GPSConfig, LASSOConfig]


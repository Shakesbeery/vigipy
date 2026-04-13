"""Configuration dataclasses for vigipy disproportionality analysis methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .utils.types import (
    DecisionMetric,
    ExpectedMethod,
    FreqRankingStatistic,
    BCPNNRankingStatistic,
    GPSRankingStatistic,
)


@dataclass(frozen=True)
class PRRConfig:
    """Configuration for Proportional Reporting Ratio analysis."""

    method: str = field(default="prr", init=False)
    relative_risk: float = 1
    min_events: int = 1
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    ranking_statistic: FreqRankingStatistic = "p_value"
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    fdr_threshold: float = 0.05


@dataclass(frozen=True)
class RORConfig:
    """Configuration for Reporting Odds Ratio analysis."""

    method: str = field(default="ror", init=False)
    relative_risk: float = 1
    min_events: int = 1
    decision_metric: DecisionMetric = "fdr"
    decision_thres: float = 0.05
    ranking_statistic: FreqRankingStatistic = "p_value"
    expected_method: ExpectedMethod = "mantel-haentzel"
    method_alpha: float = 1
    fdr_threshold: float = 0.05


@dataclass(frozen=True)
class RFETConfig:
    """Configuration for Reporting Fisher's Exact Test analysis.

    Note: RFET always uses p_value ranking and does not accept relative_risk.
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
    """Configuration for Bayesian Confidence Propagation Neural Network."""

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
    """Configuration for Multi-item Gamma Poisson Shrinkage."""

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
    minimization_bounds: tuple | None = None
    minimization_options: dict | None = None


@dataclass(frozen=True)
class LASSOConfig:
    """Configuration for LASSO regression signal detection."""

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

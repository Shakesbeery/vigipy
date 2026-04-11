"""Type aliases for vigipy disproportionality analysis methods."""

from typing import Literal

DecisionMetric = Literal["fdr", "signals", "rank"]
RankingStatistic = Literal["p_value", "CI", "quantile", "log2"]
ExpectedMethod = Literal["mantel-haentzel", "negative-binomial", "poisson"]

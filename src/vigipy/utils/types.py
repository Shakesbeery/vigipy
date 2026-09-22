"""Type aliases for vigipy disproportionality analysis methods."""

from typing import Literal

DecisionMetric = Literal["fdr", "signals", "rank"]
RankingStatistic = Literal["p_value", "CI", "quantile", "log2"]
ExpectedMethod = Literal["mantel-haentzel", "negative-binomial", "poisson"]

# Per-method ranking statistic types (narrower than RankingStatistic)
FreqRankingStatistic = Literal["p_value", "CI"]
BCPNNRankingStatistic = Literal["p_value", "quantile"]
GPSRankingStatistic = Literal["p_value", "quantile", "log2"]

MethodName = Literal["prr", "ror", "rfet", "bcpnn", "gps", "lasso"]

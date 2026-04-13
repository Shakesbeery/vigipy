"""Unified interface for running vigipy disproportionality analyses."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import (
    MethodConfig,
    PRRConfig,
    RORConfig,
    RFETConfig,
    BCPNNConfig,
    GPSConfig,
    LASSOConfig,
)
from .utils.Container import AnalysisResult, DataContainer
from .utils.types import MethodName

_METHOD_REGISTRY: dict[str, type] = {
    "prr": PRRConfig,
    "ror": RORConfig,
    "rfet": RFETConfig,
    "bcpnn": BCPNNConfig,
    "gps": GPSConfig,
    "lasso": LASSOConfig,
}


def analyze(container: DataContainer, config: MethodConfig) -> AnalysisResult:
    """Run a disproportionality analysis using the given configuration.

    This is the unified entry point for all vigipy methods. It dispatches
    to the underlying function based on the config's method attribute.

    Examples::

        from vigipy import analyze, PRRConfig, BCPNNConfig
        result = analyze(data, PRRConfig(min_events=3))
        result = analyze(data, BCPNNConfig(ranking_statistic="quantile"))

        # Loop over methods:
        for cfg in [PRRConfig(), RORConfig(), BCPNNConfig()]:
            result = analyze(data, cfg)
            print(f"{cfg.method}: {result.num_signals} signals")
    """
    params = asdict(config)
    method_name = params.pop("method")

    if method_name == "prr":
        from .PRR import prr
        return prr(container, **params)
    elif method_name == "ror":
        from .ROR import ror
        return ror(container, **params)
    elif method_name == "rfet":
        from .RFET import rfet
        return rfet(container, **params)
    elif method_name == "bcpnn":
        from .BCPNN import bcpnn
        return bcpnn(container, **params)
    elif method_name == "gps":
        from .GPS import gps
        return gps(container, **params)
    elif method_name == "lasso":
        from .LASSO import lasso
        return lasso(container, **params)
    else:
        raise ValueError(f"Unknown method: {method_name!r}")


def analyze_all(
    container: DataContainer,
    configs: list[MethodConfig] | None = None,
    **shared_overrides: Any,
) -> dict[str, AnalysisResult]:
    """Run multiple analyses and return results keyed by method name.

    If configs is None, runs PRR, ROR, RFET, BCPNN, GPS with defaults.
    LASSO is excluded by default because it requires binary data.

    shared_overrides are applied to all configs (e.g., min_events=3).
    """
    if configs is None:
        configs = [PRRConfig(), RORConfig(), RFETConfig(), BCPNNConfig(), GPSConfig()]

    results = {}
    for config in configs:
        if shared_overrides:
            params = asdict(config)
            params.pop("method")
            for k, v in shared_overrides.items():
                if k in params:
                    params[k] = v
            config = _METHOD_REGISTRY[config.method](**params)
        results[config.method] = analyze(container, config)
    return results


def get_default_config(method: MethodName) -> MethodConfig:
    """Return the default configuration for a given method name."""
    if method not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method!r}. Choose from {list(_METHOD_REGISTRY)}")
    return _METHOD_REGISTRY[method]()

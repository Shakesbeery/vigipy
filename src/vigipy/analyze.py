"""Unified interface for running vigipy disproportionality analyses."""

from __future__ import annotations

from typing import Any, Callable

from .config import (
    BaseConfig,
    MethodConfig,
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
)
from .utils.Container import AnalysisResult, DataContainer
from .utils.types import MethodName

from .PRR import prr
from .ROR import ror
from .RFET import rfet
from .BCPNN import bcpnn
from .GPS import gps
from .ZIGPS import zigps
from .Sequential import maxsprt, cusum_da
from .LASSO import lasso
from .SCORE import score_da, score_ddi
from .AEGIS import aegis

_METHOD_REGISTRY: dict[str, type[BaseConfig]] = {
    "prr": PRRConfig,
    "ror": RORConfig,
    "rfet": RFETConfig,
    "bcpnn": BCPNNConfig,
    "gps": GPSConfig,
    "zigps": ZIGPSConfig,
    "maxsprt": MaxSPRTConfig,
    "cusum": CUSUMConfig,
    "lasso": LASSOConfig,
    "score": SCOREConfig,
    "score_ddi": SCOREDDIConfig,
    "aegis": AEGISConfig,
}

_METHOD_DISPATCH: dict[str, Callable[..., AnalysisResult]] = {
    "prr": prr,
    "ror": ror,
    "rfet": rfet,
    "bcpnn": bcpnn,
    "gps": gps,
    "zigps": zigps,
    "maxsprt": maxsprt,
    "cusum": cusum_da,
    "lasso": lasso,
    "score": score_da,
    "score_ddi": score_ddi,
    "aegis": aegis,
}


def analyze(
    container: DataContainer,
    config: MethodConfig | str,
    **overrides: Any,
) -> AnalysisResult:
    """Run a disproportionality analysis using a configuration instance or method name.

    Parameters:
        container: A DataContainer holding contingency or binary report data.
        config: A method configuration dataclass instance (e.g. `PRRConfig()`)
            or a method name string ('prr', 'ror', 'gps', 'zigps', 'aegis', etc.).
        **overrides: Optional parameter overrides passed directly to the method.

    Returns:
        AnalysisResult containing all drug-event pairs, detected signals, and run parameters.

    Examples::

        import vigipy as vg

        # Using a configuration object:
        result = vg.analyze(data, vg.PRRConfig(min_events=3))

        # Using a method name string with overrides:
        result = vg.analyze(data, "prr", min_events=3)
    """
    if isinstance(config, str):
        method_key = config.lower()
        if method_key not in _METHOD_REGISTRY:
            raise ValueError(f"Unknown method {config!r}. Choose from {list(_METHOD_REGISTRY)}")
        cfg = _METHOD_REGISTRY[method_key](**overrides)
    elif isinstance(config, BaseConfig):
        cfg = config.replace(**overrides) if overrides else config
    else:
        if hasattr(config, "method"):
            cfg = config
        else:
            raise TypeError(f"Expected MethodConfig instance or method name string, got {type(config)}")

    method_name = cfg.method
    fn = _METHOD_DISPATCH.get(method_name)
    if fn is None:
        raise ValueError(f"Unknown method: {method_name!r}")

    params = cfg.to_kwargs() if isinstance(cfg, BaseConfig) else {k: v for k, v in vars(cfg).items() if k != "method"}
    return fn(container, **params)


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

    results: dict[str, AnalysisResult] = {}
    for config in configs:
        if shared_overrides:
            matching = {k: v for k, v in shared_overrides.items() if hasattr(config, k)}
            if matching:
                config = config.replace(**matching) if isinstance(config, BaseConfig) else config
        results[config.method] = analyze(container, config)
    return results


def get_default_config(method: MethodName) -> MethodConfig:
    """Return the default configuration for a given method name."""
    if method not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method!r}. Choose from {list(_METHOD_REGISTRY)}")
    return _METHOD_REGISTRY[method]()

"""Tests for architectural improvements, universal container interoperability,
BaseConfig contract, table-driven dispatch, and standardized AnalysisResult reporting.
"""

from __future__ import annotations

import inspect
import os
import numpy as np
import pandas as pd
import pytest

import vigipy as vg
from vigipy.config import (
    BaseConfig,
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


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Sample raw safety reporting data frame."""
    return pd.DataFrame({
        "name": ["DrugA", "DrugA", "DrugB", "DrugB", "DrugC", "DrugA", "DrugB"],
        "AE": ["Nausea", "Headache", "Nausea", "Rash", "Headache", "Nausea", "Nausea"],
        "count": [10, 5, 2, 8, 3, 4, 1],
    })


@pytest.fixture
def binary_container(sample_raw_df: pd.DataFrame) -> vg.DataContainer:
    """Binary DataContainer created via convert_binary."""
    return vg.convert_binary(sample_raw_df)


@pytest.fixture
def contingency_container(sample_raw_df: pd.DataFrame) -> vg.DataContainer:
    """Contingency DataContainer created via convert."""
    return vg.convert(sample_raw_df)


class TestBaseConfigContract:
    """Verify that all configuration dataclasses adhere to the BaseConfig contract."""

    CONFIG_CLASSES = [
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

    def test_all_configs_inherit_from_base_config(self):
        for cls in self.CONFIG_CLASSES:
            assert issubclass(cls, BaseConfig), f"{cls.__name__} does not inherit from BaseConfig"

    def test_to_kwargs_omits_method_key(self):
        for cls in self.CONFIG_CLASSES:
            cfg = cls()
            kwargs = cfg.to_kwargs()
            assert "method" not in kwargs, f"{cls.__name__}.to_kwargs() includes 'method'"
            assert len(kwargs) > 0

    def test_replace_creates_modified_instance(self):
        cfg = PRRConfig(min_events=3, relative_risk=1.0)
        updated = cfg.replace(min_events=10, relative_risk=2.0)
        assert isinstance(updated, PRRConfig)
        assert updated.min_events == 10
        assert updated.relative_risk == 2.0
        # Original remains immutable
        assert cfg.min_events == 3
        assert cfg.relative_risk == 1.0


class TestConfigFunctionParity:
    """Verify that every field in each MethodConfig is accepted by its corresponding function."""

    METHOD_MAPPINGS = [
        (PRRConfig, vg.prr),
        (RORConfig, vg.ror),
        (RFETConfig, vg.rfet),
        (BCPNNConfig, vg.bcpnn),
        (GPSConfig, vg.gps),
        (ZIGPSConfig, vg.zigps),
        (MaxSPRTConfig, vg.maxsprt),
        (CUSUMConfig, vg.cusum_da),
        (LASSOConfig, vg.lasso),
        (SCOREConfig, vg.score_da),
        (SCOREDDIConfig, vg.score_ddi),
        (AEGISConfig, vg.aegis),
    ]

    def test_all_config_fields_accepted_by_function(self):
        for config_cls, fn in self.METHOD_MAPPINGS:
            cfg = config_cls()
            kwargs = cfg.to_kwargs()
            sig = inspect.signature(fn)
            fn_params = set(sig.parameters.keys())

            for param_name in kwargs.keys():
                assert param_name in fn_params, (
                    f"Parameter {param_name!r} from {config_cls.__name__} "
                    f"not in function {fn.__name__} signature: {sorted(fn_params)}"
                )


class TestAnalyzeDispatch:
    """Test unified analyze() API with both config objects and string method names."""

    def test_analyze_with_config_instance(self, contingency_container):
        result = vg.analyze(contingency_container, PRRConfig(min_events=2))
        assert isinstance(result, vg.AnalysisResult)
        assert result.method == "prr"

    def test_analyze_with_string_method(self, contingency_container):
        result = vg.analyze(contingency_container, "prr", min_events=2)
        assert isinstance(result, vg.AnalysisResult)
        assert result.method == "prr"

    def test_analyze_with_invalid_string_raises(self, contingency_container):
        with pytest.raises(ValueError, match="Unknown method 'invalid_method'"):
            vg.analyze(contingency_container, "invalid_method")

    def test_analyze_all_with_shared_overrides(self, contingency_container):
        results = vg.analyze_all(contingency_container, [PRRConfig(), RORConfig()], min_events=2)
        assert "prr" in results
        assert "ror" in results
        assert results["prr"].params["input_params"]["min_events"] == 2
        assert results["ror"].params["input_params"]["min_events"] == 2


class TestUniversalContainerInteroperability:
    """Test that binary containers work with classical contingency-based methods."""

    def test_binary_container_attributes(self, binary_container):
        assert binary_container.products is not None
        assert binary_container.events is not None
        assert binary_container.num_products > 0
        assert binary_container.num_events > 0
        assert binary_container.shape == (binary_container.num_products, binary_container.num_events)
        assert "DataContainer" in repr(binary_container)

    def test_prr_on_binary_container_lazy_materialization(self, binary_container):
        # Previously, this failed with KeyError: 'events'
        result = vg.prr(binary_container, min_events=1)
        assert isinstance(result, vg.AnalysisResult)
        assert len(result.all_signals) > 0

    def test_gps_on_binary_container_lazy_materialization(self, binary_container):
        result = vg.gps(binary_container, min_events=1)
        assert isinstance(result, vg.AnalysisResult)
        assert len(result.all_signals) > 0

    def test_consensus_analysis_on_binary_container(self, binary_container):
        cr = vg.consensus_analysis(
            binary_container,
            configs=[PRRConfig(min_events=1), RORConfig(min_events=1)],
            min_agreements=1,
        )
        assert isinstance(cr, vg.ConsensusResult)
        assert len(cr.comparison_table) > 0


class TestStandardizedReporting:
    """Test AnalysisResult.summary() and inspect_pair()."""

    def test_summary_standardized_columns(self, contingency_container):
        result = vg.prr(contingency_container, min_events=1)
        summary_df = result.summary(which="all")

        expected_columns = [
            "Product",
            "Adverse Event",
            "Count",
            "Expected Count",
            "Metric",
            "Score",
            "CI Lower",
            "CI Upper",
            "p_value",
            "fdr",
        ]
        assert list(summary_df.columns) == expected_columns
        assert len(summary_df) == len(result.all_signals)

    def test_inspect_pair_returns_series_with_is_signal(self, contingency_container):
        result = vg.prr(contingency_container, min_events=1)
        first_row = result.all_signals.iloc[0]
        prod = first_row["Product"]
        ae = first_row["Adverse Event"]

        series = result.inspect_pair(prod, ae)
        assert isinstance(series, pd.Series)
        assert "is_signal" in series
        assert series["Product"] == prod
        assert series["Adverse Event"] == ae


def test_py_typed_file_exists():
    """Verify PEP 561 compliance marker."""
    import vigipy
    package_dir = os.path.dirname(vigipy.__file__)
    py_typed_path = os.path.join(package_dir, "py.typed")
    assert os.path.isfile(py_typed_path), "py.typed marker file missing in vigipy package root"

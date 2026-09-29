"""Comprehensive tests for the high-priority mathematical and architectural enhancements."""

import os
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest

from vigipy import (
    prr, ror, rfet, gps, bcpnn, lasso, score_da, score_ddi,
    convert, convert_binary, convert_ddi, consensus_analysis,
    logger,
    PRRConfig, RORConfig, RFETConfig, BCPNNConfig, GPSConfig, LASSOConfig,
    SCOREConfig, SCOREDDIConfig,
)
from vigipy.utils.distribution_funcs.quantile_funcs import quantiles
from vigipy.utils.lbe import lbe
import vigipy.utils.data_prep as dp_mod


class TestInvertedCISorting:
    def test_prr_ci_sorting_descending(self, converted_data):
        """Verify ranking_statistic='CI' sorts by lower_bound_CI descending."""
        result = prr(
            converted_data,
            min_events=3,
            ranking_statistic="CI",
            decision_metric="signals",
            decision_thres=10,
        )
        lb_col = "lower_bound_CI(95%)"
        assert lb_col in result.all_signals.columns
        values = result.all_signals[lb_col].values
        # Must be monotonically non-increasing (descending)
        assert np.all(values[:-1] >= values[1:] - 1e-9), "CI lower bounds are not sorted descending!"
        # Top signals must be the strongest
        assert values[0] > 1.0
        # Check that lower_bound_CI matches CI Lower
        np.testing.assert_allclose(values, result.all_signals["CI Lower"].values, rtol=1e-5)

    def test_ror_ci_sorting_descending(self, converted_data):
        """Verify ranking_statistic='CI' in ROR sorts descending."""
        result = ror(
            converted_data,
            min_events=3,
            ranking_statistic="CI",
            decision_metric="signals",
            decision_thres=10,
        )
        lb_col = "lower_bound_CI(95%)"
        assert lb_col in result.all_signals.columns
        values = result.all_signals[lb_col].values
        assert np.all(values[:-1] >= values[1:] - 1e-9), "ROR CI lower bounds are not sorted descending!"
        np.testing.assert_allclose(values, result.all_signals["CI Lower"].values, rtol=1e-5)


class TestGPSOutputsAndQuantiles:
    def test_gps_ebgm_and_bounds_output(self, converted_data):
        """Verify GPS outputs EBGM, LowerBound, and UpperBound without NaNs."""
        result = gps(converted_data, min_events=3, ranking_statistic="log2")
        df = result.all_signals
        assert "EBGM" in df.columns
        assert "LowerBound" in df.columns
        assert "UpperBound" in df.columns
        assert df["EBGM"].notna().all()
        assert df["LowerBound"].notna().all()
        assert df["UpperBound"].notna().all()
        # Verify ordering: LowerBound <= EBGM <= UpperBound (on non-degenerate estimates)
        valid = (df["Count"] >= 3) & (df["Expected Count"] > 0)
        assert (df.loc[valid, "LowerBound"] <= df.loc[valid, "EBGM"] + 1e-3).all()
        assert (df.loc[valid, "EBGM"] <= df.loc[valid, "UpperBound"] + 1e-3).all()

    def test_quantiles_search_bracket_and_termination(self):
        """Verify quantiles terminates within max_iter on extreme parameters."""
        q_val = quantiles(0.05, 0.5, 50.0, 10.0, 2.0, 1.0)
        assert isinstance(q_val, float)
        assert q_val > 0.0


class TestLBEClipping:
    def test_lbe_with_p_equal_to_one(self):
        """Verify LBE does not fail or produce NaN when p-values equal 1.0."""
        pvals = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0, 1.0])
        res = lbe(pvals)
        assert not np.isnan(res.pi0)
        assert 0.0 <= res.pi0 <= 1.0


class TestScoreEnhancements:
    def test_sparse_eigensolver_in_score(self):
        """Verify SCORE-DA spectral clustering with many events executes rapidly via eigsh."""
        n_reports = 100
        n_events = 25
        np.random.seed(42)
        rows = []
        for i in range(n_reports):
            prod = f"Drug_{i % 5}"
            for e in range(n_events):
                if np.random.rand() < 0.15:
                    rows.append({"name": prod, "AE": f"AE_{e}", "count": 1})
        df = pd.DataFrame(rows)
        cont = convert(df, margin_threshold=1)
        res = score_da(cont, latent_rank=2, min_events=1)
        assert "Syndrome_Cluster" in res.all_signals.columns
        assert res.all_signals["Syndrome_Cluster"].nunique() >= 1

    def test_score_ddi_relative_risk_floor(self):
        """Verify SCORE-DDI works with protective baseline relative risks (RR < 1.0)."""
        df = pd.DataFrame({
            "report_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "name": ["DrugA", "DrugB", "DrugA", "DrugB", "DrugA", "DrugC", "DrugB", "DrugC", "DrugA", "DrugB", "DrugC", "DrugD"],
            "AE": ["Nausea", "Nausea", "Headache", "Headache", "Fatigue", "Fatigue", "Nausea", "Nausea", "Nausea", "Nausea", "Rash", "Rash"],
            "count": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        cont_ddi = convert_ddi(df, max_order=2)
        res = score_ddi(cont_ddi, interaction_model="multiplicative", min_events=1)
        assert len(res.all_signals) > 0


class TestVectorizedDataPrep:
    def test_expand_dataframe_vectorized_equivalency(self):
        """Verify vectorized __expand_dataframe produces correct row expansion."""
        expand_fn = getattr(dp_mod, "__expand_dataframe")
        df = pd.DataFrame({
            "product": ["DrugA", "DrugB", "DrugC"],
            "AE": ["Nausea", "Headache", "Rash"],
            "count": [3, 2, 1],
        })
        expanded = expand_fn(df, "count", "AE", "product")
        assert len(expanded) == 6
        assert (expanded["count"] == 1).all()
        assert list(expanded["product"]) == ["DrugA", "DrugA", "DrugA", "DrugB", "DrugB", "DrugC"]

    def test_transform_dataframe_vectorized_equivalency(self):
        """Verify vectorized __transform_dataframe produces exact one-hot count matrix."""
        transform_fn = getattr(dp_mod, "__transform_dataframe")
        df = pd.DataFrame({
            "product": ["DrugA", "DrugB", "DrugA"],
            "AE": ["Nausea", "Headache", "Nausea"],
            "count": [5, 3, 2],
        })
        transformed = transform_fn(df, "count", "AE")
        assert transformed.shape == (3, 2)
        assert set(transformed.columns) == {"Nausea", "Headache"}
        assert transformed.loc[0, "Nausea"] == 5
        assert transformed.loc[0, "Headache"] == 0
        assert transformed.loc[1, "Headache"] == 3


class TestConfigDeserialization:
    def test_all_configs_roundtrip(self):
        """Verify all Config classes can be re-instantiated via **asdict(config)."""
        configs = [
            PRRConfig(),
            RORConfig(),
            RFETConfig(),
            BCPNNConfig(),
            GPSConfig(),
            LASSOConfig(),
            SCOREConfig(),
            SCOREDDIConfig(),
        ]
        for cfg in configs:
            d = asdict(cfg)
            restored = type(cfg)(**d)
            assert cfg == restored


class TestModernSerialization:
    def test_export_pathlib_and_parquet(self, converted_data, tmp_path):
        """Verify AnalysisResult.export accepts pathlib.Path and exports Parquet."""
        result = prr(converted_data, min_events=3)

        # Pathlib + CSV
        csv_path = tmp_path / "test_signals.csv"
        result.export(csv_path)
        assert csv_path.exists()
        loaded_csv = pd.read_csv(csv_path)
        assert len(loaded_csv) == len(result.signals)

        # which='both' with CSV
        both_csv = tmp_path / "test_both.csv"
        result.export(both_csv, which="both")
        assert (tmp_path / "test_both_signals.csv").exists()
        assert (tmp_path / "test_both_all.csv").exists()

        # Pathlib + Parquet (handles environments without pyarrow gracefully)
        parquet_path = tmp_path / "test_signals.parquet"
        try:
            result.export(parquet_path)
            assert parquet_path.exists()
            loaded_pq = pd.read_parquet(parquet_path)
            assert len(loaded_pq) == len(result.signals)
        except ImportError as e:
            assert "pyarrow" in str(e) or "fastparquet" in str(e)

    def test_analysis_result_equality(self, converted_data):
        """Verify custom __eq__ on AnalysisResult doesn't fail with ambiguous truth value."""
        res1 = prr(converted_data, min_events=3)
        res2 = prr(converted_data, min_events=3)
        assert res1 == res2


class TestStructuredLogging:
    def test_logger_presence(self):
        """Verify logger is configured with vigipy name."""
        assert logger.name == "vigipy"

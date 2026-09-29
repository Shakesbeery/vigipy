import pytest
import numpy as np
import pandas as pd
from vigipy import score_da, analyze, SCOREConfig, get_default_config
from vigipy.utils.Container import AnalysisResult, DataContainer


class TestSCORE:
    """Test suite for SCORE-DA signal detection method."""

    def test_score_with_contingency(self, converted_data):
        res = score_da(converted_data, latent_rank=2, syndromic_weight=0.5, fdr_threshold=0.1)
        assert isinstance(res, AnalysisResult)
        assert res.num_signals == len(res.signals)
        assert len(res.all_signals) > 0

        # Check required columns
        expected_cols = [
            "Product", "Adverse Event", "Count", "Expected",
            "SER", "SRR", "SE", "z_score", "p_value", "p_adj", "Syndrome_Cluster"
        ]
        for col in expected_cols:
            assert col in res.all_signals.columns

        # Verify values are non-negative
        assert (res.all_signals["Count"] >= 0).all()
        assert (res.all_signals["Expected"] >= 0).all()
        assert (res.all_signals["SER"] >= 0).all()
        assert (res.all_signals["p_value"] >= 0.0).all() and (res.all_signals["p_value"] <= 1.0).all()
        assert (res.all_signals["p_adj"] >= 0.0).all() and (res.all_signals["p_adj"] <= 1.0).all()

    def test_score_with_binary(self, binary_data):
        res = score_da(binary_data, latent_rank=2, syndromic_weight=0.5, fdr_threshold=0.1)
        assert isinstance(res, AnalysisResult)
        assert len(res.all_signals) > 0
        assert (res.all_signals["SER"] >= 0.0).all()

    def test_score_parallel(self, converted_data):
        res_seq = score_da(converted_data, latent_rank=2, n_jobs=1, seed=42)
        res_par = score_da(converted_data, latent_rank=2, n_jobs=2, seed=42)

        assert res_seq.num_signals == res_par.num_signals
        np.testing.assert_allclose(
            res_seq.all_signals["SER"].values,
            res_par.all_signals["SER"].values,
            atol=1e-5,
        )

    def test_score_deflate_iterations(self, converted_data):
        res_1 = score_da(converted_data, deflate_iterations=1, seed=42)
        res_2 = score_da(converted_data, deflate_iterations=2, seed=42)
        assert isinstance(res_1, AnalysisResult)
        assert isinstance(res_2, AnalysisResult)

    def test_score_zero_latent_rank(self, converted_data):
        res = score_da(converted_data, latent_rank=0, syndromic_weight=0.0)
        assert isinstance(res, AnalysisResult)
        assert len(res.all_signals) > 0

    def test_unified_api(self, converted_data):
        cfg = SCOREConfig(latent_rank=2, min_events=2)
        res = analyze(converted_data, cfg)
        assert res.params["method"] == "score"
        assert res.params["input_params"]["latent_rank"] == 2
        assert res.params["input_params"]["min_events"] == 2

        default_cfg = get_default_config("score")
        assert isinstance(default_cfg, SCOREConfig)

    def test_export_csv(self, converted_data, tmp_path):
        res = score_da(converted_data, latent_rank=2)
        out_csv = tmp_path / "score_signals.csv"
        res.export(str(out_csv))
        assert out_csv.exists()
        loaded = pd.read_csv(out_csv)
        assert len(loaded) == res.num_signals

    def test_edge_case_single_cell(self):
        # 1 drug x 1 event
        cont = pd.DataFrame([[10]], index=["DrugA"], columns=["AE1"])
        container = DataContainer(
            data=pd.DataFrame(),
            N=10,
            contingency=cont,
            type="contingency",
        )
        res = score_da(container, latent_rank=1, min_events=1)
        assert isinstance(res, AnalysisResult)
        assert len(res.all_signals) == 1

    def test_score_consensus_integration(self, converted_data):
        from vigipy import consensus_analysis, PRRConfig
        res = consensus_analysis(
            converted_data, configs=[PRRConfig(), SCOREConfig()], min_consensus=1
        )
        assert res.num_signals > 0
        assert "score_score" in res.comparison_table.columns
        assert "alert_score" in res.comparison_table.columns


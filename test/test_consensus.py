import os
import numpy as np
import pandas as pd
import pytest

from vigipy import (
    convert,
    convert_binary,
    analyze_all,
    consensus_analysis,
    ConsensusResult,
    PRRConfig,
    RORConfig,
    RFETConfig,
    BCPNNConfig,
    GPSConfig,
    LASSOConfig,
)


class TestConsensusAnalysis:
    """Tests for the consensus_analysis function and ConsensusResult container."""

    def test_default_run(self, converted_data):
        """Verify default execution runs 5 regulatory methods and produces valid ConsensusResult."""
        res = consensus_analysis(converted_data, min_events=3)
        assert isinstance(res, ConsensusResult)
        assert res.methods == ["prr", "ror", "rfet", "bcpnn", "gps"]
        assert res.num_signals > 0
        assert len(res.signals) == res.num_signals
        assert len(res.comparison_table) >= res.num_signals

        # Master columns
        expected_cols = [
            "Product",
            "Adverse Event",
            "Count",
            "Expected Count",
            "composite_rank",
            "votes",
            "total_methods",
            "consensus_score",
            "agreement_tier",
            "alert_prr",
            "alert_ror",
            "alert_rfet",
            "alert_bcpnn",
            "alert_gps",
            "score_prr",
            "score_ror",
            "score_rfet",
            "score_bcpnn",
            "score_gps",
        ]
        for col in expected_cols:
            assert col in res.comparison_table.columns, f"Missing {col}"

        # Votes and score bounds
        assert (res.comparison_table["votes"] >= 0).all()
        assert (res.comparison_table["votes"] <= 5).all()
        assert (res.comparison_table["consensus_score"] >= 0.0).all()
        assert (res.comparison_table["consensus_score"] <= 1.0 + 1e-7).all()

        # Agreement tier valid set
        valid_tiers = {"Unanimous", "Strong", "Moderate", "Weak", "Isolated", "None"}
        assert set(res.comparison_table["agreement_tier"].unique()).issubset(valid_tiers)

        # Composite rank contiguous 1..N
        ranks = res.comparison_table["composite_rank"].values
        np.testing.assert_array_equal(ranks, np.arange(1, len(res.comparison_table) + 1))

    def test_custom_configs(self, converted_data):
        """Verify running a custom subset of method configurations."""
        configs = [
            PRRConfig(min_events=3),
            RORConfig(min_events=3),
            GPSConfig(min_events=3),
        ]
        res = consensus_analysis(converted_data, configs=configs)
        assert res.methods == ["prr", "ror", "gps"]
        assert "alert_prr" in res.comparison_table.columns
        assert "alert_ror" in res.comparison_table.columns
        assert "alert_gps" in res.comparison_table.columns
        assert "alert_bcpnn" not in res.comparison_table.columns
        assert res.comparison_table["votes"].max() <= 3

    def test_shared_overrides(self, converted_data):
        """Verify shared overrides are forwarded to all methods."""
        res = consensus_analysis(converted_data, min_events=5)
        assert (res.comparison_table["Count"] >= 5).all()

    def test_min_consensus_integer(self, converted_data):
        """Verify integer vote thresholding and monotonicity."""
        res1 = consensus_analysis(converted_data, min_events=3, min_consensus=1)
        res2 = consensus_analysis(converted_data, min_events=3, min_consensus=2)
        res3 = consensus_analysis(converted_data, min_events=3, min_consensus=3)
        res5 = consensus_analysis(converted_data, min_events=3, min_consensus=5)

        assert res1.num_signals >= res2.num_signals
        assert res2.num_signals >= res3.num_signals
        assert res3.num_signals >= res5.num_signals

        assert (res3.signals["votes"] >= 3).all()
        assert (res5.signals["votes"] == 5).all()
        assert (res5.signals["agreement_tier"] == "Unanimous").all()

    def test_min_consensus_fraction(self, converted_data):
        """Verify fractional consensus score thresholding."""
        res_half = consensus_analysis(converted_data, min_events=3, min_consensus=0.5)
        assert (res_half.signals["consensus_score"] >= 0.5 - 1e-7).all()

        res_full = consensus_analysis(converted_data, min_events=3, min_consensus=1.0)
        assert (res_full.signals["consensus_score"] >= 1.0 - 1e-7).all()
        assert (res_full.signals["votes"] == 5).all()

    def test_custom_weights(self, converted_data):
        """Verify custom method importance weights scale consensus_score appropriately."""
        weights = {"gps": 3.0, "bcpnn": 2.0, "prr": 1.0, "ror": 1.0, "rfet": 1.0}
        total_w = sum(weights.values())  # 8.0

        res_weighted = consensus_analysis(converted_data, min_events=3, weights=weights)
        top = res_weighted.signals.iloc[0]

        # Verify consensus_score calculation matches custom weights
        expected_score = sum(
            weights[m] for m in res_weighted.methods if top[f"alert_{m}"]
        ) / total_w
        assert np.isclose(top["consensus_score"], expected_score)

    def test_method_agreement_matrices(self, converted_data):
        """Verify structure, symmetry, and properties of method agreement matrices."""
        res = consensus_analysis(converted_data, min_events=3)
        m_agree = res.method_agreement

        for key in ("jaccard", "kappa", "correlation", "overlap"):
            assert key in m_agree
            df = m_agree[key]
            assert isinstance(df, pd.DataFrame)
            assert list(df.index) == res.methods
            assert list(df.columns) == res.methods

        # Check Jaccard properties
        jaccard = m_agree["jaccard"]
        np.testing.assert_allclose(np.diag(jaccard), 1.0)
        assert (jaccard.values >= 0.0).all() and (jaccard.values <= 1.0).all()
        np.testing.assert_allclose(jaccard.values, jaccard.values.T)

        # Check Overlap properties
        overlap = m_agree["overlap"]
        np.testing.assert_allclose(overlap.values, overlap.values.T)
        for m in res.methods:
            assert overlap.loc[m, m] == res.raw_results[m].num_signals

        # Check Kappa properties
        kappa = m_agree["kappa"]
        np.testing.assert_allclose(np.diag(kappa), 1.0)
        assert (kappa.values >= -1.0).all() and (kappa.values <= 1.0).all()
        np.testing.assert_allclose(kappa.values, kappa.values.T)

        # Check Spearman correlation properties
        corr = m_agree["correlation"]
        np.testing.assert_allclose(np.diag(corr), 1.0)
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_contingency_table(self, converted_data):
        """Verify 2x2 contingency table generation between two methods."""
        res = consensus_analysis(converted_data, min_events=3)
        ct = res.contingency_table("prr", "gps")

        assert isinstance(ct, pd.DataFrame)
        assert "Total" in ct.index
        assert "Total" in ct.columns
        assert ct.loc["Total", "Total"] == len(res.comparison_table)

        with pytest.raises(ValueError, match="not in consensus analysis"):
            res.contingency_table("prr", "unknown_method")

    def test_inspect_signal(self, converted_data):
        """Verify detailed signal inspection for a specific product-event pair."""
        res = consensus_analysis(converted_data, min_events=3)
        sample_prod = res.signals.iloc[0]["Product"]
        sample_ae = res.signals.iloc[0]["Adverse Event"]

        insp = res.inspect_signal(sample_prod, sample_ae)
        assert isinstance(insp, pd.DataFrame)
        assert len(insp) == len(res.methods)
        assert set(insp.columns) == {
            "Method",
            "Alert",
            "Metric",
            "Score",
            "CI Lower",
            "CI Upper",
            "p-value",
            "FDR",
            "Count",
            "Expected Count",
        }
        assert insp.attrs["Product"] == sample_prod
        assert insp.attrs["Adverse Event"] == sample_ae
        assert "votes" in insp.attrs
        assert "consensus_score" in insp.attrs
        assert "agreement_tier" in insp.attrs

        with pytest.raises(KeyError, match="not found"):
            res.inspect_signal("NONEXISTENT_DRUG_XYZ", "NO_AE")

    def test_precomputed_results_dict(self, converted_data):
        """Verify consensus_analysis accepts pre-computed results dictionary."""
        raw_res = analyze_all(converted_data, min_events=3)
        res_from_dict = consensus_analysis(raw_res)

        assert isinstance(res_from_dict, ConsensusResult)
        assert res_from_dict.methods == list(raw_res.keys())
        assert res_from_dict.num_signals == 173

    def test_export_csv_and_excel(self, converted_data, tmp_path):
        """Verify exporting to CSV and multi-tab Excel files."""
        res = consensus_analysis(converted_data, min_events=3)

        # CSV export
        csv_path = str(tmp_path / "test_consensus.csv")
        res.export(csv_path)
        assert os.path.exists(csv_path)
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == res.num_signals

        # Excel export
        xlsx_path = str(tmp_path / "test_consensus.xlsx")
        res.export(xlsx_path)
        assert os.path.exists(xlsx_path)

        import openpyxl

        wb = openpyxl.load_workbook(xlsx_path)
        expected_sheets = {
            "Consensus Signals",
            "Comparison Table",
            "Jaccard Similarity",
            "Cohens Kappa",
            "Spearman Correlation",
            "Alert Overlap",
        }
        assert set(wb.sheetnames) == expected_sheets
        wb.close()

    def test_single_method_graceful(self, converted_data):
        """Verify single method handles execution and assigns tiers cleanly."""
        res = consensus_analysis(
            converted_data, configs=[PRRConfig(min_events=3)], min_consensus=1
        )
        assert res.methods == ["prr"]
        assert len(res.comparison_table) > 0
        assert set(res.comparison_table["agreement_tier"].unique()).issubset(
            {"Unanimous", "None"}
        )

    def test_invalid_input_guards(self):
        """Verify invalid data input types raise appropriate exceptions."""
        with pytest.raises(TypeError, match="Expected data to be a DataContainer"):
            consensus_analysis(12345)  # type: ignore

        with pytest.raises(ValueError, match="No analysis results"):
            consensus_analysis({})

    def test_empty_signals_high_threshold(self, converted_data):
        """Verify empty signals table when min_consensus is unattainable."""
        res = consensus_analysis(converted_data, min_events=3, min_consensus=99)
        assert res.num_signals == 0
        assert len(res.signals) == 0
        assert list(res.signals.columns) == list(res.comparison_table.columns)

    def test_lasso_with_binary_data(self, converted_data, binary_data):
        """Verify consensus analysis combining PRR and LASSO results."""
        from vigipy import lasso, prr

        res_prr = prr(converted_data, min_events=3)
        res_lasso = lasso(binary_data, min_events=3)
        res = consensus_analysis({"prr": res_prr, "lasso": res_lasso})

        assert res.methods == ["prr", "lasso"]
        assert res.num_signals >= 0
        assert "score_lasso" in res.comparison_table.columns
        assert "alert_lasso" in res.comparison_table.columns
        assert "score_prr" in res.comparison_table.columns
        assert "alert_prr" in res.comparison_table.columns

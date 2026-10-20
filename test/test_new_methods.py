import numpy as np
import pandas as pd
import pytest

from vigipy import (
    aegis,
    AEGISConfig,
    analyze,
    analyze_all,
    consensus_analysis,
    convert,
    convert_binary,
    cusum_da,
    CUSUMConfig,
    get_default_config,
    LongitudinalModel,
    maxsprt,
    MaxSPRTConfig,
    prr,
    PRRConfig,
    rfet,
    ror,
    SCOREConfig,
    zigps,
    ZIGPSConfig,
)
from vigipy.utils.Container import AnalysisResult


class TestZIGPS:
    """Unit tests for Zero-Inflated Gamma-Poisson Shrinker (ZIGPS)."""

    def test_zigps_basic(self, converted_data):
        res = zigps(converted_data, min_events=2)
        assert isinstance(res, AnalysisResult)
        assert len(res.all_signals) > 0
        assert res.num_signals == len(res.signals)
        for col in ["Product", "Adverse Event", "Count", "Expected Count", "EBGM", "LowerBound", "UpperBound", "StructuralZeroProb"]:
            assert col in res.all_signals.columns
        assert 0.0 <= res.params["structural_zero_pi0"] < 1.0
        assert (res.all_signals["EBGM"] >= 0).all()

    def test_zigps_unified_api(self, converted_data):
        cfg = ZIGPSConfig(min_events=2, decision_metric="fdr", decision_thres=0.1)
        res = analyze(converted_data, cfg)
        assert res.params["method"] == "zigps"
        default_cfg = get_default_config("zigps")
        assert isinstance(default_cfg, ZIGPSConfig)


class TestSequentialMethods:
    """Unit tests for Kulldorff's MaxSPRT and Page's Poisson CUSUM."""

    def test_maxsprt_poisson_and_binomial(self, converted_data):
        res_p = maxsprt(converted_data, alpha=0.05, min_events=2, model="poisson")
        res_b = maxsprt(converted_data, alpha=0.05, min_events=2, model="binomial")
        assert isinstance(res_p, AnalysisResult)
        assert isinstance(res_b, AnalysisResult)
        assert "LLR" in res_p.all_signals.columns
        assert "CriticalValue" in res_p.all_signals.columns
        assert (res_p.all_signals["LLR"] >= 0.0).all()
        assert (res_b.all_signals["LLR"] >= 0.0).all()

    def test_cusum_da(self, converted_data):
        res = cusum_da(converted_data, target_rr=2.0, arl0=200.0, min_events=2)
        assert isinstance(res, AnalysisResult)
        assert "CUSUM" in res.all_signals.columns
        assert "CUSUM_Z" in res.all_signals.columns
        assert "ControlLimit" in res.all_signals.columns
        assert (res.all_signals["CUSUM"] >= 0.0).all()

        with pytest.raises(ValueError, match="target_rr must be > 1.0"):
            cusum_da(converted_data, target_rr=0.8)

    def test_sequential_unified_api(self, converted_data):
        res_m = analyze(converted_data, MaxSPRTConfig(min_events=2))
        res_c = analyze(converted_data, CUSUMConfig(min_events=2))
        assert res_m.params["method"] == "maxsprt"
        assert res_c.params["method"] == "cusum"


class TestAEGIS:
    """Unit tests for AEGIS (Anytime Empirical Graph-Informed Shrinkage)."""

    def test_aegis_with_contingency(self, converted_data):
        res = aegis(converted_data, min_events=2, fdr_threshold=0.1)
        assert isinstance(res, AnalysisResult)
        assert res.num_signals == len(res.signals)
        expected_cols = [
            "Product", "Adverse Event", "Count", "Effective_Count",
            "Expected", "Naive_Expected", "Unmasking_Ratio",
            "AEGIS_EBGM", "AEGIS_LB", "AEGIS_UB", "Context_Prior",
            "Posterior_Sig_Prob", "Syndromic_Coherence", "Surge_Velocity",
            "E_Value", "p_value", "fdr", "e_fdr", "Syndrome_Cluster",
        ]
        for col in expected_cols:
            assert col in res.all_signals.columns

        # Zero-count cells must never hallucinate positive scores
        zero_rows = res.all_signals[res.all_signals["Count"] == 0]
        if len(zero_rows) > 0:
            assert (zero_rows["AEGIS_EBGM"] == 0.0).all()
            assert (zero_rows["AEGIS_LB"] == 0.0).all()
            assert (zero_rows["E_Value"] == 1.0).all()

    def test_aegis_with_binary_and_velocity(self, binary_data):
        res = aegis(binary_data, min_events=1, decision_metric="combined")
        assert isinstance(res, AnalysisResult)
        assert len(res.all_signals) > 0
        assert (res.all_signals["AEGIS_EBGM"] >= 0.0).all()
        assert (res.all_signals["E_Value"] >= 0.0).all()

    def test_aegis_unmasking_blockbuster_competition(self):
        # Construct a synthetic table where a blockbuster drug has a huge spike (5000 reports on AE1),
        # which masks a genuine emerging signal on TargetDrug + AE1 (12 reports out of 38 total).
        rows = []
        # Blockbuster drug with 5000 on AE1 and 100 on AE2..AE11
        rows.append({"name": "Blockbuster", "AE": "AE1", "count": 5000})
        for k in range(2, 12):
            rows.append({"name": "Blockbuster", "AE": f"AE{k}", "count": 100})
        # TargetDrug has 12 on AE1, 8 on AE2 (co-occurring syndrome), and 2 on others
        rows.append({"name": "TargetDrug", "AE": "AE1", "count": 12})
        rows.append({"name": "TargetDrug", "AE": "AE2", "count": 8})
        for k in range(3, 12):
            rows.append({"name": "TargetDrug", "AE": f"AE{k}", "count": 2})
        # 15 background drugs with balanced counts across AE1..AE11
        for d in range(1, 16):
            for k in range(1, 12):
                rows.append({"name": f"BgDrug{d}", "AE": f"AE{k}", "count": 10})

        df = pd.DataFrame(rows)
        cont = convert(df)
        res = aegis(cont, min_events=3, latent_rank=0, sinkhorn_iters=6)
        target_row = res.all_signals[
            (res.all_signals["Product"] == "TargetDrug") & (res.all_signals["Adverse Event"] == "AE1")
        ].iloc[0]
        # Sinkhorn-Huber unmasking should reduce Expected significantly below Naive_Expected
        assert target_row["Expected"] < target_row["Naive_Expected"]
        assert target_row["Unmasking_Ratio"] > 2.0
        assert target_row["AEGIS_LB"] > 1.0

    def test_aegis_consensus_and_longitudinal(self, sample_df, converted_data):
        cons = consensus_analysis(
            converted_data,
            configs=[PRRConfig(min_events=2), ZIGPSConfig(min_events=2), MaxSPRTConfig(min_events=2), AEGISConfig(min_events=2)],
            min_consensus=1,
        )
        assert cons.num_signals >= 0
        assert "score_aegis" in cons.comparison_table.columns
        assert "score_zigps" in cons.comparison_table.columns
        assert "score_maxsprt" in cons.comparison_table.columns

        lm = LongitudinalModel(sample_df, "YE", decay_half_life="365D")
        lm.run(aegis, min_events=1)
        df_long = lm.to_dataframe(which="all")
        assert len(df_long) > 0
        assert "Surge_Velocity" in df_long.columns

    def test_extreme_zscore_no_underflow_and_fractional_rfet(self, sample_df):
        # Verify norm.sf prevents p_value == 0.0 underflow for strong signals (z ~ 12)
        rows = [
            {"name": "StrongDrug", "AE": "TargetAE", "count": 250},
            {"name": "StrongDrug", "AE": "OtherAE", "count": 500},
            {"name": "OtherDrug", "AE": "TargetAE", "count": 50},
            {"name": "OtherDrug", "AE": "OtherAE", "count": 5000},
        ]
        cont = convert(pd.DataFrame(rows))
        res_prr = prr(cont, min_events=1)
        top_p = res_prr.all_signals.loc[
            (res_prr.all_signals["Product"] == "StrongDrug")
            & (res_prr.all_signals["Adverse Event"] == "TargetAE"),
            "p_value",
        ].iloc[0]
        assert top_p > 0.0  # Previously underflowed to 0.0 due to 1.0 - norm.cdf(z)

        # Verify RFET handles fractional time-decayed counts without NaN
        lm = LongitudinalModel(sample_df, "YE", decay_half_life="180D")
        lm.run(rfet, min_events=1)
        df_rfet = lm.to_dataframe(which="all")
        assert not df_rfet["p_value"].isna().any()

import pytest
import numpy as np
import pandas as pd
from vigipy import (
    convert_ddi,
    score_ddi,
    score_da,
    prr,
    ror,
    gps,
    lasso,
    analyze,
    consensus_analysis,
    SCOREDDIConfig,
    SCOREConfig,
    RORConfig,
    get_default_config,
)
from vigipy.utils.Container import AnalysisResult, DataContainer


@pytest.fixture
def synthetic_ddi_df():
    """Create a controlled multi-drug reporting dataset.

    - Drug_A + Drug_B: Strongly synergistic interaction causing 'Serotonin Syndrome' and 'Hyperthermia'.
    - Drug_A alone: Causes 'Headache' predominantly; negligible 'Serotonin Syndrome'.
    - Drug_B alone: Causes 'Dizziness' predominantly; negligible 'Serotonin Syndrome'.
    - Drug_C alone: Causes 'Nausea' predominantly.
    - Drug_C + Drug_D: Common co-prescription causing 'Nausea' (no interaction synergy).
    - Drug_D alone: Baseline background medication across general symptoms.
    """
    rows = []
    rng = np.random.default_rng(42)

    # 1. 50 reports with Drug_A + Drug_B (synergistic combination)
    for rep_id in range(1, 51):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Serotonin Syndrome"})
        rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Serotonin Syndrome"})
        rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Hyperthermia"})
        rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Hyperthermia"})
        if rep_id % 5 == 0:
            rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Tremor"})
            rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Tremor"})

    # 2. 60 reports with Drug_A alone
    for rep_id in range(51, 111):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Headache"})
        if rep_id % 20 == 0:
            rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Serotonin Syndrome"})

    # 3. 60 reports with Drug_B alone
    for rep_id in range(111, 171):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Dizziness"})
        if rep_id % 20 == 0:
            rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Serotonin Syndrome"})

    # 4. 80 reports with Drug_C alone (high nausea)
    for rep_id in range(171, 251):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_C", "event": "Nausea"})
        if rep_id % 10 == 0:
            rows.append({"report_id": r_id, "drug": "Drug_C", "event": "Headache"})

    # 5. 30 reports with Drug_C + Drug_D (common benign co-prescription)
    for rep_id in range(251, 281):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_C", "event": "Nausea"})
        rows.append({"report_id": r_id, "drug": "Drug_D", "event": "Nausea"})
        rows.append({"report_id": r_id, "drug": "Drug_D", "event": "Fatigue"})

    # 6. 100 reports with Drug_D alone (general baseline)
    for rep_id in range(281, 381):
        r_id = f"REP_{rep_id}"
        event = rng.choice(["Headache", "Fatigue", "Dizziness", "Insomnia"])
        rows.append({"report_id": r_id, "drug": "Drug_D", "event": event})

    # 7. 15 reports with Drug_A + Drug_B + Drug_C (triplet interaction)
    for rep_id in range(381, 396):
        r_id = f"REP_{rep_id}"
        rows.append({"report_id": r_id, "drug": "Drug_A", "event": "Coma"})
        rows.append({"report_id": r_id, "drug": "Drug_B", "event": "Coma"})
        rows.append({"report_id": r_id, "drug": "Drug_C", "event": "Coma"})

    return pd.DataFrame(rows)


class TestConvertDDI:
    """Test suite for DDI data conversion."""

    def test_convert_ddi_structure(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        assert isinstance(container, DataContainer)
        assert container.type == "binary_ddi"
        assert container.pair_mapping is not None
        assert "Drug_A + Drug_B" in container.pair_mapping
        assert "Drug_C + Drug_D" in container.pair_mapping
        assert container.pair_mapping["Drug_A + Drug_B"] == ("Drug_A", "Drug_B")

        # Verify features include both single drugs and pairs
        for expected in ["Drug_A", "Drug_B", "Drug_C", "Drug_D", "Drug_A + Drug_B", "Drug_C + Drug_D"]:
            assert expected in container.product_features.columns
            assert expected in container.contingency.index

        # Verify event columns
        assert "Serotonin Syndrome" in container.event_outcomes.columns
        assert "Serotonin Syndrome" in container.contingency.columns

        # Verify contingency counts
        assert container.contingency.loc["Drug_A + Drug_B", "Serotonin Syndrome"] == 50

    def test_convert_ddi_target_drugs(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
            target_drugs=["Drug_A"],
        )
        assert "Drug_A + Drug_B" in container.pair_mapping
        assert "Drug_C + Drug_D" not in container.pair_mapping

    def test_convert_ddi_high_threshold_error(self, synthetic_ddi_df):
        with pytest.raises(ValueError, match="No drug combinations met the threshold"):
            convert_ddi(
                synthetic_ddi_df,
                product_label="drug",
                ae_label="event",
                report_id_label="report_id",
                min_co_reports=500,
            )

    def test_convert_ddi_missing_column_error(self, synthetic_ddi_df):
        with pytest.raises(ValueError, match="report_id_label 'missing_id' not found"):
            convert_ddi(
                synthetic_ddi_df,
                product_label="drug",
                ae_label="event",
                report_id_label="missing_id",
            )

    def test_convert_ddi_dense_mode(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
            sparse=False,
        )
        assert isinstance(container, DataContainer)
        assert not hasattr(container.product_features, "sparse")

    def test_max_order_triplets_convert(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
            max_order=3,
        )
        assert "Drug_A + Drug_B + Drug_C" in container.pair_mapping
        assert container.pair_mapping["Drug_A + Drug_B + Drug_C"] == ("Drug_A", "Drug_B", "Drug_C")
        assert "Drug_A + Drug_B + Drug_C" in container.contingency.index
        assert container.contingency.loc["Drug_A + Drug_B + Drug_C", "Coma"] == 15


class TestSCOREDDI:
    """Test suite for SCORE-DDI interaction discovery."""

    def test_score_ddi_basic_detection(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        res = score_ddi(container, min_events=2, syndromic_weight=0.5, fdr_threshold=0.05)
        assert isinstance(res, AnalysisResult)
        assert res.num_signals > 0

        # Required output columns
        expected_cols = [
            "Drug_1", "Drug_2", "Product", "Adverse Event", "Count",
            "Expected_Null", "SER_Interaction", "DDI_Ratio",
            "Interaction_Archetype", "Syndrome_Cluster", "fdr",
        ]
        for col in expected_cols:
            assert col in res.all_signals.columns

        # Verify the true synergistic interaction is detected with high significance
        synergy_hits = res.signals[
            (res.signals["Product"] == "Drug_A + Drug_B")
            & (res.signals["Adverse Event"] == "Serotonin Syndrome")
        ]
        assert len(synergy_hits) == 1
        hit = synergy_hits.iloc[0]
        assert hit["Count"] == 50
        assert hit["SER_Interaction"] > 0
        assert hit["DDI_Ratio"] > 1.0
        assert hit["fdr"] < 0.01
        assert hit["Interaction_Archetype"] == "EMERGENT"

    def test_score_ddi_models_additive_and_multiplicative(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        res_mult = score_ddi(container, interaction_model="multiplicative", min_events=2)
        res_add = score_ddi(container, interaction_model="additive", min_events=2)

        assert isinstance(res_mult, AnalysisResult)
        assert isinstance(res_add, AnalysisResult)
        assert res_mult.num_signals > 0
        assert res_add.num_signals > 0

    def test_score_ddi_parallel(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        res_seq = score_ddi(container, n_jobs=1, seed=42)
        res_par = score_ddi(container, n_jobs=2, seed=42)

        assert res_seq.num_signals == res_par.num_signals
        np.testing.assert_allclose(
            res_seq.all_signals["SER_Interaction"].values,
            res_par.all_signals["SER_Interaction"].values,
            atol=1e-5,
        )

    def test_score_ddi_unified_api(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        cfg = SCOREDDIConfig(min_events=2, syndromic_weight=0.3)
        res = analyze(container, cfg)
        assert isinstance(res, AnalysisResult)
        assert res.num_signals > 0

        default_cfg = get_default_config("score_ddi")
        assert isinstance(default_cfg, SCOREDDIConfig)

    def test_score_ddi_invalid_model_error(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        with pytest.raises(ValueError, match="Invalid interaction_model"):
            score_ddi(container, interaction_model="log_linear")

    def test_score_ddi_triplets(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
            max_order=3,
        )
        res = score_ddi(container, min_events=2, syndromic_weight=0.3)
        assert isinstance(res, AnalysisResult)
        triplet_hits = res.signals[res.signals["Product"] == "Drug_A + Drug_B + Drug_C"]
        assert len(triplet_hits) > 0
        hit = triplet_hits.iloc[0]
        assert hit["Adverse Event"] == "Coma"
        assert hit["Order"] == 3
        assert hit["Components"] == "Drug_A, Drug_B, Drug_C"
        assert hit["Count"] == 15
        assert hit["SER_Interaction"] > 0
        assert hit["Interaction_Archetype"] == "EMERGENT"


class TestDDIPlatformCompatibility:
    """Verify that containers created by convert_ddi are 100% compatible with existing methods."""

    def test_cross_method_compatibility(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )

        # 1. PRR
        res_prr = prr(container, min_events=2)
        assert isinstance(res_prr, AnalysisResult)
        assert "Drug_A + Drug_B" in res_prr.all_signals["Product"].values

        # 2. ROR
        res_ror = ror(container, min_events=2)
        assert isinstance(res_ror, AnalysisResult)
        assert "Drug_A + Drug_B" in res_ror.all_signals["Product"].values

        # 3. GPS
        res_gps = gps(container, min_events=2)
        assert isinstance(res_gps, AnalysisResult)
        assert "Drug_A + Drug_B" in res_gps.all_signals["Product"].values

        # 4. LASSO
        res_lasso = lasso(container, min_events=2)
        assert isinstance(res_lasso, AnalysisResult)
        assert "Drug_A + Drug_B" in res_lasso.all_signals["Product"].values

        # 5. SCORE-DA (single-method)
        res_score = score_da(container, min_events=2, latent_rank=2)
        assert isinstance(res_score, AnalysisResult)
        assert "Drug_A + Drug_B" in res_score.all_signals["Product"].values

    def test_consensus_analysis_with_ddi(self, synthetic_ddi_df):
        container = convert_ddi(
            synthetic_ddi_df,
            product_label="drug",
            ae_label="event",
            report_id_label="report_id",
            min_co_reports=5,
        )
        configs = [
            RORConfig(min_events=2),
            SCOREConfig(min_events=2, latent_rank=2),
            SCOREDDIConfig(min_events=2),
        ]
        cres = consensus_analysis(container, configs)
        assert len(cres.methods) == 3
        assert len(cres.signals) > 0
        assert "consensus_score" in cres.comparison_table.columns
        assert "Drug_A + Drug_B" in cres.signals["Product"].values

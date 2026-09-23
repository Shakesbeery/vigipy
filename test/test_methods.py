import numpy as np
import pandas as pd
import pytest

from vigipy import (
    bcpnn, gps, prr, ror, rfet, lasso, LongitudinalModel,
    analyze, analyze_all, get_default_config,
    PRRConfig, RORConfig, RFETConfig, BCPNNConfig, GPSConfig, LASSOConfig,
)

METHODS = ("mantel-haentzel", "negative-binomial", "poisson")
METRICS = ("fdr", "signals", "rank")
STATS_PRR_ROR = ("p_value", "CI")
STATS_BCPNN = ("p_value", "quantile")
STATS_GPS = ("p_value", "quantile", "log2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_result(result, expected_columns=None):
    """Common assertions for any disproportionality analysis result."""
    assert hasattr(result, "all_signals")
    assert hasattr(result, "signals")
    assert hasattr(result, "num_signals")
    assert isinstance(result.all_signals, pd.DataFrame)
    assert isinstance(result.signals, pd.DataFrame)
    assert result.num_signals >= 0
    # signals length must match num_signals
    assert len(result.signals) == result.num_signals, (
        f"len(signals)={len(result.signals)} != num_signals={result.num_signals}"
    )
    assert len(result.signals) <= len(result.all_signals)
    if expected_columns:
        assert set(expected_columns).issubset(result.all_signals.columns)
    for col in (
        "p_value", "PRR", "ROR", "RFET", "fdr", "quantile", "log2",
        "Count", "Expected Count", "FNR", "FOR", "Se", "Sp", "LASSO Coefficient",
    ):
        if col in result.all_signals.columns:
            assert result.all_signals[col].notna().all(), f"Column {col} contains NaN values"
    for col in ("Count", "Expected Count"):
        if col in result.all_signals.columns:
            assert (result.all_signals[col] >= 0).all(), f"Column {col} has negative values"
    for col in ("fdr", "FNR", "FOR", "Se", "Sp"):
        if col in result.all_signals.columns:
            assert (
                (result.all_signals[col] >= -1e-7).all()
                and (result.all_signals[col] <= 1.0 + 1e-7).all()
            ), f"Column {col} outside [0, 1]"


# ---------------------------------------------------------------------------
# PRR Tests
# ---------------------------------------------------------------------------

class TestPRR:
    def test_basic_run(self, converted_data):
        result = prr(converted_data, min_events=3)
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "Expected Count",
            "p_value", "PRR", "product margin", "event margin", "fdr",
        ])

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("stat", STATS_PRR_ROR)
    def test_all_combinations(self, converted_data, method, metric, stat):
        result = prr(
            converted_data,
            expected_method=method,
            decision_metric=metric,
            ranking_statistic=stat,
            min_events=3,
        )
        assert_valid_result(result)

    def test_golden_values(self, converted_data):
        """Regression test: pin numeric output for PRR with default settings."""
        result = prr(converted_data, min_events=3, decision_metric="rank")
        top = result.all_signals.iloc[0]
        assert top["Product"] == "XENMATRIX"
        assert top["Adverse Event"] == "Infection"
        assert top["Count"] == 61.0
        np.testing.assert_allclose(top["PRR"], 5.200984, rtol=1e-4)
        np.testing.assert_allclose(top["Expected Count"], 15.153476, rtol=1e-4)
        assert result.num_signals == 151
        # Baseline comparison without continuity correction
        result_uncorr = prr(converted_data, min_events=3, decision_metric="rank", continuity_correction=False)
        assert result_uncorr.num_signals == 116


# ---------------------------------------------------------------------------
# ROR Tests
# ---------------------------------------------------------------------------

class TestROR:
    def test_basic_run(self, converted_data):
        result = ror(converted_data, min_events=3)
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "Expected Count",
            "p_value", "ROR", "product margin", "event margin", "fdr",
        ])

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("stat", STATS_PRR_ROR)
    def test_all_combinations(self, converted_data, method, metric, stat):
        result = ror(
            converted_data,
            expected_method=method,
            decision_metric=metric,
            ranking_statistic=stat,
            min_events=3,
        )
        assert_valid_result(result)

    def test_golden_values(self, converted_data):
        """Regression test: pin numeric output for ROR with default settings."""
        result = ror(converted_data, min_events=3, decision_metric="rank")
        top = result.all_signals.iloc[0]
        assert top["Product"] == "XENMATRIX"
        assert top["Adverse Event"] == "Infection"
        assert top["Count"] == 61.0
        np.testing.assert_allclose(top["ROR"], 6.259909, rtol=1e-4)
        assert result.num_signals == 151
        # Baseline comparison without continuity correction
        result_uncorr = ror(converted_data, min_events=3, decision_metric="rank", continuity_correction=False)
        assert result_uncorr.num_signals == 116


# ---------------------------------------------------------------------------
# RFET Tests
# ---------------------------------------------------------------------------

class TestRFET:
    def test_basic_run(self, converted_data):
        result = rfet(converted_data, min_events=3)
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "Expected Count",
            "p_value", "RFET", "product margin", "event margin", "fdr",
        ])

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("metric", METRICS)
    def test_all_combinations(self, converted_data, method, metric):
        result = rfet(
            converted_data,
            expected_method=method,
            decision_metric=metric,
            min_events=3,
        )
        assert_valid_result(result)

    def test_mid_pval(self, converted_data):
        result_std = rfet(converted_data, min_events=3, mid_pval=False)
        result_mid = rfet(converted_data, min_events=3, mid_pval=True)
        assert_valid_result(result_mid)
        assert result_mid.all_signals["p_value"].notna().all()
        # Mid-p values should be <= standard one-sided Fisher p-values
        assert (
            result_mid.all_signals["p_value"].values
            <= result_std.all_signals["p_value"].values + 1e-10
        ).all()

    def test_golden_values(self, converted_data):
        """Regression test: pin numeric output for RFET with default settings."""
        result = rfet(converted_data, min_events=3, decision_metric="rank")
        top = result.all_signals.iloc[0]
        assert top["Product"] == "PELVISOFT"
        assert top["Adverse Event"] == "Dehydration"
        assert top["Count"] == 518.0
        np.testing.assert_allclose(top["p_value"], 2.388804e-297, rtol=1e-3)
        assert result.num_signals == 143


# ---------------------------------------------------------------------------
# BCPNN Tests
# ---------------------------------------------------------------------------

class TestBCPNN:
    def test_basic_run(self, converted_data):
        result = bcpnn(
            converted_data, min_events=3,
            decision_metric="rank", ranking_statistic="quantile",
        )
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "Expected Count",
            "quantile", "count/expected", "product margin", "event margin",
            "fdr", "FNR", "FOR", "Se", "Sp",
        ])
        # BCPNN should store params
        assert hasattr(result, "param")

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("stat", STATS_BCPNN)
    def test_all_combinations(self, converted_data, method, metric, stat):
        result = bcpnn(
            converted_data,
            expected_method=method,
            decision_metric=metric,
            ranking_statistic=stat,
            min_events=3,
        )
        assert_valid_result(result)

    def test_golden_values_quantile(self, converted_data):
        """Regression test: pin numeric output for BCPNN quantile ranking."""
        result = bcpnn(
            converted_data, min_events=3,
            decision_metric="rank", ranking_statistic="quantile",
        )
        top = result.all_signals.iloc[0]
        assert top["Product"] == "COLLAMEND"
        assert top["Adverse Event"] == "Plaque (lesion)"
        np.testing.assert_allclose(top["quantile"], 2.087242, rtol=1e-4)
        assert result.num_signals == 66

    def test_golden_values_pvalue(self, converted_data):
        """Regression test: pin numeric output for BCPNN p_value ranking."""
        result = bcpnn(
            converted_data, min_events=3,
            decision_metric="rank", ranking_statistic="p_value",
        )
        top = result.all_signals.iloc[0]
        assert top["Product"] == "PELVISOFT"
        assert top["Adverse Event"] == "Dehydration"
        np.testing.assert_allclose(top["p_value"], 1.151430e-76, rtol=1e-3)
        assert result.num_signals == 90
        
    def test_monte_carlo(self, converted_data):
        result = bcpnn(converted_data, min_events=3, MC=True, num_MC=1000)
        assert_valid_result(result)
        assert result.all_signals["quantile"].notna().all()
        assert len(result.signals) >= 0


# ---------------------------------------------------------------------------
# GPS Tests
# ---------------------------------------------------------------------------

class TestGPS:
    def test_basic_run(self, converted_data):
        result = gps(
            converted_data, min_events=3,
            decision_metric="rank", ranking_statistic="quantile",
            truncate=True,
        )
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "Expected Count",
            "quantile", "count/expected", "product margin", "event margin",
            "fdr", "FNR", "FOR", "Se", "Sp",
        ])
        assert hasattr(result, "param")
        assert "prior_param" in result.param
        assert "convergence" in result.param

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("stat", STATS_BCPNN)  # GPS uses p_value/quantile too
    def test_all_combinations(self, converted_data, method, metric, stat):
        result = gps(
            converted_data,
            expected_method=method,
            decision_metric=metric,
            ranking_statistic=stat,
            min_events=3,
            truncate=True,
        )
        assert_valid_result(result)

    def test_user_provided_priors(self, converted_data):
        """Test that providing prior_param skips optimization and doesn't crash."""
        priors = [0.2, 0.06, 1.4, 1.8, 0.1]
        result = gps(
            converted_data, min_events=3,
            decision_metric="rank", ranking_statistic="quantile",
            prior_param=priors,
        )
        assert_valid_result(result)
        assert result.param["convergence"] == "User-provided priors"


# ---------------------------------------------------------------------------
# LASSO Tests
# ---------------------------------------------------------------------------

class TestLASSO:
    def test_basic_run(self, binary_data):
        result = lasso(binary_data, lasso_thresh=0.1, min_events=3)
        assert_valid_result(result, [
            "Product", "Adverse Event", "LASSO Coefficient",
            "CI Lower", "CI Upper",
        ])

    def test_lars(self, binary_data):
        result = lasso(
            binary_data, lasso_thresh=0.1, min_events=3,
            use_lars=True, num_bootstrap=5,
        )
        assert_valid_result(result)

    def test_lars_ic(self, binary_data):
        for criterion in ("aic", "bic"):
            result = lasso(
                binary_data, lasso_thresh=0.1, min_events=3,
                use_IC=True, IC_criterion=criterion, num_bootstrap=5,
            )
            assert_valid_result(result)

    def test_glm(self, binary_data):
        result = lasso(
            binary_data, lasso_thresh=0.0, min_events=3,
            use_glm=True,
        )
        assert_valid_result(result)
        # GLM path should not produce bootstrap CIs
        assert (result.all_signals["CI Lower"] == 0).all()
        assert (result.all_signals["CI Upper"] == 0).all()

    def test_stores_params(self, binary_data):
        result = lasso(binary_data, lasso_thresh=0.1, min_events=3)
        assert hasattr(result, "param")
        assert result.param["method"] == "lasso"
        assert "lasso_thresh" in result.param["input_params"]

    def test_logistic_lasso_defaults(self, binary_data):
        result = lasso(binary_data, min_events=3)
        assert_valid_result(result, [
            "Product", "Adverse Event", "Count", "LASSO Coefficient",
            "aROR", "CI Lower", "CI Upper", "aROR Lower", "aROR Upper",
            "SE", "p_value",
        ])
        assert result.num_signals > 0  # Should detect signals by default
        assert (result.all_signals["aROR"] > 0).all()
        assert (result.all_signals["p_value"] >= 0).all() and (result.all_signals["p_value"] <= 1.0).all()

    def test_logistic_decision_metrics(self, binary_data):
        res_lb = lasso(binary_data, decision_metric="lower_bound", min_events=3)
        res_coef = lasso(binary_data, decision_metric="coefficient", min_events=3)
        assert res_lb.num_signals <= res_coef.num_signals
        assert (res_lb.signals["CI Lower"] > 0).all()

    def test_logistic_cv(self, binary_data):
        from vigipy.utils.Container import DataContainer
        sub_container = DataContainer(
            data=binary_data.data,
            N=binary_data.N,
            product_features=binary_data.product_features,
            event_outcomes=binary_data.event_outcomes[["Traumatic injury", "Seroma"]],
            type=binary_data.type,
        )
        result = lasso(sub_container, use_cv=True, cv=3, min_events=3)
        assert_valid_result(result)
        assert result.num_signals > 0

    def test_logistic_bootstrap(self, binary_data):
        from vigipy.utils.Container import DataContainer
        sub_container = DataContainer(
            data=binary_data.data,
            N=binary_data.N,
            product_features=binary_data.product_features,
            event_outcomes=binary_data.event_outcomes[["Traumatic injury", "Seroma"]],
            type=binary_data.type,
        )
        result = lasso(sub_container, use_bootstrap=True, num_bootstrap=5, min_events=3)
        assert_valid_result(result)

    def test_polypharmacy_confounding_adjustment(self):
        from vigipy import convert_binary

        # Synthetic polypharmacy dataset:
        # Drug A is a true toxic drug causing Jaundice (50 reports).
        # Drug B is an innocent bystander frequently co-prescribed with Drug A.
        # Drug C is background control.
        records = []
        report_id = 0

        # 40 reports with Drug A + Drug B + Jaundice
        for _ in range(40):
            records.append({"report_id": report_id, "name": "DrugA", "AE": "Jaundice", "count": 1})
            records.append({"report_id": report_id, "name": "DrugB", "AE": "Jaundice", "count": 1})
            report_id += 1

        # 10 reports with Drug A alone + Jaundice
        for _ in range(10):
            records.append({"report_id": report_id, "name": "DrugA", "AE": "Jaundice", "count": 1})
            report_id += 1

        # 50 reports with Drug B alone + Headache (no Jaundice)
        for _ in range(50):
            records.append({"report_id": report_id, "name": "DrugB", "AE": "Headache", "count": 1})
            report_id += 1

        # 200 reports with Drug C + Headache
        for _ in range(200):
            records.append({"report_id": report_id, "name": "DrugC", "AE": "Headache", "count": 1})
            report_id += 1

        df = pd.DataFrame(records)
        container = convert_binary(df, report_id_label="report_id")
        assert container.type == "binary_report"
        assert container.product_features.shape[1] == 3

        # Run multivariable LASSO
        res = lasso(container, min_events=3, C=1.0)
        jaundice_signals = res.all_signals[res.all_signals["Adverse Event"] == "Jaundice"]

        drugA_row = jaundice_signals[jaundice_signals["Product"] == "DrugA"].iloc[0]
        drugB_row = jaundice_signals[jaundice_signals["Product"] == "DrugB"].iloc[0]

        # Drug A should have a strong positive coefficient
        assert drugA_row["LASSO Coefficient"] > 1.0
        # Drug B should have a significantly smaller coefficient than Drug A
        assert drugA_row["LASSO Coefficient"] > drugB_row["LASSO Coefficient"] + 1.0


# ---------------------------------------------------------------------------
# LongitudinalModel Tests
# ---------------------------------------------------------------------------

class TestLongitudinalModel:
    def test_cumulative_run(self, sample_df):
        lm = LongitudinalModel(sample_df.copy(), "A")
        lm.run(bcpnn, False, decision_metric="signals", ranking_statistic="quantile")
        assert len(lm.results) > 0
        # Each result is a (timestamp, result_or_None) tuple
        for ts, res in lm.results:
            assert ts is not None
            if res is not None:
                assert hasattr(res, "all_signals")

    def test_disjoint_run(self, sample_df):
        lm = LongitudinalModel(sample_df.copy(), "A")
        lm.run_disjoint(bcpnn, False, decision_metric="signals", ranking_statistic="quantile")
        assert len(lm.results) > 0

    def test_run_with_prr(self, sample_df):
        lm = LongitudinalModel(sample_df.copy(), "A")
        lm.run(prr, False, min_events=1, decision_metric="signals", ranking_statistic="p_value")
        assert len(lm.results) > 0

    def test_regroup_dates(self, sample_df):
        lm = LongitudinalModel(sample_df.copy(), "A")
        lm.regroup_dates("Q")
        assert lm.time_unit == "Q"


# ---------------------------------------------------------------------------
# Unified Interface Tests
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_prr_via_analyze(self, converted_data):
        result = analyze(converted_data, PRRConfig(min_events=3, decision_metric="rank"))
        assert_valid_result(result)
        top = result.all_signals.iloc[0]
        assert top["Product"] == "XENMATRIX"
        np.testing.assert_allclose(top["PRR"], 5.200984, rtol=1e-4)
        assert result.params["method"] == "prr"

    def test_loop_over_methods(self, converted_data):
        configs = [
            PRRConfig(min_events=3),
            RORConfig(min_events=3),
            RFETConfig(min_events=3),
            BCPNNConfig(min_events=3),
        ]
        for cfg in configs:
            result = analyze(converted_data, cfg)
            assert_valid_result(result)
            assert result.params["method"] == cfg.method
            assert "input_params" in result.params

    def test_analyze_all(self, converted_data):
        results = analyze_all(converted_data, min_events=3)
        assert set(results.keys()) == {"prr", "ror", "rfet", "bcpnn", "gps"}
        for name, result in results.items():
            assert_valid_result(result)
            assert result.params["method"] == name

    def test_lasso_via_analyze(self, binary_data):
        result = analyze(binary_data, LASSOConfig(min_events=3, lasso_thresh=0.1))
        assert_valid_result(result)
        assert result.params["method"] == "lasso"

    def test_params_standardized(self, converted_data):
        for cfg in [PRRConfig(min_events=3), BCPNNConfig(min_events=3)]:
            result = analyze(converted_data, cfg)
            assert "method" in result.params
            assert "input_params" in result.params

    def test_config_immutable(self):
        cfg = PRRConfig(min_events=3)
        with pytest.raises(AttributeError):
            cfg.min_events = 5

    def test_get_default_config(self):
        cfg = get_default_config("prr")
        assert cfg.method == "prr"
        assert cfg.min_events == 1
        assert cfg.decision_metric == "fdr"

    def test_get_default_config_invalid(self):
        with pytest.raises(ValueError):
            get_default_config("invalid")


# ---------------------------------------------------------------------------
# FDR and LBE Tests
# ---------------------------------------------------------------------------

class TestFDRAndLBE:
    def test_compute_fdr_alignment(self):
        from vigipy.utils.common import compute_fdr

        pvals = np.array([0.9, 0.001, 0.05, 0.4, 0.8])
        fdr = compute_fdr(pvals, len(pvals))
        assert len(fdr) == len(pvals)
        assert not np.isnan(fdr).any()
        # Smallest p-value should have smallest FDR
        assert fdr[1] < fdr[0]
        assert fdr[1] < fdr[2]
        assert fdr[2] < fdr[3]

    def test_lbe_vectorized(self):
        from vigipy.utils.lbe import lbe

        pvals = np.array([0.001, 0.02, 0.05, 0.1, 0.3, 0.7])
        res = lbe(pvals)
        assert res.qvalues is not None
        assert not np.isnan(res.qvalues).any()
        assert not np.isinf(res.qvalues).any()
        # Check monotonicity of qvalues relative to sorted pvals
        sort_idx = np.argsort(pvals)
        sorted_q = res.qvalues[sort_idx]
        assert np.all(np.diff(sorted_q) >= -1e-10)

    def test_compute_fdr_monotonicity(self):
        from vigipy.utils.common import compute_fdr

        pvals = np.array([0.031, 0.01, 0.5, 0.03, 0.8, 0.001])
        fdr = compute_fdr(pvals, len(pvals))
        sort_idx = np.argsort(pvals)
        sorted_fdr = fdr[sort_idx]
        assert np.all(np.diff(sorted_fdr) >= -1e-10)

    def test_compute_bayesian_metrics_monotonicity(self):
        from vigipy.utils.common import compute_bayesian_metrics

        post_prob = np.array([0.05, 0.001, 0.02, 0.01])
        rank_stat = np.array([1.5, 1.2, 1.8, 1.0])
        fdr, fnr, for_val, se, sp = compute_bayesian_metrics(post_prob, len(post_prob), "quantile", rank_stat)
        sort_idx = np.argsort(-rank_stat)
        sorted_fdr = fdr[sort_idx]
        assert np.all(np.diff(sorted_fdr) >= -1e-10)


# ---------------------------------------------------------------------------
# Robustness, Edge Cases, and Model Options
# ---------------------------------------------------------------------------

class TestRobustnessAndEdgeCases:
    def test_gps_config_default_bounds(self, converted_data):
        from vigipy import analyze, GPSConfig
        cfg = GPSConfig(min_events=3)
        assert cfg.minimization_bounds is not None
        result = analyze(converted_data, config=cfg)
        assert result.num_signals >= 0

    def test_container_export_csv_and_excel(self, converted_data, tmp_path):
        import os
        from vigipy import prr
        result = prr(converted_data, min_events=3)
        csv_file = str(tmp_path / "signals.csv")
        result.export(csv_file)
        assert os.path.exists(csv_file)
        df_csv = pd.read_csv(csv_file)
        assert len(df_csv) == len(result.signals)

    def test_longitudinal_no_mutation_and_custom_count(self):
        dates = ["2020-01-15", "2020-06-15", "2021-01-15", "2021-06-15"]
        df = pd.DataFrame({
            "date": dates,
            "drug": ["DrugA", "DrugB", "DrugA", "DrugB"],
            "ae": ["Nausea", "Headache", "Nausea", "Headache"],
            "my_events": [5, 10, 8, 12],
        })
        original_date_dtype = df["date"].dtype
        model = LongitudinalModel(df, time_unit="A", count_col="my_events")
        assert model.time_unit == "A"
        assert df["date"].dtype == original_date_dtype
        assert model.count_col == "my_events"

    def test_lasso_vectorized_bootstrap(self, binary_data):
        res = lasso(binary_data, num_bootstrap=5, min_events=3)
        assert res.num_signals >= 0
        assert not np.isnan(res.all_signals["LASSO Coefficient"]).any()
        assert not np.isnan(res.all_signals["CI Lower"]).any()
        assert not np.isnan(res.all_signals["CI Upper"]).any()

    def test_zero_cell_continuity_correction(self):
        from vigipy import convert, prr, ror, rfet
        df = pd.DataFrame({
            "name": ["DrugA", "DrugB", "DrugA", "DrugB"],
            "AE": ["AE1", "AE1", "AE2", "AE2"],
            "count": [0, 5, 10, 20],
        })
        cont = convert(df)
        res_prr = prr(cont, min_events=0)
        res_ror = ror(cont, min_events=0)
        res_rfet = rfet(cont, min_events=0)
        for res, name in [(res_prr, "PRR"), (res_ror, "ROR"), (res_rfet, "RFET")]:
            assert not np.isinf(res.all_signals[name]).any()
            assert not np.isnan(res.all_signals[name]).any()
            assert not np.isnan(res.all_signals["p_value"]).any()


# ---------------------------------------------------------------------------
# Bayesian Decision Metrics (FDR, FNR, FOR, Se, Sp) Tests
# ---------------------------------------------------------------------------

class TestBayesianDecisionMetrics:
    def test_compute_bayesian_metrics_exact_math(self):
        from vigipy.utils.common import compute_bayesian_metrics

        # 5 candidate signals with known posterior null probabilities P(H0)
        p_h0 = np.array([0.05, 0.10, 0.30, 0.70, 0.85])
        FDR, FNR, FOR, Se, Sp = compute_bayesian_metrics(p_h0, num_cell=5)

        # 1. FDR matches cumulative mean null probability
        np.testing.assert_allclose(FDR, [0.05, 0.075, 0.15, 0.2875, 0.40], rtol=1e-5)

        # 2. Sensitivity (Se) = TP_cum / Total_True_Signals (Total Signal = 3.0)
        expected_se = np.array([0.95, 1.85, 2.55, 2.85, 3.00]) / 3.0
        np.testing.assert_allclose(Se, expected_se, rtol=1e-5)

        # 3. FNR = 1.0 - Se (classic miss rate)
        np.testing.assert_allclose(FNR, 1.0 - expected_se, rtol=1e-5)
        np.testing.assert_allclose(FNR + Se, 1.0, rtol=1e-5)

        # 4. FOR = Missed signals / remaining unselected cells
        expected_for = [2.05 / 4.0, 1.15 / 3.0, 0.45 / 2.0, 0.15 / 1.0, 0.0]
        np.testing.assert_allclose(FOR, expected_for, rtol=1e-5, atol=1e-6)

        # 5. Specificity (Sp) = Remaining true negatives / total true negatives (Total Null = 2.0)
        expected_sp = np.array([1.95, 1.85, 1.55, 0.85, 0.0]) / 2.0
        np.testing.assert_allclose(Sp, expected_sp, rtol=1e-5)

        # 6. All metrics bounded in [0, 1]
        for m in (FDR, FNR, FOR, Se, Sp):
            assert (m >= 0.0).all() and (m <= 1.0).all()

    def test_bcpnn_and_gps_report_fnr_and_for(self, converted_data):
        from vigipy import bcpnn, gps

        res_b = bcpnn(converted_data, min_events=3)
        assert "FNR" in res_b.all_signals.columns
        assert "FOR" in res_b.all_signals.columns
        np.testing.assert_allclose(
            res_b.all_signals["FNR"] + res_b.all_signals["Se"],
            1.0,
            rtol=1e-5,
            atol=1e-5,
        )

        res_g = gps(converted_data, min_events=3, truncate=True)
        assert "FNR" in res_g.all_signals.columns
        assert "FOR" in res_g.all_signals.columns
        np.testing.assert_allclose(
            res_g.all_signals["FNR"] + res_g.all_signals["Se"],
            1.0,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_edge_cases(self):
        from vigipy.utils.common import compute_bayesian_metrics

        # Empty inputs
        empty = np.array([])
        FDR, FNR, FOR, Se, Sp = compute_bayesian_metrics(empty, num_cell=0)
        assert len(FDR) == len(FNR) == len(FOR) == len(Se) == len(Sp) == 0

        # Unsorted inputs with rank_stat: verify output matches correct row indices
        p_raw = np.array([0.85, 0.05, 0.70, 0.10, 0.30])
        # Suppose ranking by p-value ascending (0.05 is rank 1, 0.85 is rank 5)
        FDR, FNR, FOR, Se, Sp = compute_bayesian_metrics(p_raw, num_cell=5, ranking_statistic="p_value")
        # Item at index 1 (p=0.05) is the highest priority alert (rank 1)
        assert np.isclose(FDR[1], 0.05)
        assert np.isclose(Se[1], 0.95 / 3.0)
        assert np.isclose(FNR[1], 1.0 - 0.95 / 3.0)
        # Item at index 0 (p=0.85) is the lowest priority (rank 5)
        assert np.isclose(Se[0], 1.0)
        assert np.isclose(FNR[0], 0.0)


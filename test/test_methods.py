import numpy as np
import pandas as pd
import pytest

from vigipy import bcpnn, gps, prr, ror, rfet, lasso, LongitudinalModel

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
    # signals must be a subset of all_signals by length
    assert len(result.signals) <= len(result.all_signals)
    if expected_columns:
        assert set(expected_columns).issubset(result.all_signals.columns)


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
        assert result.num_signals == 116


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
        assert result.num_signals == 116


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
        result = rfet(converted_data, min_events=3, mid_pval=True)
        assert_valid_result(result)

    def test_golden_values(self, converted_data):
        """Regression test: pin numeric output for RFET with default settings."""
        result = rfet(converted_data, min_events=3, decision_metric="rank")
        top = result.all_signals.iloc[0]
        assert top["Product"] == "PELVISOFT"
        assert top["Adverse Event"] == "Dehydration"
        assert top["Count"] == 518.0
        np.testing.assert_allclose(top["p_value"], 2.710960e-297, rtol=1e-3)
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
            "fdr", "FNR", "Se", "Sp",
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
        assert result.num_signals == 65

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
        assert result.num_signals == 89


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
            "fdr", "FNR", "Se", "Sp",
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
        assert "lasso_thresh" in result.param


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

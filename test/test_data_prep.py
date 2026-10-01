import numpy as np
import pandas as pd

from vigipy import convert, convert_binary
from vigipy.utils import test_dispersion as run_dispersion_test


class TestConvert:
    def test_returns_container_with_expected_attributes(self, sample_df):
        data = convert(sample_df)
        assert hasattr(data, "data")
        assert hasattr(data, "N")
        assert hasattr(data, "contingency")
        assert hasattr(data, "type")

    def test_data_is_dataframe(self, converted_data):
        assert isinstance(converted_data.data, pd.DataFrame)

    def test_contingency_is_dataframe(self, converted_data):
        assert isinstance(converted_data.contingency, pd.DataFrame)

    def test_n_is_positive(self, converted_data):
        assert converted_data.N > 0

    def test_data_has_required_columns(self, converted_data):
        required = {"events", "product_aes", "count_across_brands", "ae_name", "product_name"}
        assert required.issubset(converted_data.data.columns)

    def test_n_equals_sum_of_events(self, converted_data):
        assert converted_data.N == converted_data.data["events"].sum()

    def test_data_shape_with_sample(self, converted_data):
        assert converted_data.data.shape == (802, 5)
        assert converted_data.N == 4359

    def test_margin_threshold_filters_rows(self, sample_df):
        data_low = convert(sample_df, margin_threshold=1)
        data_high = convert(sample_df, margin_threshold=10)
        assert len(data_high.data) <= len(data_low.data)


class TestConvertBinary:
    def test_returns_container_with_expected_attributes(self, sample_df):
        data = convert_binary(sample_df)
        assert hasattr(data, "product_features")
        assert hasattr(data, "event_outcomes")
        assert hasattr(data, "N")
        assert hasattr(data, "type")

    def test_product_features_is_dataframe(self, binary_data):
        assert isinstance(binary_data.product_features, pd.DataFrame)

    def test_event_outcomes_is_dataframe(self, binary_data):
        assert isinstance(binary_data.event_outcomes, pd.DataFrame)

    def test_n_is_positive(self, binary_data):
        assert binary_data.N > 0

    def test_shapes_with_sample(self, binary_data):
        assert binary_data.product_features.shape == (4359, 10)
        assert binary_data.event_outcomes.shape == (4359, 463)

    def test_type_is_binary(self, binary_data):
        assert binary_data.type == "binary"

    def test_sparse_convert_binary(self, sample_df):
        dense_data = convert_binary(sample_df, sparse=False)
        sparse_data = convert_binary(sample_df, sparse=True)
        assert hasattr(sparse_data.product_features, "sparse")
        assert hasattr(sparse_data.event_outcomes, "sparse")
        assert sparse_data.product_features.shape == dense_data.product_features.shape
        assert sparse_data.event_outcomes.shape == dense_data.event_outcomes.shape
        assert sparse_data.feature_names == list(dense_data.product_features.columns)
        assert sparse_data.event_names == list(dense_data.event_outcomes.columns)

    def test_covariate_labels_without_report_id_raises(self, sample_df):
        import pytest
        with pytest.raises(ValueError, match="covariate_labels requires report_id_label"):
            convert_binary(sample_df, covariate_labels=["age"])

    def test_covariates_extraction_and_standardization(self):
        import numpy as np

        df = pd.DataFrame({
            "report_id": [1, 1, 2, 2, 3],
            "name": ["DrugA", "DrugB", "DrugA", "DrugC", "DrugB"],
            "AE": ["Nausea", "Headache", "Nausea", "Fever", "Headache"],
            "count": [1, 1, 1, 1, 1],
            "age": [30.0, 30.0, 50.0, 50.0, 70.0],
            "sex": ["F", "F", "M", "M", "F"],
            "constant_col": [1.0, 1.0, 1.0, 1.0, 1.0],
        })

        container = convert_binary(
            df,
            report_id_label="report_id",
            covariate_labels=["age", "sex", "constant_col"],
        )
        assert container.covariates is not None
        assert container.covariate_names is not None
        assert len(container.covariates) == 3
        # Age should be standardized (mean=0, std=1)
        np.testing.assert_allclose(container.covariates["age"].mean(), 0.0, atol=1e-7)
        np.testing.assert_allclose(container.covariates["age"].std(), 1.0, atol=1e-7)
        # Sex should be one-hot encoded with drop_first=True
        assert "sex_M" in container.covariates.columns
        assert "sex_F" not in container.covariates.columns
        # Constant col should have zero-variance guard and not be NaN
        assert not container.covariates["constant_col"].isna().any()
        np.testing.assert_allclose(container.covariates["constant_col"], 0.0, atol=1e-7)

    def test_sparse_report_id_crosstab(self):
        df = pd.DataFrame({
            "report_id": [1, 1, 2, 3],
            "name": ["DrugA", "DrugB", "DrugA", "DrugB"],
            "AE": ["Nausea", "Headache", "Nausea", "Headache"],
            "count": [1, 1, 1, 1],
        })
        dense = convert_binary(df, report_id_label="report_id", sparse=False)
        sparse = convert_binary(df, report_id_label="report_id", sparse=True)
        assert hasattr(sparse.product_features, "sparse")
        assert hasattr(sparse.event_outcomes, "sparse")
        np.testing.assert_array_equal(
            dense.product_features.values,
            sparse.product_features.sparse.to_coo().toarray(),
        )


class TestDispersion:
    def test_returns_dict_with_expected_keys(self, converted_data):
        result = run_dispersion_test(converted_data)
        assert isinstance(result, dict)
        assert "dispersion" in result
        assert "alpha" in result
        assert "lb" in result
        assert "ub" in result

    def test_dispersion_values_with_sample(self, converted_data):
        result = run_dispersion_test(converted_data)
        assert result["dispersion"] > 2
        assert result["alpha"] > 1
        assert result["lb"] < result["alpha"] < result["ub"]

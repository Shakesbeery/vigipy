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
        assert result["dispersion"] > 10
        assert result["alpha"] > 1
        assert result["lb"] < result["alpha"] < result["ub"]

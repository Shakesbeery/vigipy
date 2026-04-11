import os
import pytest
import pandas as pd

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="session")
def sample_df():
    """Raw sample DataFrame loaded from CSV."""
    return pd.read_csv(os.path.join(FIXTURE_DIR, "sample.csv"))


@pytest.fixture(scope="session")
def converted_data(sample_df):
    """Converted container for disproportionality analyses."""
    from vigipy import convert

    return convert(sample_df)


@pytest.fixture(scope="session")
def binary_data(sample_df):
    """Binary-converted container for LASSO analyses."""
    from vigipy import convert_binary

    return convert_binary(sample_df)

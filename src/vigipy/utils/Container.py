import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd


@dataclass
class AnalysisResult:
    """Result container returned by disproportionality analysis methods.

    Attributes:
        all_signals: DataFrame of all drug-event pairs with computed statistics.
        signals: Filtered DataFrame of detected signals.
        num_signals: Number of detected signals.
        params: Dictionary of input parameters and model metadata.
    """

    all_signals: pd.DataFrame
    signals: pd.DataFrame
    num_signals: int
    params: Optional[dict[str, Any]] = field(default=None)

    @property
    def param(self) -> Optional[dict[str, Any]]:
        """Backward-compatible alias for params."""
        return self.params

    def export(
        self,
        name: Union[str, os.PathLike],
        index: bool = False,
        which: str = "signals",
    ) -> None:
        """Export signals and all data to an Excel (.xlsx), CSV (.csv), or Parquet (.parquet) file.

        Parameters:
            name: Output filepath or PathLike object. If the path ends with '.parquet', exports
                to Apache Parquet format. If it ends with '.csv', exports to CSV.
                Otherwise, writes 'Signals' and 'all_data' sheets to Excel (.xlsx).
            index: Whether to write row index labels to the output file.
            which: Which table(s) to export ('signals', 'all', or 'both'). Default is 'signals'.
                When which='both' for CSV or Parquet, exports '{base}_signals.{ext}' and
                '{base}_all.{ext}'.
        """
        path_str = os.fspath(name)

        if path_str.endswith(".parquet"):
            try:
                if which == "signals":
                    self.signals.to_parquet(path_str, index=index)
                elif which == "all":
                    self.all_signals.to_parquet(path_str, index=index)
                else:
                    base, ext = os.path.splitext(path_str)
                    self.signals.to_parquet(f"{base}_signals{ext}", index=index)
                    self.all_signals.to_parquet(f"{base}_all{ext}", index=index)
                return
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "Exporting to Parquet (.parquet) requires 'pyarrow' or 'fastparquet'. "
                    "Install with 'pip install pyarrow' or 'pip install fastparquet'."
                ) from exc

        if path_str.endswith(".csv"):
            if which == "signals":
                self.signals.to_csv(path_str, index=index)
            elif which == "all":
                self.all_signals.to_csv(path_str, index=index)
            else:
                base, ext = os.path.splitext(path_str)
                self.signals.to_csv(f"{base}_signals{ext}", index=index)
                self.all_signals.to_csv(f"{base}_all{ext}", index=index)
            return

        try:
            with pd.ExcelWriter(path_str) as writer:
                self.signals.to_excel(writer, sheet_name="Signals", index=index)
                self.all_signals.to_excel(writer, sheet_name="all_data", index=index)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Excel (.xlsx) requires 'openpyxl'. "
                "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
            ) from exc

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisResult):
            return False
        if self.num_signals != other.num_signals:
            return False
        if self.params != other.params:
            return False
        try:
            pd.testing.assert_frame_equal(self.signals, other.signals)
            pd.testing.assert_frame_equal(self.all_signals, other.all_signals)
            return True
        except (AssertionError, ValueError):
            return False

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(num_signals={self.num_signals}, "
            f"total={len(self.all_signals)})"
        )


@dataclass
class DataContainer:
    """Container for converted input data used by analysis methods.

    Attributes:
        data: Flattened DataFrame of drug-event pairs with counts.
        N: Total number of adverse events across all products.
        contingency: Contingency table (products x events), if applicable.
        product_features: Binary product feature matrix, for LASSO.
        event_outcomes: Event outcome matrix, for LASSO.
        type: The conversion type used ('contingency', 'binary', 'binary_count').
        covariates: Per-report covariate matrix (standardized continuous, drop_first categoricals).
        feature_names: Product/drug feature column names.
        event_names: Adverse event column names.
        covariate_names: Covariate column names.
    """

    data: pd.DataFrame
    N: int
    contingency: Optional[pd.DataFrame] = None
    product_features: Optional[pd.DataFrame] = None
    event_outcomes: Optional[pd.DataFrame] = None
    type: str = "contingency"
    covariates: Optional[pd.DataFrame] = None
    feature_names: Optional[list] = None
    event_names: Optional[list] = None
    covariate_names: Optional[list] = None
    pair_mapping: Optional[dict[str, tuple[str, str]]] = None


class Container:
    """Legacy container class preserved for backward compatibility.

    New applications should use AnalysisResult or DataContainer directly.
    """

    def __init__(self, params=False):
        if params:
            self.param = dict()

    def export(
        self,
        name: Union[str, os.PathLike],
        index: bool = False,
        which: str = "signals",
    ) -> None:
        """Export signals and all data to an Excel (.xlsx), CSV (.csv), or Parquet (.parquet) file.

        Parameters:
            name: Output filepath or PathLike object. If the path ends with '.parquet', exports
                to Apache Parquet format. If it ends with '.csv', exports to CSV.
                Otherwise, writes 'Signals' and 'all_data' sheets to Excel (.xlsx).
            index: Whether to write row index labels to the output file.
            which: Which table(s) to export ('signals', 'all', or 'both'). Default is 'signals'.
        """
        path_str = os.fspath(name)

        if path_str.endswith(".parquet"):
            if which == "signals":
                self.signals.to_parquet(path_str, index=index)
            elif which == "all":
                self.all_signals.to_parquet(path_str, index=index)
            else:
                base, ext = os.path.splitext(path_str)
                self.signals.to_parquet(f"{base}_signals{ext}", index=index)
                self.all_signals.to_parquet(f"{base}_all{ext}", index=index)
            return

        if path_str.endswith(".csv"):
            if which == "signals":
                self.signals.to_csv(path_str, index=index)
            elif which == "all":
                self.all_signals.to_csv(path_str, index=index)
            else:
                base, ext = os.path.splitext(path_str)
                self.signals.to_csv(f"{base}_signals{ext}", index=index)
                self.all_signals.to_csv(f"{base}_all{ext}", index=index)
            return

        try:
            with pd.ExcelWriter(path_str) as writer:
                self.signals.to_excel(writer, sheet_name="Signals", index=index)
                self.all_signals.to_excel(writer, sheet_name="all_data", index=index)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Excel (.xlsx) requires 'openpyxl'. "
                "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
            ) from exc

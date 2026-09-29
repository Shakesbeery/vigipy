import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd


def _export_tabular_result(
    primary_df: pd.DataFrame,
    secondary_df: Optional[pd.DataFrame],
    name: Union[str, os.PathLike],
    primary_sheet: str = "Signals",
    secondary_sheet: str = "all_data",
    which: str = "signals",
    index: bool = False,
) -> None:
    """Export one or two tabular DataFrames to Excel (.xlsx), CSV (.csv), or Parquet (.parquet)."""
    path_str = os.fspath(name)

    if path_str.endswith(".parquet") or path_str.endswith(".pq"):
        try:
            if which == "signals":
                primary_df.to_parquet(path_str, index=index)
            elif which == "all":
                target = secondary_df if secondary_df is not None else primary_df
                target.to_parquet(path_str, index=index)
            else:
                base, ext = os.path.splitext(path_str)
                primary_df.to_parquet(f"{base}_signals{ext}", index=index)
                if secondary_df is not None:
                    secondary_df.to_parquet(f"{base}_all{ext}", index=index)
            return
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Parquet (.parquet) requires 'pyarrow' or 'fastparquet'. "
                "Install with 'pip install pyarrow' or 'pip install fastparquet'."
            ) from exc

    if path_str.endswith(".csv"):
        if which == "signals":
            primary_df.to_csv(path_str, index=index)
        elif which == "all":
            target = secondary_df if secondary_df is not None else primary_df
            target.to_csv(path_str, index=index)
        else:
            base, ext = os.path.splitext(path_str)
            primary_df.to_csv(f"{base}_signals{ext}", index=index)
            if secondary_df is not None:
                secondary_df.to_csv(f"{base}_all{ext}", index=index)
        return

    try:
        with pd.ExcelWriter(path_str) as writer:
            primary_df.to_excel(writer, sheet_name=primary_sheet, index=index)
            if secondary_df is not None:
                secondary_df.to_excel(writer, sheet_name=secondary_sheet, index=index)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Exporting to Excel (.xlsx) requires 'openpyxl'. "
            "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
        ) from exc


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
        _export_tabular_result(
            self.signals,
            self.all_signals,
            name,
            primary_sheet="Signals",
            secondary_sheet="all_data",
            which=which,
            index=index,
        )

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
        method = ""
        if isinstance(self.params, dict) and "method" in self.params:
            method = f" [{self.params['method'].upper()}]"
        total = len(self.all_signals) if self.all_signals is not None else 0
        header = f"<AnalysisResult: {self.num_signals} signal(s) detected out of {total} pairs{method}>"
        if self.num_signals == 0 or self.signals is None or len(self.signals) == 0:
            return header

        # Format a clean preview of top signals
        top_n = min(3, len(self.signals))
        lines = [header, "  Top Signals:"]
        score_col = None
        for cand in ["PRR", "ROR", "RFET", "aROR", "EBGM", "SER", "quantile", "p_value"]:
            if cand in self.signals.columns:
                score_col = cand
                break

        for idx in range(top_n):
            row = self.signals.iloc[idx]
            prod = row.get("Product", "?")
            ae = row.get("Adverse Event", "?")
            cnt = int(row.get("Count", 0)) if pd.notna(row.get("Count", 0)) else 0
            if score_col:
                val = row[score_col]
                val_str = f"{val:.3f}" if isinstance(val, (int, float, np.floating)) else str(val)
                lines.append(f"    - {prod} | {ae} (Count: {cnt}, {score_col}: {val_str})")
            else:
                lines.append(f"    - {prod} | {ae} (Count: {cnt})")

        if self.num_signals > top_n:
            lines.append(f"    ... and {self.num_signals - top_n} more signal(s)")
        return "\n".join(lines)


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
        _export_tabular_result(
            self.signals,
            self.all_signals,
            name,
            primary_sheet="Signals",
            secondary_sheet="all_data",
            which=which,
            index=index,
        )

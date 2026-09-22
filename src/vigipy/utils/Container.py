from dataclasses import dataclass, field
from typing import Any, Optional

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

    def export(self, name: str, index: bool = False) -> None:
        """Export signals and all data to an Excel (.xlsx) or CSV (.csv) file.

        Parameters:
            name: Output filepath. If the path ends with '.csv', signals are exported
                to CSV format. Otherwise, writes 'Signals' and 'all_data' sheets to Excel.
            index: Whether to write row index labels to the output file.
        """
        if name.endswith(".csv"):
            self.signals.to_csv(name, index=index)
            return

        try:
            with pd.ExcelWriter(name) as writer:
                self.signals.to_excel(writer, sheet_name="Signals", index=index)
                self.all_signals.to_excel(writer, sheet_name="all_data", index=index)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Excel (.xlsx) requires 'openpyxl'. "
                "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
            ) from exc

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
    """

    data: pd.DataFrame
    N: int
    contingency: Optional[pd.DataFrame] = None
    product_features: Optional[pd.DataFrame] = None
    event_outcomes: Optional[pd.DataFrame] = None
    type: str = "contingency"


class Container:
    """Legacy container class preserved for backward compatibility.

    New applications should use AnalysisResult or DataContainer directly.
    """

    def __init__(self, params=False):
        if params:
            self.param = dict()

    def export(self, name: str, index: bool = False) -> None:
        """Export signals and all data to an Excel (.xlsx) or CSV (.csv) file.

        Parameters:
            name: Output filepath. If the path ends with '.csv', signals are exported
                to CSV format. Otherwise, writes 'Signals' and 'all_data' sheets to Excel.
            index: Whether to write row index labels to the output file.
        """
        if name.endswith(".csv"):
            self.signals.to_csv(name, index=index)
            return

        try:
            with pd.ExcelWriter(name) as writer:
                self.signals.to_excel(writer, sheet_name="Signals", index=index)
                self.all_signals.to_excel(writer, sheet_name="all_data", index=index)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Excel (.xlsx) requires 'openpyxl'. "
                "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
            ) from exc

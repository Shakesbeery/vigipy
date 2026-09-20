from __future__ import annotations

import warnings
from typing import Any, Callable

import pandas as pd

from ..utils import convert, convert_binary, convert_multi_item
from ..utils.Container import AnalysisResult, DataContainer


TIME_UNIT_MAP = {
    "A": "YE",
    "Y": "YE",
    "Q": "QE",
    "M": "ME",
}


def _normalize_time_unit(unit: str) -> str:
    return TIME_UNIT_MAP.get(unit.upper(), unit)


class LongitudinalModel:

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_unit: str,
        count_col: str = "count",
    ) -> None:
        """
        Initialize the longitudinal model with raw data and a time unit.

        Arguments:
            dataframe: A dataframe containing counts, AEs, product/brands and AE dates.
            time_unit: One of Pandas' time unit aliases (e.g. 'YE', 'QE', 'ME' or legacy 'A', 'Q', 'M').
            count_col: Name of the event count column (defaults to 'count').
        """
        self.time_unit = time_unit
        self.data = dataframe.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])

        if count_col not in self.data.columns:
            if "events" in self.data.columns:
                self.count_col = "events"
            else:
                raise ValueError(
                    f"Count column '{count_col}' not found in dataframe. "
                    f"Available columns: {list(self.data.columns)}"
                )
        else:
            self.count_col = count_col

        self.date_groups = self.data.resample(_normalize_time_unit(self.time_unit), on="date")
        self.results: list[tuple[pd.Timestamp, AnalysisResult | None]] = []

    def _convert(
        self,
        data: pd.DataFrame,
        conversion_type: str,
        conversion_kwargs: dict[str, Any] | None,
    ) -> DataContainer:
        if conversion_type not in self.CONVERSION_TYPES:
            raise ValueError(f"Provided `conversion_type` not in {self.CONVERSION_TYPES}")

        if conversion_kwargs is None:
            conversion_kwargs = {}

        if conversion_type == "base":
            return convert(data, **conversion_kwargs)
        elif conversion_type == "binary":
            return convert_binary(data, **conversion_kwargs)
        elif conversion_type == "multi-item":
            return convert_multi_item(data, **conversion_kwargs)

    def run(
        self,
        model: Callable[..., AnalysisResult],
        include_gaps: bool = True,
        conversion_type: str = "base",
        conversion_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the longitudinal model cumulatively over time.

        Evaluates the analysis model on cumulative subsets of data up to each
        resampled time boundary.

        Parameters:
            model: Disproportionality analysis callable (e.g. prr, ror, bcpnn, gps).
            include_gaps: Whether to record (timestamp, None) for intervals with no events.
            conversion_type: Container conversion type ('base', 'binary', or 'multi-item').
            conversion_kwargs: Optional dictionary of keyword arguments passed to converter.
            **kwargs: Additional parameters forwarded to the analysis model callable.
        """
        self.results = []
        counts = self.date_groups[self.count_col].sum()
        for timestamp, count in counts.items():
            if count == 0:
                if include_gaps:
                    self.results.append((timestamp, None))
                continue

            subset = self.data.loc[self.data["date"] <= timestamp]
            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            self._run_model(model, sub_container, timestamp, include_gaps, kwargs)

    def run_disjoint(
        self,
        model: Callable[..., AnalysisResult],
        include_gaps: bool = True,
        conversion_type: str = "base",
        conversion_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the longitudinal model on disjoint time intervals.

        Evaluates the analysis model independently on each time window without
        accumulating historical events.

        Parameters:
            model: Disproportionality analysis callable (e.g. prr, ror, bcpnn, gps).
            include_gaps: Whether to record (timestamp, None) for intervals with no events.
            conversion_type: Container conversion type ('base', 'binary', or 'multi-item').
            conversion_kwargs: Optional dictionary of keyword arguments passed to converter.
            **kwargs: Additional parameters forwarded to the analysis model callable.
        """
        self.results = []
        counts = self.date_groups[self.count_col].sum()
        for count, (timestamp, subset) in zip(counts, self.date_groups):
            if count == 0:
                if include_gaps:
                    self.results.append((timestamp, None))
                continue

            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            self._run_model(model, sub_container, timestamp, include_gaps, kwargs)

    def _run_model(
        self,
        model: Callable[..., AnalysisResult],
        sub_container: DataContainer,
        timestamp: pd.Timestamp,
        include_gaps: bool,
        kwargs: dict[str, Any],
    ) -> None:
        try:
            da_results = model(sub_container, **kwargs)
            self.results.append((timestamp, da_results))
        except ValueError:
            warnings.warn(f"Insufficient data for this model. Skipping time slice: {timestamp}")
            if include_gaps:
                self.results.append((timestamp, None))

    def regroup_dates(self, time_unit: str) -> None:
        """Regroup the data by a new time unit alias.

        Parameters:
            time_unit: Resample frequency alias (e.g. 'YE', 'QE', 'ME', or legacy 'A', 'Q', 'M').
        """
        self.time_unit = time_unit
        self.date_groups = self.data.resample(_normalize_time_unit(self.time_unit), on="date")

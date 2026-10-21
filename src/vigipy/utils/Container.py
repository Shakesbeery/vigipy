from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
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

    @property
    def method(self) -> Optional[str]:
        """Method name that generated this result, if recorded in params."""
        if isinstance(self.params, dict):
            return self.params.get("method")
        return None

    @property
    def primary_metric(self) -> str:
        """Name of the primary effect-size or ranking column in the result."""
        candidates = [
            "AEGIS_EBGM", "EBGM", "PRR", "ROR", "RFET", "aROR", "SER",
            "SER_Interaction", "LLR", "CUSUM", "quantile", "log2", "p_value",
        ]
        source = self.signals if self.signals is not None and len(self.signals) > 0 else self.all_signals
        if source is not None:
            for cand in candidates:
                if cand in source.columns:
                    return cand
        return "Score"

    def summary(self, which: str = "signals") -> pd.DataFrame:
        """Return a standardized summary DataFrame with uniform columns across all methods.

        Columns: Product, Adverse Event, Count, Expected Count, Metric, Score, CI Lower, CI Upper, p_value, fdr.

        Parameters:
            which: 'signals' (default) or 'all'.
        """
        target = self.signals if which == "signals" else self.all_signals
        if target is None or len(target) == 0:
            return pd.DataFrame(columns=[
                "Product", "Adverse Event", "Count", "Expected Count",
                "Metric", "Score", "CI Lower", "CI Upper", "p_value", "fdr",
            ])

        metric = self.primary_metric
        score_series = target[metric] if metric in target.columns else pd.Series(np.nan, index=target.index)

        # Map confidence / credible intervals
        ci_lower = pd.Series(np.nan, index=target.index)
        ci_upper = pd.Series(np.nan, index=target.index)
        for cl, cu in [
            ("CI Lower", "CI Upper"),
            ("AEGIS_LB", "AEGIS_UB"),
            ("LowerBound", "UpperBound"),
            ("LowerCI", "UpperCI"),
            ("SER Lower", "SER Upper"),
            ("aROR Lower", "aROR Upper"),
        ]:
            if cl in target.columns:
                ci_lower = target[cl]
                if cu in target.columns:
                    ci_upper = target[cu]
                break

        if ci_lower.isna().all() and "quantile" in target.columns and metric != "quantile":
            ci_lower = target["quantile"]

        exp_col = "Expected Count" if "Expected Count" in target.columns else (
            "Expected" if "Expected" in target.columns else (
                "Expected_Null" if "Expected_Null" in target.columns else None
            )
        )
        exp_series = target[exp_col] if exp_col else pd.Series(np.nan, index=target.index)
        pval_series = target["p_value"] if "p_value" in target.columns else (
            target["posterior_probability"] if "posterior_probability" in target.columns else pd.Series(np.nan, index=target.index)
        )
        fdr_series = target["fdr"] if "fdr" in target.columns else pd.Series(np.nan, index=target.index)

        return pd.DataFrame({
            "Product": target["Product"].values,
            "Adverse Event": target["Adverse Event"].values,
            "Count": target["Count"].values if "Count" in target.columns else np.nan,
            "Expected Count": exp_series.values,
            "Metric": metric,
            "Score": score_series.values,
            "CI Lower": ci_lower.values,
            "CI Upper": ci_upper.values,
            "p_value": pval_series.values,
            "fdr": fdr_series.values,
        })

    def inspect_pair(self, product: str, adverse_event: str) -> pd.Series:
        """Lookup a specific drug-event pair across evaluated candidates.

        Returns a pandas Series containing all computed statistics for the pair,
        annotated with an 'is_signal' boolean flag.
        """
        source = self.all_signals if self.all_signals is not None else self.signals
        if source is None:
            raise KeyError(f"No evaluated pairs in AnalysisResult.")

        match = source[(source["Product"] == product) & (source["Adverse Event"] == adverse_event)]
        if match.empty:
            raise KeyError(f"Pair ({product!r}, {adverse_event!r}) not found in AnalysisResult.")

        row = match.iloc[0].copy()
        is_sig = False
        if self.signals is not None and not self.signals.empty:
            sig_match = self.signals[(self.signals["Product"] == product) & (self.signals["Adverse Event"] == adverse_event)]
            is_sig = not sig_match.empty
        row["is_signal"] = is_sig
        return row

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
        for cand in ["PRR", "ROR", "RFET", "aROR", "AEGIS_EBGM", "EBGM", "SER", "LLR", "CUSUM", "quantile", "p_value"]:
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
    """Container for converted input data used by disproportionality analysis methods.

    Attributes:
        data: Flattened DataFrame of drug-event pairs with counts (or report-level microdata).
        N: Total number of reports or adverse event counts.
        contingency: Optional contingency table DataFrame (products x events).
        product_features: Optional binary product indicator DataFrame (reports x products).
        event_outcomes: Optional binary event outcome DataFrame (reports x events).
        type: The conversion representation used ('contingency', 'binary', 'binary_report', etc.).
        covariates: Optional standardized per-report covariate DataFrame.
        feature_names: Product/feature column names.
        event_names: Adverse event column names.
        covariate_names: Covariate column names.
        pair_mapping: Multi-drug combination mapping for DDI analysis.
        fast_contingency: Optional recent-window contingency matrix for surge velocity.
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
    pair_mapping: Optional[dict[str, tuple[str, ...]]] = None
    fast_contingency: Optional[pd.DataFrame] = None
    raw_data: Optional[pd.DataFrame] = None

    @property
    def products(self) -> list[str]:
        """List of product / drug names in the container."""
        if self.contingency is not None:
            return [str(p) for p in self.contingency.index]
        if self.product_features is not None:
            return [str(p) for p in self.product_features.columns]
        if self.feature_names is not None:
            return [str(p) for p in self.feature_names]
        if "product_name" in self.data.columns:
            return [str(p) for p in self.data["product_name"].unique()]
        if "name" in self.data.columns:
            return [str(p) for p in self.data["name"].unique()]
        return []

    @property
    def events(self) -> list[str]:
        """List of adverse event terms in the container."""
        if self.contingency is not None:
            return [str(e) for e in self.contingency.columns]
        if self.event_outcomes is not None:
            return [str(e) for e in self.event_outcomes.columns]
        if self.event_names is not None:
            return [str(e) for e in self.event_names]
        if "ae_name" in self.data.columns:
            return [str(e) for e in self.data["ae_name"].unique()]
        if "AE" in self.data.columns:
            return [str(e) for e in self.data["AE"].unique()]
        return []

    @property
    def num_products(self) -> int:
        """Number of unique products in the container."""
        return len(self.products)

    @property
    def num_events(self) -> int:
        """Number of unique adverse events in the container."""
        return len(self.events)

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions of the drug-event space: (num_products, num_events)."""
        return (self.num_products, self.num_events)

    @property
    def is_sparse(self) -> bool:
        """Whether feature matrices are stored in sparse format."""
        if self.product_features is not None:
            return hasattr(self.product_features, "sparse")
        return False

    def ensure_contingency(self) -> pd.DataFrame:
        """Ensure contingency table is materialized and cached on the container."""
        if self.contingency is not None:
            return self.contingency

        if self.product_features is not None and self.event_outcomes is not None:
            X = self.product_features
            Y = self.event_outcomes
            prods = self.products
            evs = self.events

            import scipy.sparse as sp
            if hasattr(X, "sparse") or sp.issparse(X):
                X_mat = X.sparse.to_coo().tocsr() if hasattr(X, "sparse") else X.tocsr()
            else:
                X_mat = np.ascontiguousarray(X.values, dtype=np.float64)

            if hasattr(Y, "sparse") or sp.issparse(Y):
                Y_mat = Y.sparse.to_coo().tocsr() if hasattr(Y, "sparse") else Y.tocsr()
            else:
                Y_mat = np.ascontiguousarray(Y.values, dtype=np.float64)

            if sp.issparse(X_mat) or sp.issparse(Y_mat):
                C_raw = X_mat.T @ Y_mat
                C = C_raw.toarray() if sp.issparse(C_raw) else np.asarray(C_raw)
            else:
                C = X_mat.T @ Y_mat

            self.contingency = pd.DataFrame(C, index=prods, columns=evs)
            return self.contingency

        # Compute from data table if possible
        if "name" in self.data.columns and "AE" in self.data.columns:
            from .data_prep import compute_contingency
            count_col = "count" if "count" in self.data.columns else None
            if count_col is None:
                df_temp = self.data.copy()
                df_temp["count"] = 1
                count_col = "count"
            else:
                df_temp = self.data
            self.contingency = compute_contingency(df_temp, "name", count_col, "AE", margin_threshold=1)
            return self.contingency

        raise ValueError("Cannot materialize contingency table: no suitable data representation found.")

    def ensure_flattened_data(self) -> pd.DataFrame:
        """Ensure standard flattened DataFrame with ['events', 'product_aes', 'count_across_brands'] exists."""
        curr_data = super().__getattribute__("data")
        if (
            curr_data is not None
            and "events" in curr_data.columns
            and "product_aes" in curr_data.columns
            and "count_across_brands" in curr_data.columns
        ):
            return curr_data

        if self.raw_data is None:
            self.raw_data = curr_data

        cont = self.ensure_contingency()
        from .data_prep import count as flatten_count
        row_sums = np.sum(cont.values, axis=1)
        col_sums = np.sum(cont.values, axis=0)
        flattened = flatten_count(cont, row_sums, col_sums)
        self.data = flattened
        self.N = int(flattened["events"].sum()) if len(flattened) > 0 else 0
        return self.data

    def __getattribute__(self, name: str) -> Any:
        if name == "data":
            d = super().__getattribute__("data")
            if d is not None and "events" not in d.columns:
                try:
                    return self.ensure_flattened_data()
                except Exception:
                    return d
            return d
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        shape_str = f"{self.num_products} products x {self.num_events} events"
        sparse_str = ", sparse" if self.is_sparse else ""
        cov_str = f", {len(self.covariate_names)} covariates" if self.covariate_names else ""
        return f"<DataContainer [{self.type}]: {shape_str} (N={self.N:,}{sparse_str}{cov_str})>"


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

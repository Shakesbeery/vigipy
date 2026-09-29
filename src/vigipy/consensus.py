"""Consensus analysis module for cross-method disproportionality comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd

from .analyze import analyze_all
from .config import MethodConfig
from .utils.Container import AnalysisResult, DataContainer


# Candidate columns for primary disproportionality scores
_PRIMARY_STAT_CANDIDATES = [
    "PRR",
    "ROR",
    "RFET",
    "IC",
    "quantile",
    "EBGM",
    "log2",
    "aROR",
    "LASSO Coefficient",
    "count/expected",
]


def _extract_method_data(
    method: str, result: AnalysisResult
) -> tuple[str, pd.DataFrame]:
    """Extract and standardize columns from an individual AnalysisResult."""
    df = result.all_signals.copy()

    # 1. Determine primary score column
    score_col = None
    for cand in _PRIMARY_STAT_CANDIDATES:
        if cand in df.columns:
            score_col = cand
            break

    metric_name = score_col if score_col is not None else "Score"

    # 2. Determine bounds (CI or credible intervals)
    ci_lower_col, ci_upper_col = None, None
    if score_col == "aROR" and "aROR Lower" in df.columns:
        ci_lower_col = "aROR Lower"
        ci_upper_col = "aROR Upper" if "aROR Upper" in df.columns else None
    else:
        for cand_l, cand_u in [
            ("CI Lower", "CI Upper"),
            ("LowerBound", "UpperBound"),
            ("aROR Lower", "aROR Upper"),
        ]:
            if cand_l in df.columns:
                ci_lower_col = cand_l
                ci_upper_col = cand_u if cand_u in df.columns else None
                break

    # If quantile was not chosen as the primary score, it may act as lower bound (e.g. BCPNN IC_025)
    if ci_lower_col is None and "quantile" in df.columns and score_col != "quantile":
        ci_lower_col = "quantile"

    p_val_col = (
        "p_value"
        if "p_value" in df.columns
        else ("posterior_probability" if "posterior_probability" in df.columns else None)
    )
    fdr_col = "fdr" if "fdr" in df.columns else None

    # 3. Build alert set
    sig_df = result.signals
    alert_pairs = set(
        zip(sig_df["Product"].astype(str), sig_df["Adverse Event"].astype(str))
    )
    is_alert = [
        pair in alert_pairs
        for pair in zip(df["Product"].astype(str), df["Adverse Event"].astype(str))
    ]

    cols_dict: dict[str, Any] = {
        "Product": df["Product"].astype(str),
        "Adverse Event": df["Adverse Event"].astype(str),
        f"alert_{method}": is_alert,
        f"score_{method}": df[score_col].astype(float) if score_col else np.nan,
    }
    if "Count" in df.columns:
        cols_dict["Count"] = df["Count"].astype(float)
    if "Expected Count" in df.columns:
        cols_dict["Expected Count"] = df["Expected Count"].astype(float)
    if ci_lower_col:
        cols_dict[f"ci_lower_{method}"] = df[ci_lower_col].astype(float)
    if ci_upper_col:
        cols_dict[f"ci_upper_{method}"] = df[ci_upper_col].astype(float)
    if p_val_col:
        cols_dict[f"p_value_{method}"] = df[p_val_col].astype(float)
    if fdr_col:
        cols_dict[f"fdr_{method}"] = df[fdr_col].astype(float)

    return metric_name, pd.DataFrame(cols_dict)


def _compute_jaccard(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute Jaccard similarity between two boolean alert arrays."""
    inter = np.sum(b1 & b2)
    union = np.sum(b1 | b2)
    if union == 0:
        return 1.0
    return float(inter / union)


def _compute_cohens_kappa(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute Cohen's Kappa inter-rater agreement between two binary alert vectors."""
    n = len(b1)
    if n == 0:
        return 1.0

    n11 = np.sum(b1 & b2)
    n10 = np.sum(b1 & (~b2))
    n01 = np.sum((~b1) & b2)
    n00 = np.sum((~b1) & (~b2))

    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)

    if np.isclose(1.0 - pe, 0.0):
        return 1.0 if np.isclose(po, 1.0) else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return float(np.clip(kappa, -1.0, 1.0))


def _assign_agreement_tier(
    votes: np.ndarray, score: np.ndarray, num_methods: int
) -> np.ndarray:
    """Assign consensus agreement tier based on method votes and normalized score."""
    n = len(votes)
    tiers = np.empty(n, dtype=object)

    if num_methods == 1:
        tiers[votes == 1] = "Unanimous"
        tiers[votes == 0] = "None"
        return tiers

    tiers[votes == 0] = "None"
    tiers[votes == 1] = "Isolated"
    tiers[votes >= 2] = "Weak"
    tiers[score >= 0.50] = "Moderate"
    tiers[score >= 0.75] = "Strong"
    tiers[votes == num_methods] = "Unanimous"
    return tiers


@dataclass
class ConsensusResult:
    """Result container returned by consensus_analysis across multiple DA methods.

    Attributes:
        comparison_table: Master DataFrame aligning all candidate drug-event pairs,
            individual method alert flags, scores, confidence bounds, p-values,
            composite ranks, consensus scores, and agreement tiers.
        signals: Subset DataFrame of pairs meeting the min_consensus threshold.
        num_signals: Total count of consensus signals.
        method_agreement: Dictionary containing method-level agreement matrices:
            - 'jaccard': Pairwise Jaccard similarity of alert sets.
            - 'kappa': Pairwise Cohen's Kappa inter-rater concordance.
            - 'correlation': Pairwise Spearman rank correlation of primary scores.
            - 'overlap': Pairwise count matrix of shared alert pairs.
        raw_results: Dictionary of {method_name: AnalysisResult} from each individual method.
        metric_names: Dictionary mapping method names to their primary metric name.
        params: Dictionary of analysis parameters and metadata.
    """

    comparison_table: pd.DataFrame
    signals: pd.DataFrame
    num_signals: int
    method_agreement: dict[str, pd.DataFrame]
    raw_results: dict[str, AnalysisResult]
    metric_names: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def methods(self) -> list[str]:
        """List of method names included in the consensus analysis."""
        return list(self.raw_results.keys())

    def inspect_signal(self, product: str, adverse_event: str) -> pd.DataFrame:
        """Detailed multi-method inspection for a specific product and adverse event pair.

        Parameters:
            product: The product / drug name.
            adverse_event: The adverse event name.

        Returns:
            pd.DataFrame: A formatted table showing each method's alert status, primary metric name,
                score value, confidence/credibility interval, p-value, FDR, and pair counts.
        """
        prod_str = str(product)
        ae_str = str(adverse_event)
        match = self.comparison_table[
            (self.comparison_table["Product"] == prod_str)
            & (self.comparison_table["Adverse Event"] == ae_str)
        ]
        if match.empty:
            raise KeyError(
                f"Signal pair ({product!r}, {adverse_event!r}) not found in consensus analysis."
            )

        row = match.iloc[0]
        records = []
        for m in self.methods:
            records.append(
                {
                    "Method": m.upper(),
                    "Alert": bool(row.get(f"alert_{m}", False)),
                    "Metric": self.metric_names.get(m, "Score"),
                    "Score": row.get(f"score_{m}", np.nan),
                    "CI Lower": row.get(f"ci_lower_{m}", np.nan),
                    "CI Upper": row.get(f"ci_upper_{m}", np.nan),
                    "p-value": row.get(f"p_value_{m}", np.nan),
                    "FDR": row.get(f"fdr_{m}", np.nan),
                    "Count": row.get("Count", np.nan),
                    "Expected Count": row.get("Expected Count", np.nan),
                }
            )
        df_inspect = pd.DataFrame(records)
        df_inspect.attrs["Product"] = prod_str
        df_inspect.attrs["Adverse Event"] = ae_str
        df_inspect.attrs["votes"] = row.get("votes", 0)
        df_inspect.attrs["total_methods"] = row.get("total_methods", len(self.methods))
        df_inspect.attrs["consensus_score"] = row.get("consensus_score", 0.0)
        df_inspect.attrs["agreement_tier"] = row.get("agreement_tier", "None")
        df_inspect.attrs["composite_rank"] = row.get("composite_rank", np.nan)
        return df_inspect

    def contingency_table(self, method_a: str, method_b: str) -> pd.DataFrame:
        """Return a 2x2 contingency table of alert agreement between two methods.

        Parameters:
            method_a: First method name (e.g. 'prr').
            method_b: Second method name (e.g. 'gps').

        Returns:
            pd.DataFrame: 2x2 contingency table with marginal totals.
        """
        ma = str(method_a).lower()
        mb = str(method_b).lower()
        col_a = f"alert_{ma}"
        col_b = f"alert_{mb}"
        if col_a not in self.comparison_table.columns:
            raise ValueError(f"Method {method_a!r} not in consensus analysis results.")
        if col_b not in self.comparison_table.columns:
            raise ValueError(f"Method {method_b!r} not in consensus analysis results.")

        s_a = pd.Categorical(self.comparison_table[col_a], categories=[False, True])
        s_b = pd.Categorical(self.comparison_table[col_b], categories=[False, True])
        ct = pd.crosstab(
            pd.Series(s_a, name=f"{ma.upper()} Alert"),
            pd.Series(s_b, name=f"{mb.upper()} Alert"),
            margins=True,
            margins_name="Total",
            dropna=False,
        )
        return ct

    def export(self, filepath: str, index: bool = False) -> None:
        """Export consensus signals, comparison table, and method agreement matrices.

        Parameters:
            filepath: Output filepath. If the path ends with '.csv', signals are exported
                to CSV format. If '.xlsx', writes a multi-sheet workbook including
                'Consensus Signals', 'Comparison Table', and agreement matrices.
            index: Whether to write row index labels to the output file (for signals/comparison).
                Agreement matrices are always exported with method names as index labels.
        """
        if filepath.endswith(".csv"):
            self.signals.to_csv(filepath, index=index)
            return

        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                self.signals.to_excel(
                    writer, sheet_name="Consensus Signals", index=index
                )
                self.comparison_table.to_excel(
                    writer, sheet_name="Comparison Table", index=index
                )
                if "jaccard" in self.method_agreement:
                    self.method_agreement["jaccard"].to_excel(
                        writer, sheet_name="Jaccard Similarity", index=True
                    )
                if "kappa" in self.method_agreement:
                    self.method_agreement["kappa"].to_excel(
                        writer, sheet_name="Cohens Kappa", index=True
                    )
                if "correlation" in self.method_agreement:
                    self.method_agreement["correlation"].to_excel(
                        writer, sheet_name="Spearman Correlation", index=True
                    )
                if "overlap" in self.method_agreement:
                    self.method_agreement["overlap"].to_excel(
                        writer, sheet_name="Alert Overlap", index=True
                    )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Exporting to Excel (.xlsx) requires 'openpyxl'. "
                "Install it with 'pip install openpyxl' or 'pip install vigipy[excel]'."
            ) from exc

    def __repr__(self) -> str:
        return (
            f"ConsensusResult(num_signals={self.num_signals}, "
            f"total_pairs={len(self.comparison_table)}, "
            f"methods={self.methods})"
        )


def consensus_analysis(
    data: Union[DataContainer, dict[str, AnalysisResult]],
    configs: list[MethodConfig] | None = None,
    min_consensus: int | float = 2,
    weights: dict[str, float] | None = None,
    **shared_overrides: Any,
) -> ConsensusResult:
    """Compare and synthesize results across multiple disproportionality analysis (DA) methods.

    Parameters:
        data: A DataContainer instance or a dictionary of pre-computed {method_name: AnalysisResult}.
            If a DataContainer is passed, methods are executed using analyze_all().
        configs: List of MethodConfig instances. If None, runs PRR, ROR, RFET, BCPNN, and GPS
            with their standard regulatory defaults.
        min_consensus: Threshold for filtering consensus signals.
            - If int (or float > 1.0): Minimum number of method alert votes required (e.g. 2 or 3).
            - If float in (0.0, 1.0]: Minimum normalized consensus score required (e.g. 0.5 for 50%).
        weights: Optional dictionary mapping method names to importance weights (e.g. {"gps": 1.5}).
            Defaults to uniform weights (1.0 for each method).
        **shared_overrides: Parameter overrides passed to all method configurations (e.g. min_events=3).

    Returns:
        ConsensusResult: A unified container containing the full comparison table, filtered
            consensus signals, method agreement matrices (Jaccard, Kappa, Spearman, Overlap),
            raw results from each method, and inspection tools.

    Examples:
        >>> from vigipy import convert, consensus_analysis
        >>> cont = convert(df)
        >>> res = consensus_analysis(cont, min_events=3, min_consensus=3)
        >>> print(f"Found {res.num_signals} consensus signals")
        >>> print(res.method_agreement["jaccard"])
        >>> res.inspect_signal("DRUGA", "CARDIAC_ARREST")
    """
    if isinstance(data, dict):
        raw_results = {str(k).lower(): v for k, v in data.items()}
    elif isinstance(data, DataContainer):
        raw_results = analyze_all(data, configs=configs, **shared_overrides)
    else:
        raise TypeError(
            f"Expected data to be a DataContainer or dict[str, AnalysisResult], got {type(data).__name__}"
        )

    if not raw_results:
        raise ValueError("No analysis results provided or generated.")

    methods = list(raw_results.keys())
    metric_names: dict[str, str] = {}
    master_df: pd.DataFrame | None = None

    # Standardize and merge results from each method
    for m in methods:
        res = raw_results[m]
        metric_name, sub_df = _extract_method_data(m, res)
        sub_df = sub_df.drop_duplicates(subset=["Product", "Adverse Event"], keep="first")
        metric_names[m] = metric_name

        if master_df is None:
            master_df = sub_df
        else:
            master_df = pd.merge(
                master_df,
                sub_df,
                on=["Product", "Adverse Event"],
                how="outer",
                suffixes=("", "_new"),
            )
            for c in ["Count", "Expected Count"]:
                if f"{c}_new" in master_df.columns:
                    if c in master_df.columns:
                        master_df[c] = master_df[c].fillna(master_df[f"{c}_new"])
                    else:
                        master_df[c] = master_df[f"{c}_new"]
                    master_df.drop(columns=[f"{c}_new"], inplace=True)

    assert master_df is not None

    # Ensure all alert columns are boolean
    for m in methods:
        if f"alert_{m}" not in master_df.columns:
            master_df[f"alert_{m}"] = False
        else:
            col_vals = master_df[f"alert_{m}"]
            master_df[f"alert_{m}"] = np.where(
                col_vals.isna(), False, col_vals
            ).astype(bool)

    # Cast Count to integer if all counts are non-null
    if "Count" in master_df.columns and master_df["Count"].notna().all():
        master_df["Count"] = master_df["Count"].astype(int)

    # Compute votes
    alert_cols = [f"alert_{m}" for m in methods]
    master_df["votes"] = master_df[alert_cols].sum(axis=1).astype(int)
    master_df["total_methods"] = len(methods)

    # Compute normalized consensus score
    if weights is not None:
        norm_weights = {
            str(k).lower(): float(v)
            for k, v in weights.items()
            if str(k).lower() in methods
        }
        for m in methods:
            if m not in norm_weights:
                norm_weights[m] = 1.0
        total_w = sum(norm_weights.values())
        if total_w > 0:
            weighted_sum = sum(
                master_df[f"alert_{m}"].astype(float) * norm_weights[m] for m in methods
            )
            master_df["consensus_score"] = (weighted_sum / total_w).astype(float)
        else:
            master_df["consensus_score"] = (
                master_df["votes"] / len(methods)
            ).astype(float)
    else:
        master_df["consensus_score"] = (
            master_df["votes"] / len(methods)
        ).astype(float)

    # Assign consensus agreement tier
    master_df["agreement_tier"] = _assign_agreement_tier(
        master_df["votes"].values, master_df["consensus_score"].values, len(methods)
    )

    # Sort comparison table by consensus strength and count
    sort_cols = ["votes", "consensus_score"]
    ascending_order = [False, False]
    if "Count" in master_df.columns:
        sort_cols.append("Count")
        ascending_order.append(False)

    master_df = master_df.sort_values(
        by=sort_cols, ascending=ascending_order
    ).reset_index(drop=True)
    master_df["composite_rank"] = np.arange(1, len(master_df) + 1)

    # Organize column ordering: group alerts together, then scores, CIs, p-values, FDRs
    leading_cols = [
        "Product",
        "Adverse Event",
        "Count",
        "Expected Count",
        "composite_rank",
        "votes",
        "total_methods",
        "consensus_score",
        "agreement_tier",
    ]
    existing_leading = [c for c in leading_cols if c in master_df.columns]
    systematic_method_cols = []
    for prefix in ("alert_", "score_", "ci_lower_", "ci_upper_", "p_value_", "fdr_"):
        for m in methods:
            col_name = f"{prefix}{m}"
            if col_name in master_df.columns:
                systematic_method_cols.append(col_name)
    remaining_cols = [
        c
        for c in master_df.columns
        if c not in existing_leading and c not in systematic_method_cols
    ]
    ordered_cols = existing_leading + systematic_method_cols + remaining_cols
    master_df = master_df[ordered_cols]

    # Filter consensus signals
    if isinstance(min_consensus, float) and 0.0 < min_consensus <= 1.0:
        signal_mask = master_df["consensus_score"] >= (min_consensus - 1e-9)
    else:
        min_votes_int = int(min_consensus)
        signal_mask = master_df["votes"] >= min_votes_int

    signals_df = master_df[signal_mask].copy().reset_index(drop=True)
    num_signals = len(signals_df)

    # Compute Method Agreement Matrices
    # 1. Alert Overlap Matrix
    overlap_df = pd.DataFrame(index=methods, columns=methods, dtype=int)
    for m1 in methods:
        for m2 in methods:
            overlap_df.loc[m1, m2] = int(
                (master_df[f"alert_{m1}"] & master_df[f"alert_{m2}"]).sum()
            )

    # 2. Jaccard Similarity Matrix
    jaccard_df = pd.DataFrame(index=methods, columns=methods, dtype=float)
    # 3. Cohen's Kappa Concordance Matrix
    kappa_df = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                jaccard_df.loc[m1, m2] = 1.0
                kappa_df.loc[m1, m2] = 1.0
            elif i < j:
                b1 = master_df[f"alert_{m1}"].values
                b2 = master_df[f"alert_{m2}"].values
                j_val = _compute_jaccard(b1, b2)
                k_val = _compute_cohens_kappa(b1, b2)
                jaccard_df.loc[m1, m2] = j_val
                jaccard_df.loc[m2, m1] = j_val
                kappa_df.loc[m1, m2] = k_val
                kappa_df.loc[m2, m1] = k_val

    # 4. Spearman Rank Correlation Matrix of Primary Scores
    score_cols = [f"score_{m}" for m in methods]
    scores_sub = master_df[score_cols].rename(
        columns={f"score_{m}": m for m in methods}
    )
    correlation_df = scores_sub.corr(method="spearman")
    np.fill_diagonal(correlation_df.values, 1.0)

    method_agreement = {
        "jaccard": jaccard_df,
        "kappa": kappa_df,
        "correlation": correlation_df,
        "overlap": overlap_df,
    }

    params = {
        "methods": methods,
        "min_consensus": min_consensus,
        "weights": weights,
        "shared_overrides": shared_overrides,
    }

    return ConsensusResult(
        comparison_table=master_df,
        signals=signals_df,
        num_signals=num_signals,
        method_agreement=method_agreement,
        raw_results=raw_results,
        metric_names=metric_names,
        params=params,
    )

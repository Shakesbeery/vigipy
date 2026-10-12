from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
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


def _parse_half_life_days(decay_half_life: str | pd.Timedelta | float | int | None) -> float | None:
    """Parse half-life specification into an elapsed duration in days."""
    if decay_half_life is None:
        return None

    if isinstance(decay_half_life, (int, float, np.number)):
        val = float(decay_half_life)
        if val <= 0:
            raise ValueError(f"decay_half_life must be positive (got {val}).")
        return val

    if isinstance(decay_half_life, pd.Timedelta):
        val = decay_half_life.total_seconds() / 86400.0
        if val <= 0:
            raise ValueError(f"decay_half_life must be positive (got {decay_half_life}).")
        return val

    if isinstance(decay_half_life, str):
        cleaned = decay_half_life.strip().upper()
        if cleaned.endswith("YE") or cleaned.endswith("Y"):
            num_part = cleaned.rstrip("YE").rstrip("Y").strip()
            num = float(num_part) if num_part else 1.0
            return num * 365.25
        elif cleaned.endswith("ME") or cleaned.endswith("M"):
            num_part = cleaned.rstrip("ME").rstrip("M").strip()
            num = float(num_part) if num_part else 1.0
            return num * 30.4375
        elif cleaned.endswith("W"):
            num_part = cleaned.rstrip("W").strip()
            num = float(num_part) if num_part else 1.0
            return num * 7.0
        elif cleaned.endswith("D"):
            num_part = cleaned.rstrip("D").strip()
            num = float(num_part) if num_part else 1.0
            return num
        else:
            try:
                td = pd.to_timedelta(cleaned)
                val = td.total_seconds() / 86400.0
                if val <= 0:
                    raise ValueError(f"decay_half_life must be positive (got {decay_half_life}).")
                return val
            except Exception as exc:
                raise ValueError(
                    f"Unable to parse decay_half_life '{decay_half_life}'. "
                    f"Use a duration string like '2Y', '365D', '6M', or a numeric day count."
                ) from exc

    raise TypeError(f"decay_half_life must be str, Timedelta, or number, got {type(decay_half_life).__name__}")


def _apply_time_decay(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    half_life_days: float,
    count_col: str,
) -> pd.DataFrame:
    """Weight event counts in df using continuous exponential half-life decay relative to timestamp."""
    ts_ns = np.datetime64(timestamp, "ns")
    dates_ns = df["date"].values.astype("datetime64[ns]")
    age_days = np.maximum(0.0, (ts_ns - dates_ns).astype(np.float64) / 86_400_000_000_000.0)
    weights = np.exp2(-age_days / half_life_days)
    weighted_counts = df[count_col].to_numpy(dtype=np.float64, copy=False) * weights
    res = df.copy(deep=False)
    res[count_col] = weighted_counts
    return res


def _build_container_from_matrix(
    mat: np.ndarray,
    present_p: np.ndarray,
    present_a: np.ndarray,
    p_uniques: pd.Index,
    ae_uniques: pd.Index,
    product_label: str,
    ae_label: str,
    margin_threshold: int,
) -> DataContainer:
    """Construct a contingency DataContainer directly from a pre-aggregated 2D matrix."""
    from ..utils.data_prep import count as flatten_count

    mat_sub = mat[np.ix_(present_p, present_a)]
    p_sub = p_uniques[present_p]
    a_sub = ae_uniques[present_a]

    if margin_threshold > 1:
        r_tot = np.sum(mat_sub, axis=1) >= margin_threshold
        c_tot = np.sum(mat_sub, axis=0) >= margin_threshold
        mat_sub = mat_sub[np.ix_(r_tot, c_tot)]
        p_sub = p_sub[r_tot]
        a_sub = a_sub[c_tot]

    r_sums_mask = np.sum(mat_sub, axis=1) >= margin_threshold
    c_sums_mask = np.sum(mat_sub, axis=0) >= margin_threshold
    mat_sub = mat_sub[np.ix_(r_sums_mask, c_sums_mask)]
    p_sub = p_sub[r_sums_mask]
    a_sub = a_sub[c_sums_mask]

    data_cont = pd.DataFrame(mat_sub, index=p_sub, columns=a_sub)
    data_cont.index.name = product_label
    data_cont.columns.name = ae_label

    col_sums = np.sum(data_cont, axis=0)
    row_sums = np.sum(data_cont, axis=1)
    data_df = flatten_count(data_cont, row_sums, col_sums)

    return DataContainer(
        data=data_df,
        N=data_df["events"].sum(),
        contingency=data_cont,
        type="contingency",
    )


def _fit_longitudinal_slice(
    model: Callable[..., AnalysisResult],
    data_input: pd.DataFrame | DataContainer | None,
    timestamp: pd.Timestamp,
    include_gaps: bool,
    kwargs: dict[str, Any] | None,
    store_all_signals: bool = True,
    conversion_type: str = "base",
    conversion_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.Timestamp, AnalysisResult | None] | None:
    """Worker function for executing data conversion and analysis on a single time slice.

    Defined at module level for clean joblib multiprocessing serialization on Windows.
    Accepts either an already-converted DataContainer or a raw slice DataFrame to parallelize
    both conversion and model fitting across CPU cores.
    """
    if data_input is None:
        return (timestamp, None) if include_gaps else None

    if isinstance(data_input, pd.DataFrame):
        if len(data_input) == 0:
            return (timestamp, None) if include_gaps else None
        c_kw = conversion_kwargs or {}
        if conversion_type == "base":
            sub_container = convert(data_input, **c_kw)
        elif conversion_type == "binary":
            sub_container = convert_binary(data_input, **c_kw)
        elif conversion_type == "multi-item":
            sub_container = convert_multi_item(data_input, **c_kw)
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")
    else:
        sub_container = data_input

    try:
        da_results = model(sub_container, **(kwargs or {}))
        if not store_all_signals and hasattr(da_results, "all_signals"):
            da_results.all_signals = pd.DataFrame()
        return (timestamp, da_results)
    except ValueError:
        warnings.warn(f"Insufficient data for this model. Skipping time slice: {timestamp}")
        return (timestamp, None) if include_gaps else None


class LongitudinalModel:
    """Longitudinal Pharmacovigilance Surveillance Model.

    Clinical Intuition:
        Adverse drug reactions do not occur in a static batch; they arrive sequentially
        over months and years as patient exposure expands in the general population.
        A drug may show no statistical disproportionality in year 1 with 50 reports, but
        reach unequivocal statistical significance by year 3 with 500 reports.

        Longitudinal modeling allows safety teams to simulate or execute prospective
        surveillance:
        - **Cumulative Surveillance (`run`)**: Replicates real-world post-marketing drug
          safety monitoring. Slices data cumulatively up to each time boundary, evaluating
          how evidence evolves over time and precisely when a safety signal first breached
          significance thresholds (e.g. for Periodic Safety Update Reports / PSURs).
        - **Disjoint Surveillance (`run_disjoint`)**: Analyzes discrete, non-overlapping
          time intervals (e.g., quarterly or annual windows). Ideal for detecting transient
          manufacturing batch anomalies, seasonal fluctuations, sudden label-change impacts,
          or media-driven notoriety spikes.

    Arguments:
        dataframe: A dataframe containing counts, AEs, product/brands and AE dates.
        time_unit: One of Pandas' time unit aliases (e.g. 'YE', 'QE', 'ME' or legacy 'A', 'Q', 'M').
        count_col: Name of the event count column (defaults to 'count').
    """

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_unit: str,
        count_col: str = "count",
        decay_half_life: str | pd.Timedelta | float | None = None,
    ) -> None:
        self.time_unit = time_unit
        self.decay_half_life = decay_half_life
        self.data = dataframe.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])
        # Pre-sort chronologically for O(log N) binary search slicing
        self.data = self.data.sort_values(by="date").reset_index(drop=True)
        self._sorted_dates = self.data["date"].values

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

    def _prepare_base_index(
        self, conversion_kwargs: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        c_kw = conversion_kwargs or {}
        product_label = c_kw.get("product_label", "name")
        ae_label = c_kw.get("ae_label", "AE")
        count_label = c_kw.get("count_label", self.count_col)
        margin_threshold = c_kw.get("margin_threshold", 1)

        if (
            not isinstance(product_label, str)
            or not isinstance(ae_label, str)
            or product_label not in self.data.columns
            or ae_label not in self.data.columns
            or count_label not in self.data.columns
        ):
            return None

        p_codes, p_uniques = pd.factorize(self.data[product_label], sort=True)
        ae_codes, ae_uniques = pd.factorize(self.data[ae_label], sort=True)
        if np.any(p_codes < 0) or np.any(ae_codes < 0) or len(p_uniques) == 0 or len(ae_uniques) == 0:
            return None

        n_p = len(p_uniques)
        n_a = len(ae_uniques)
        flat_idx = p_codes * n_a + ae_codes
        counts_arr = self.data[count_label].to_numpy(dtype=np.float64, copy=False)
        is_int = np.issubdtype(self.data[count_label].dtype, np.integer)

        return {
            "p_codes": p_codes,
            "p_uniques": p_uniques,
            "ae_codes": ae_codes,
            "ae_uniques": ae_uniques,
            "n_p": n_p,
            "n_a": n_a,
            "flat_idx": flat_idx,
            "counts_arr": counts_arr,
            "is_int": is_int,
            "orig_dtype": self.data[count_label].dtype,
            "product_label": product_label,
            "ae_label": ae_label,
            "margin_threshold": margin_threshold,
        }

    def run(
        self,
        model: Callable[..., AnalysisResult],
        include_gaps: bool = True,
        conversion_type: str = "base",
        conversion_kwargs: dict[str, Any] | None = None,
        n_jobs: int = 1,
        warm_start: bool = False,
        store_all_signals: bool = True,
        decay_half_life: str | pd.Timedelta | float | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the longitudinal model cumulatively over time.

        Evaluates the analysis model on cumulative subsets of data up to each
        resampled time boundary. Uses binary search for fast slicing and supports
        multi-core parallelism as well as Bayesian hyperprior warm-starting.

        Parameters:
            model: Disproportionality analysis callable (e.g. prr, ror, bcpnn, gps).
            include_gaps: Whether to record (timestamp, None) for intervals with no events.
            conversion_type: Container conversion type ('base', 'binary', or 'multi-item').
            conversion_kwargs: Optional dictionary of keyword arguments passed to converter.
            n_jobs: Number of CPU workers for parallel slice execution (default: 1).
            warm_start: If True, uses previous slice's fitted priors to warm-start GPS optimization.
            store_all_signals: If False, prunes all_signals to save memory across slices.
            decay_half_life: Optional continuous exponential half-life for historical reports
                (e.g., '2Y', '730D', '6M', or a float number of days). When enabled, historical
                report counts decay smoothly as 2^(-elapsed_time / half_life) relative to each
                evaluated time slice boundary. Allows out-of-date historical bursts to naturally
                attenuate over time unless refreshed by new reports. Defaults to None (disabled).
            **kwargs: Additional parameters forwarded to the analysis model callable.
        """
        self.results = []
        effective_decay = decay_half_life if decay_half_life is not None else self.decay_half_life
        hl_days = _parse_half_life_days(effective_decay)
        counts = self.date_groups[self.count_col].sum()
        is_gps = (getattr(model, "__name__", "") == "gps")

        if warm_start and n_jobs != 1:
            warnings.warn(
                "warm_start is only supported in sequential execution (n_jobs=1). "
                "Disabling warm_start for parallel execution.",
                UserWarning,
                stacklevel=2,
            )

        fast_idx = self._prepare_base_index(conversion_kwargs) if conversion_type == "base" else None
        if fast_idx is not None and hl_days is None:
            cum_mat = np.zeros((fast_idx["n_p"], fast_idx["n_a"]), dtype=np.float64)
            present_p = np.zeros(fast_idx["n_p"], dtype=bool)
            present_a = np.zeros(fast_idx["n_a"], dtype=bool)
            prev_idx = 0

        def _get_cumulative_container(timestamp: pd.Timestamp, idx: int) -> DataContainer:
            nonlocal cum_mat, present_p, present_a, prev_idx
            if fast_idx is not None:
                n_p, n_a = fast_idx["n_p"], fast_idx["n_a"]
                flat_idx = fast_idx["flat_idx"]
                counts_arr = fast_idx["counts_arr"]
                p_codes, ae_codes = fast_idx["p_codes"], fast_idx["ae_codes"]

                if hl_days is None:
                    if idx > prev_idx:
                        delta = np.bincount(
                            flat_idx[prev_idx:idx],
                            weights=counts_arr[prev_idx:idx],
                            minlength=n_p * n_a,
                        ).reshape(n_p, n_a)
                        cum_mat = cum_mat + delta
                        present_p[p_codes[prev_idx:idx]] = True
                        present_a[ae_codes[prev_idx:idx]] = True
                        prev_idx = idx
                    mat_use = cum_mat.astype(fast_idx["orig_dtype"], copy=False) if fast_idx["is_int"] else cum_mat
                    return _build_container_from_matrix(
                        mat_use,
                        present_p,
                        present_a,
                        fast_idx["p_uniques"],
                        fast_idx["ae_uniques"],
                        fast_idx["product_label"],
                        fast_idx["ae_label"],
                        fast_idx["margin_threshold"],
                    )
                else:
                    ts_ns = np.datetime64(timestamp, "ns")
                    age_days = np.maximum(
                        0.0, (ts_ns - self._sorted_dates[:idx]).astype(np.float64) / 86_400_000_000_000.0
                    )
                    w_counts = counts_arr[:idx] * np.exp2(-age_days / hl_days)
                    mat = np.bincount(flat_idx[:idx], weights=w_counts, minlength=n_p * n_a).reshape(n_p, n_a)
                    p_pres = np.bincount(p_codes[:idx], minlength=n_p) > 0
                    a_pres = np.bincount(ae_codes[:idx], minlength=n_a) > 0
                    return _build_container_from_matrix(
                        mat,
                        p_pres,
                        a_pres,
                        fast_idx["p_uniques"],
                        fast_idx["ae_uniques"],
                        fast_idx["product_label"],
                        fast_idx["ae_label"],
                        fast_idx["margin_threshold"],
                    )

            subset = self.data.iloc[:idx]
            if hl_days is not None:
                subset = _apply_time_decay(subset, timestamp, hl_days, self.count_col)
            return self._convert(subset, conversion_type, conversion_kwargs)

        if n_jobs != 1:
            tasks = []
            for timestamp, count in counts.items():
                if count == 0:
                    if include_gaps:
                        tasks.append((timestamp, None, None))
                    continue

                idx = int(np.searchsorted(self._sorted_dates, np.datetime64(timestamp), side="right"))
                if idx == 0:
                    if include_gaps:
                        tasks.append((timestamp, None, None))
                    continue

                sub_container = _get_cumulative_container(timestamp, idx)
                tasks.append((timestamp, sub_container, dict(kwargs)))

            from joblib import Parallel, delayed

            raw_results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_longitudinal_slice)(
                    model,
                    sub_cont,
                    ts,
                    include_gaps,
                    kw,
                    store_all_signals,
                    conversion_type,
                    conversion_kwargs,
                )
                for ts, sub_cont, kw in tasks
            )
            self.results = [r for r in raw_results if r is not None]
        else:
            # Sequential slice processing with warm-start support
            last_prior_param = None
            for timestamp, count in counts.items():
                if count == 0:
                    if include_gaps:
                        self.results.append((timestamp, None))
                    continue

                idx = int(np.searchsorted(self._sorted_dates, np.datetime64(timestamp), side="right"))
                if idx == 0:
                    if include_gaps:
                        self.results.append((timestamp, None))
                    continue

                sub_container = _get_cumulative_container(timestamp, idx)

                slice_kwargs = dict(kwargs)
                if warm_start and is_gps and last_prior_param is not None and "prior_param" not in slice_kwargs:
                    slice_kwargs["prior_init"] = {
                        "alpha1": float(last_prior_param[0]),
                        "beta1": float(last_prior_param[1]),
                        "alpha2": float(last_prior_param[2]),
                        "beta2": float(last_prior_param[3]),
                        "w": float(last_prior_param[4]),
                    }

                res = self._run_model(
                    model, sub_container, timestamp, include_gaps, slice_kwargs, store_all_signals
                )
                if res is not None and getattr(res, "params", None):
                    p = res.params.get("prior_param")
                    if p is not None:
                        last_prior_param = p

    def run_disjoint(
        self,
        model: Callable[..., AnalysisResult],
        include_gaps: bool = True,
        conversion_type: str = "base",
        conversion_kwargs: dict[str, Any] | None = None,
        n_jobs: int = 1,
        store_all_signals: bool = True,
        decay_half_life: str | pd.Timedelta | float | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the longitudinal model on disjoint time intervals.

        Evaluates the analysis model independently on each time window without
        accumulating historical events. Supports multi-core parallelism.

        Parameters:
            model: Disproportionality analysis callable (e.g. prr, ror, bcpnn, gps).
            include_gaps: Whether to record (timestamp, None) for intervals with no events.
            conversion_type: Container conversion type ('base', 'binary', or 'multi-item').
            conversion_kwargs: Optional dictionary of keyword arguments passed to converter.
            n_jobs: Number of CPU workers for parallel slice execution (default: 1).
            store_all_signals: If False, prunes all_signals to save memory across slices.
            decay_half_life: Optional continuous exponential half-life for historical reports.
                Defaults to None (disabled).
            **kwargs: Additional parameters forwarded to the analysis model callable.
        """
        self.results = []
        effective_decay = decay_half_life if decay_half_life is not None else self.decay_half_life
        hl_days = _parse_half_life_days(effective_decay)
        counts = self.date_groups[self.count_col].sum()
        fast_idx = self._prepare_base_index(conversion_kwargs) if conversion_type == "base" else None

        tasks = []
        prev_ts = None
        for timestamp, count in counts.items():
            if count == 0:
                if include_gaps:
                    tasks.append((timestamp, None, None))
                prev_ts = timestamp
                continue

            idx_start = 0 if prev_ts is None else int(
                np.searchsorted(self._sorted_dates, np.datetime64(prev_ts), side="right")
            )
            idx_end = int(
                np.searchsorted(self._sorted_dates, np.datetime64(timestamp), side="right")
            )
            prev_ts = timestamp

            if idx_start >= idx_end:
                if include_gaps:
                    tasks.append((timestamp, None, None))
                continue

            if fast_idx is not None:
                n_p, n_a = fast_idx["n_p"], fast_idx["n_a"]
                flat_idx = fast_idx["flat_idx"][idx_start:idx_end]
                counts_sub = fast_idx["counts_arr"][idx_start:idx_end]
                p_sub = fast_idx["p_codes"][idx_start:idx_end]
                a_sub = fast_idx["ae_codes"][idx_start:idx_end]

                if hl_days is not None:
                    ts_ns = np.datetime64(timestamp, "ns")
                    age_days = np.maximum(
                        0.0,
                        (ts_ns - self._sorted_dates[idx_start:idx_end]).astype(np.float64) / 86_400_000_000_000.0,
                    )
                    w_counts = counts_sub * np.exp2(-age_days / hl_days)
                else:
                    w_counts = counts_sub

                mat = np.bincount(flat_idx, weights=w_counts, minlength=n_p * n_a).reshape(n_p, n_a)
                if hl_days is None and fast_idx["is_int"]:
                    mat = mat.astype(fast_idx["orig_dtype"], copy=False)

                p_pres = np.bincount(p_sub, minlength=n_p) > 0
                a_pres = np.bincount(a_sub, minlength=n_a) > 0
                sub_container = _build_container_from_matrix(
                    mat,
                    p_pres,
                    a_pres,
                    fast_idx["p_uniques"],
                    fast_idx["ae_uniques"],
                    fast_idx["product_label"],
                    fast_idx["ae_label"],
                    fast_idx["margin_threshold"],
                )
            else:
                subset = self.data.iloc[idx_start:idx_end]
                if hl_days is not None:
                    subset = _apply_time_decay(subset, timestamp, hl_days, self.count_col)
                sub_container = self._convert(subset, conversion_type, conversion_kwargs)

            tasks.append((timestamp, sub_container, dict(kwargs)))

        if n_jobs != 1:
            from joblib import Parallel, delayed

            raw_results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_longitudinal_slice)(
                    model,
                    sub_cont,
                    ts,
                    include_gaps,
                    kw,
                    store_all_signals,
                    conversion_type,
                    conversion_kwargs,
                )
                for ts, sub_cont, kw in tasks
            )
            self.results = [r for r in raw_results if r is not None]
        else:
            for ts, sub_container, kw in tasks:
                if sub_container is None:
                    if include_gaps:
                        self.results.append((ts, None))
                else:
                    self._run_model(model, sub_container, ts, include_gaps, kw, store_all_signals)

    def _run_model(
        self,
        model: Callable[..., AnalysisResult],
        sub_container: DataContainer,
        timestamp: pd.Timestamp,
        include_gaps: bool,
        kwargs: dict[str, Any],
        store_all_signals: bool = True,
    ) -> AnalysisResult | None:
        try:
            da_results = model(sub_container, **kwargs)
            if not store_all_signals and hasattr(da_results, "all_signals"):
                da_results.all_signals = pd.DataFrame()
            self.results.append((timestamp, da_results))
            return da_results
        except ValueError:
            warnings.warn(f"Insufficient data for this model. Skipping time slice: {timestamp}")
            if include_gaps:
                self.results.append((timestamp, None))
            return None

    def regroup_dates(self, time_unit: str) -> None:
        """Regroup the data by a new time unit alias.

        Parameters:
            time_unit: Resample frequency alias (e.g. 'YE', 'QE', 'ME', or legacy 'A', 'Q', 'M').
        """
        self.time_unit = time_unit
        self.date_groups = self.data.resample(_normalize_time_unit(self.time_unit), on="date")

    def to_dataframe(self, which: str = "signals") -> pd.DataFrame:
        """Export longitudinal results into a tidy panel DataFrame.

        Consolidates results across all evaluated time slices into a single DataFrame
        with a leading 'date' column, facilitating time-series plotting, signal progression
        tracking, and longitudinal alert monitoring.

        Parameters:
            which: Which signals table to extract ('signals' for alerted combinations,
                or 'all' / 'all_signals' for the full combinatorial table). Defaults to 'signals'.

        Returns:
            A concatenated DataFrame sorted by date, or an empty DataFrame if no results exist.
        """
        if not self.results:
            return pd.DataFrame()

        which_key = "signals" if which in ("signals", "alert", "alerts") else "all_signals"
        frames = [
            table.assign(date=timestamp)
            for timestamp, res in self.results
            if res is not None and (table := getattr(res, which_key, None)) is not None and not table.empty
        ]

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        cols = ["date"] + [c for c in out.columns if c != "date"]
        return out[cols]

    def summary(self) -> pd.DataFrame:
        """Alias for to_dataframe(which='signals'). Returns all longitudinal alerts."""
        return self.to_dataframe(which="signals")

    def __repr__(self) -> str:
        decay_str = f", decay_half_life='{self.decay_half_life}'" if self.decay_half_life is not None else ""
        num_slices = len(self.results)
        if num_slices == 0:
            return f"<LongitudinalModel(time_unit='{self.time_unit}'{decay_str}, status='unfitted')>"
        valid_slices = [r for _, r in self.results if r is not None]
        total_signals = sum(r.num_signals for r in valid_slices)
        return (
            f"<LongitudinalModel(time_unit='{self.time_unit}'{decay_str}, slices={num_slices}, "
            f"active_slices={len(valid_slices)}, total_alerts={total_signals})>"
        )

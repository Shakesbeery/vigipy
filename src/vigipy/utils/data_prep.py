from itertools import chain, combinations
from collections import Counter, defaultdict
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .Container import DataContainer


def convert(
    data_frame,
    margin_threshold=1,
    product_label="name",
    count_label="count",
    ae_label="AE",
):
    """
    Convert a Pandas dataframe object into a container class for use
    with the disproportionality analyses. Column names in the DataFrame
    must include or be specified in the arguments:
        "name" -- A brand/generic name for the product. This module
                    expects that you have already cleaned the data
                    so there is only one name associated with a class.
        "AE" -- The adverse event(s) associated with a drug/device.
        "count" -- The number of AEs associated with that drug/device
                    and AE. You can input a sheet with single counts
                    (i.e. duplicate rows) or pre-aggregated counts.

    Arguments:
        data_frame (Pandas DataFrame): The Pandas DataFrame object

        margin_threshold (int): The threshold for counts. Lower numbers will
                             be removed from consideration

    Returns:
        RES (DataStorage object): A container object that holds the necessary
                                    components for DA.

    """
    data_cont = compute_contingency(data_frame, product_label, count_label, ae_label, margin_threshold)
    col_sums = np.sum(data_cont, axis=0)
    row_sums = np.sum(data_cont, axis=1)

    # Compute the flattened table from the contingency table.
    data_df = count(data_cont, row_sums, col_sums)

    # Initialize the container object and assign the data
    return DataContainer(
        data=data_df,
        N=data_df["events"].sum(),
        contingency=data_cont,
        type="contingency",
    )


def compute_contingency(data_frame, product_label, count_label, ae_label, margin_threshold):
    """Compute the contingency table for DA

    Args:
        data_frame (pd.DataFrame): A count data dataframe of the drug/device and events data
        product_label (str): Label of the column containing the product names
        count_label (str): Label of the column containing the event counts
        ae_label (str): Label of the column containing the adverse event counts
        margin_threshold (int): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        pd.DataFrame: A contingency table with adverse events as columns and products as rows.
    """
    # Create a contingency table based on the brands and AEs
    data_cont = pd.pivot_table(
        data_frame,
        values=count_label,
        index=product_label,
        columns=ae_label,
        aggfunc="sum",
        fill_value=0,
    )

    # Calculate empty rows/columns based on margin_threshold and filter efficiently
    r_sums = np.sum(data_cont.values, axis=1)
    c_sums = np.sum(data_cont.values, axis=0)
    data_cont = data_cont.iloc[r_sums >= margin_threshold, c_sums >= margin_threshold]
    return data_cont


def convert_binary(
    data,
    product_label="name",
    ae_label="AE",
    use_counts=False,
    count_label="count",
    expand_counts=True,
    report_id_label=None,
    sparse=False,
    covariate_labels=None,
):
    """Convert input data consisting of unique product-event pairs into a
       binary dataframe indicating which event and which product are
       associated with each other.

    Args:
        data (pd.DataFrame): A DataFrame consisting of unique product-event pairs for each row
        product_label (str, optional): If the product name is not in a column called `name`, override here. Defaults to "name".
        ae_label (str, optional): If the adverse event is not in a column called `AE`, override here.. Defaults to "AE".
        use_counts (bool, optional): Whether to use aggregated count representation. Defaults to False.
        count_label (str, optional): Column name containing counts. Defaults to "count".
        expand_counts (bool, optional): Whether to expand counts > 1 into duplicate rows. Defaults to True.
        report_id_label (str, optional): Column name for report/patient IDs. When specified, groups co-reported
            products and adverse events at the individual report level. Defaults to None.
        sparse (bool, optional): If True, construct sparse CSR-backed DataFrames to reduce memory
            for large datasets. Defaults to False.
        covariate_labels (list[str], optional): Column names for per-report covariates (e.g. ['age', 'sex']).
            Requires report_id_label. Continuous covariates are standardized, categorical are
            one-hot encoded with drop_first=True. Defaults to None.

    Returns:
        Container: A container with two binary dataframes. One is the X data of product names and the other is the
        y data with adverse events. Index locations are associated with the input DataFrame.

    """
    if covariate_labels is not None and report_id_label is None:
        raise ValueError(
            "covariate_labels requires report_id_label to identify per-report covariates."
        )

    if report_id_label is not None and report_id_label in data.columns:
        keep_labels = [product_label, ae_label, count_label, report_id_label]
        if covariate_labels:
            keep_labels.extend(covariate_labels)
        data_clean = _sanitize_data(data, keep_labels)

        if sparse:
            prod_df = _build_sparse_crosstab(data_clean, report_id_label, product_label)
            event_df = _build_sparse_crosstab(data_clean, report_id_label, ae_label)
        else:
            prod_df = pd.crosstab(data_clean[report_id_label], data_clean[product_label]).clip(upper=1)
            event_df = pd.crosstab(data_clean[report_id_label], data_clean[ae_label]).clip(upper=1)

        common_idx = prod_df.index.intersection(event_df.index)
        prod_df = prod_df.loc[common_idx]
        event_df = event_df.loc[common_idx]

        # Extract and encode covariates
        covariates = None
        covariate_names = None
        if covariate_labels:
            covariates, covariate_names = _extract_covariates(
                data_clean, report_id_label, covariate_labels, common_idx
            )

        return DataContainer(
            data=data_clean,
            N=len(common_idx),
            product_features=prod_df,
            event_outcomes=event_df,
            type="binary_report",
            covariates=covariates,
            feature_names=list(prod_df.columns),
            event_names=list(event_df.columns),
            covariate_names=covariate_names,
        )

    # Sanitize df to remove unnecessary information during transforms
    data = _sanitize_data(data, [product_label, ae_label, count_label])

    if use_counts:
        if not isinstance(product_label, str):
            group_list = [*product_label, ae_label]
        else:
            group_list = [product_label, ae_label]
        data = data.groupby(group_list).sum().reset_index()
        event_df = __transform_dataframe(data, count_label, ae_label)
        dc_type = "binary_count"
    else:
        if data[count_label].max() > 1 and expand_counts:
            data = __expand_dataframe(data, count_label, ae_label, product_label)
        if sparse:
            event_df = _build_sparse_dummies(data[ae_label])
        else:
            event_df = pd.get_dummies(data[ae_label], prefix="", prefix_sep="")
            event_df = event_df.T.groupby(level=0).sum().T
        dc_type = "binary"

    if sparse:
        prod_df = _build_sparse_dummies(data[product_label])
    else:
        prod_df = pd.get_dummies(data[product_label], prefix="", prefix_sep="")
        prod_df = prod_df.T.groupby(level=0).sum().T

    return DataContainer(
        data=data,
        N=data.shape[0],
        product_features=prod_df,
        event_outcomes=event_df,
        type=dc_type,
        feature_names=list(prod_df.columns),
        event_names=list(event_df.columns),
    )


def convert_multi_item(df, product_label=None, ae_label="AE", count_label="count", min_threshold=3):
    """***WARNING*** Currently experimental and not guaranteed to perform as expected.
    Convert data with multiple product columns into a multi-item flattened dataframe for the DA methods.

    Args:
        df (pd.DataFrame): A dataframe where each row is a unique adverse event and has multiple columns
        indicating the presence of multiple devices/drugs/interventions.
        product_label (list, optional): A list of column names associated with the co-occuring products. Defaults to ["name"].
        ae_label (str, optional): The column name that contains the adverse events. Defaults to "AE".
        min_threshold (int, optional): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        Container: A container object that holds the necessary components for DA.
    """
    if product_label is None:
        product_label = ["name"]

    ae_counts = defaultdict(int)
    product_counts = defaultdict(int)
    for col in product_label:
        for ae, product, count in df.loc[df[col] != ""][[ae_label, col, count_label]].itertuples(index=False):
            ae_counts[ae] += count
            product_counts[product] += count

    # Initialize an empty list to store the result
    result = []

    # Sanitize df to remove unnecessary information during transforms
    df = _sanitize_data(df, [product_label, ae_label, count_label])

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Extract product names from the current row
        names = {row[x] for x in product_label if row[x]}
        # Get all unique combinations of names (without repetition)
        combos = list(chain.from_iterable(combinations(names, r) for r in range(1, len(names) + 1)))
        # Append combinations to the result list with the other column info
        for combo in combos:
            new_data = {idx: row[idx] for idx in row.index if idx not in product_label}
            new_data["product_name"] = f"{'|'.join([c for c in combo if c])}"
            new_data["product_aes"] = sum([product_counts[p] for p in combo])
            new_data["count_across_brands"] = ae_counts[row[ae_label]]
            result.append(new_data)

    # Convert the result list to a new dataframe
    new_df = pd.DataFrame(result)
    event_series = new_df.groupby(by=["AE", "product_name"]).sum()["count"]
    new_df["events"] = new_df.apply(lambda x: event_series[x["AE"]][x["product_name"]], axis=1)
    new_df.rename(columns={ae_label: "ae_name"}, inplace=True)

    return DataContainer(
        data=new_df[["ae_name", "product_name", "count_across_brands", "product_aes", "events"]].drop_duplicates(),
        N=new_df["events"].sum(),
        contingency=compute_contingency(new_df, "product_name", "count", "ae_name", min_threshold),
    )


def convert_ddi(
    data: pd.DataFrame,
    product_label: str = "name",
    ae_label: str = "AE",
    report_id_label: str = "report_id",
    min_co_reports: int = 3,
    include_singles: bool = True,
    target_drugs: Optional[Union[list[str], set[str]]] = None,
    pair_delimiter: str = " + ",
    sparse: bool = True,
    covariate_labels: Optional[list[str]] = None,
) -> DataContainer:
    """Convert case-level report data into a Drug-Drug Interaction (DDI) DataContainer.

    Identifies co-prescribed drug pairs meeting a co-reporting threshold and generates
    interaction columns alongside single-drug features. Produces a full DataContainer
    compatible with all vigipy disproportionality methods (PRR, ROR, GPS, LASSO, SCORE-DA,
    and SCORE-DDI).

    Parameters:
        data: A DataFrame consisting of report-level product and adverse event occurrences.
        product_label: Column containing product/drug names. Defaults to "name".
        ae_label: Column containing adverse event terms. Defaults to "AE".
        report_id_label: Column containing individual report / patient identifiers. Required
            to determine true patient-level co-exposure. Defaults to "report_id".
        min_co_reports: Minimum number of distinct reports where Drug A and Drug B must
            appear together to be included as a candidate interaction pair. Defaults to 3.
        include_singles: Whether to include single-drug main effect features alongside the
            interaction pairs. True is strongly recommended so models can condition on
            individual drug baselines. Defaults to True.
        target_drugs: Optional list or set of drug names to focus screening. If specified,
            only pairs containing at least one target drug are retained. Defaults to None.
        pair_delimiter: String delimiter used to join drug names in pairs. Defaults to " + ".
        sparse: If True, uses memory-efficient scipy.sparse CSR representation. Defaults to True.
        covariate_labels: Optional list of per-report covariate column names (e.g. ['age', 'sex']).

    Returns:
        DataContainer: Fully initialized container with product_features, event_outcomes,
            contingency, data, and pair_mapping.
    """
    if report_id_label not in data.columns:
        raise ValueError(
            f"report_id_label '{report_id_label}' not found in data columns. "
            "DDI conversion requires a report identifier to model patient-level co-prescription."
        )
    if product_label not in data.columns:
        raise ValueError(f"product_label '{product_label}' not found in data columns.")
    if ae_label not in data.columns:
        raise ValueError(f"ae_label '{ae_label}' not found in data columns.")
    if covariate_labels is not None and not all(c in data.columns for c in covariate_labels):
        missing = [c for c in covariate_labels if c not in data.columns]
        raise ValueError(f"covariate_labels {missing} not found in data columns.")

    keep_cols = [report_id_label, product_label, ae_label]
    if covariate_labels:
        keep_cols.extend(covariate_labels)
    data_clean = _sanitize_data(data, keep_cols).dropna(subset=[report_id_label, product_label, ae_label])
    data_clean[product_label] = data_clean[product_label].astype(str).str.strip()
    data_clean[ae_label] = data_clean[ae_label].astype(str).str.strip()

    prod_df = _build_sparse_crosstab(data_clean, report_id_label, product_label)
    event_df = _build_sparse_crosstab(data_clean, report_id_label, ae_label)
    common_idx = prod_df.index.intersection(event_df.index)
    prod_df = prod_df.loc[common_idx]
    event_df = event_df.loc[common_idx]

    X_single_csr = prod_df.sparse.to_coo().tocsr()
    single_drugs = list(prod_df.columns)

    # Co-occurrence across reports: M_co = X.T @ X
    M_co = (X_single_csr.T @ X_single_csr).tocoo()
    target_set = set(target_drugs) if target_drugs is not None else None

    pair_list = []
    for j, k, count_val in zip(M_co.row, M_co.col, M_co.data):
        if j < k and count_val >= min_co_reports:
            d1, d2 = single_drugs[j], single_drugs[k]
            if target_set is not None and (d1 not in target_set and d2 not in target_set):
                continue
            pair_list.append((j, k, d1, d2, int(count_val)))

    if not pair_list:
        msg = f"No drug pairs met the threshold of min_co_reports={min_co_reports}."
        if target_drugs:
            msg += f" (target_drugs={target_drugs})"
        raise ValueError(msg)

    # Sort pairs by co-occurrence count descending
    pair_list.sort(key=lambda x: x[4], reverse=True)

    pair_names = []
    pair_cols = []
    pair_mapping = {}

    for j, k, d1, d2, _ in pair_list:
        pname = f"{d1}{pair_delimiter}{d2}"
        pair_names.append(pname)
        pair_mapping[pname] = (d1, d2)
        # Sparse column elementwise product
        inter_col = X_single_csr[:, j].multiply(X_single_csr[:, k])
        pair_cols.append(inter_col)

    X_pairs_csr = sp.hstack(pair_cols, format="csr")

    if include_singles:
        X_all_csr = sp.hstack([X_single_csr, X_pairs_csr], format="csr")
        all_feature_names = single_drugs + pair_names
    else:
        X_all_csr = X_pairs_csr
        all_feature_names = pair_names

    if sparse:
        full_prod_df = pd.DataFrame.sparse.from_spmatrix(
            X_all_csr, index=common_idx, columns=all_feature_names
        ).astype(pd.SparseDtype(float, 0.0))
        final_event_df = event_df
    else:
        full_prod_df = pd.DataFrame(
            X_all_csr.toarray(), index=common_idx, columns=all_feature_names
        )
        final_event_df = pd.DataFrame(
            event_df.sparse.to_coo().toarray(),
            index=common_idx,
            columns=list(event_df.columns),
        )

    Y_csr = event_df.sparse.to_coo().tocsr()
    C_mat = (X_all_csr.T @ Y_csr).toarray()
    contingency_df = pd.DataFrame(
        C_mat, index=all_feature_names, columns=list(event_df.columns)
    )
    r_sums = np.sum(C_mat, axis=1)
    c_sums = np.sum(C_mat, axis=0)
    data_df = count(contingency_df, r_sums, c_sums)

    covariates = None
    covariate_names = None
    if covariate_labels:
        covariates, covariate_names = _extract_covariates(
            data_clean, report_id_label, covariate_labels, common_idx
        )

    return DataContainer(
        data=data_df,
        N=int(np.sum(C_mat)),
        contingency=contingency_df,
        product_features=full_prod_df,
        event_outcomes=final_event_df,
        type="binary_ddi",
        covariates=covariates,
        feature_names=all_feature_names,
        event_names=list(event_df.columns),
        covariate_names=covariate_names,
        pair_mapping=pair_mapping,
    )


def count(data, rows, cols):
    """
    Convert the input contingency table to a flattened table

    Arguments:
        data (Pandas DataFrame): A contingency table of brands and events

    Returns:
        df: A Pandas DataFrame with the count information

    """
    mat = data.values
    c_idx, r_idx = np.nonzero(mat.T)
    rows_arr = np.asarray(rows)
    cols_arr = np.asarray(cols)
    return pd.DataFrame({
        "events": mat[r_idx, c_idx],
        "product_aes": rows_arr[r_idx],
        "count_across_brands": cols_arr[c_idx],
        "ae_name": np.asarray(data.columns)[c_idx],
        "product_name": np.asarray(data.index)[r_idx],
    })[["events", "product_aes", "count_across_brands", "ae_name", "product_name"]]

def _sanitize_data(df, keep_labels):
    keep = []
    for label in keep_labels:
        if not isinstance(label, str):
            keep.extend(label)
        else:
            keep.append(label)

    return df[keep].copy()


def _build_sparse_crosstab(data, index_label, column_label):
    """Build a binary sparse DataFrame via categorical indexing.

    Constructs a CSR matrix mapping (index_label × column_label) co-occurrences,
    clipped to binary, and wraps it in a pandas SparseDtype DataFrame.
    """
    idx_cat = pd.Categorical(data[index_label])
    col_cat = pd.Categorical(data[column_label])
    row_codes = idx_cat.codes
    col_codes = col_cat.codes
    ones = np.ones(len(row_codes), dtype=np.float64)
    csr = sp.coo_matrix(
        (ones, (row_codes, col_codes)),
        shape=(len(idx_cat.categories), len(col_cat.categories)),
    ).tocsr()
    # Clip to binary (co-occurrence → 0/1)
    csr.data = np.minimum(csr.data, 1.0)
    csr.eliminate_zeros()
    result = pd.DataFrame.sparse.from_spmatrix(
        csr, index=idx_cat.categories, columns=col_cat.categories
    ).astype(pd.SparseDtype(float, 0.0))
    return result


def _build_sparse_dummies(series):
    """Build a sparse one-hot DataFrame from a categorical series.

    Equivalent to pd.get_dummies but returns SparseDtype-backed DataFrame,
    avoiding the deprecated pd.get_dummies(sparse=True).
    """
    cat = pd.Categorical(series)
    n = len(cat)
    row_idx = np.arange(n)
    col_idx = cat.codes
    # Filter out -1 codes (NaN values)
    mask = col_idx >= 0
    ones = np.ones(mask.sum(), dtype=np.float64)
    csr = sp.coo_matrix(
        (ones, (row_idx[mask], col_idx[mask])),
        shape=(n, len(cat.categories)),
    ).tocsr()
    result = pd.DataFrame.sparse.from_spmatrix(
        csr, columns=cat.categories
    ).astype(pd.SparseDtype(float, 0.0))
    return result


def _extract_covariates(data, report_id_label, covariate_labels, common_idx):
    """Extract, aggregate, and encode per-report covariates.

    Continuous covariates are standardized (mean=0, std=1) with a zero-variance guard.
    Categorical covariates are one-hot encoded with drop_first=True.

    Returns:
        (covariates_df, covariate_names): DataFrame aligned to common_idx and list of column names.
    """
    # Aggregate covariates to one row per report (take first value per report)
    cov_cols = [report_id_label] + list(covariate_labels)
    cov_data = data[cov_cols].drop_duplicates(subset=[report_id_label]).set_index(report_id_label)
    cov_data = cov_data.loc[common_idx]

    encoded_parts = []
    for col in covariate_labels:
        series = cov_data[col]
        if pd.api.types.is_numeric_dtype(series):
            # Continuous: standardize with zero-variance guard
            mu = series.mean()
            sigma = series.std()
            if sigma == 0 or pd.isna(sigma):
                sigma = 1.0
            standardized = (series - mu) / sigma
            encoded_parts.append(standardized.to_frame(name=col))
        else:
            # Categorical: one-hot with drop_first
            dummies = pd.get_dummies(series, prefix=col, prefix_sep="_", drop_first=True)
            encoded_parts.append(dummies)

    covariates_df = pd.concat(encoded_parts, axis=1).astype(np.float64)
    covariate_names = list(covariates_df.columns)
    return covariates_df, covariate_names


def __expand_dataframe(df, count_label, ae_label, product_label):
    new = defaultdict(list)
    for row in df.itertuples(index=False):
        for _ in range(int(getattr(row, count_label))):
            new[product_label].append(getattr(row, product_label))
            new[ae_label].append(getattr(row, ae_label))
            new[count_label].append(1)

    new_data = pd.DataFrame(new)
    return new_data


def __transform_dataframe(df, count_label, ae_label):
    # Create a new dataframe with unique values from 'AE' as columns, and initialize all cells with 0
    new_df = pd.DataFrame(0, index=range(len(df)), columns=df[ae_label].unique())

    # Iterate through the rows and set the appropriate value from 'count' in the corresponding 'AE' column
    for i, row in df.iterrows():
        new_df.at[i, row[ae_label]] = row[count_label]

    return new_df

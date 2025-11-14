from itertools import product, chain, combinations
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .Container import Container


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
    DC = Container()
    DC.contingency = data_cont
    DC.data = data_df
    DC.N = data_df["events"].sum()
    DC.type = "contingency"
    return DC


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

    # Calculate empty rows/columns based on margin_threshold and remove
    cut_rows = np.where(np.sum(data_cont, axis=1) < margin_threshold)
    drop_rows = data_cont.index[cut_rows]

    cut_cols = np.where(np.sum(data_cont, axis=0) < margin_threshold)
    drop_cols = data_cont.columns[cut_cols]

    data_cont = data_cont.drop(drop_rows)
    data_cont = data_cont.drop(drop_cols, axis=1)
    return data_cont


def convert_binary(
    data, product_label="name", ae_label="AE", use_counts=False, count_label="count", expand_counts=True
):
    """Convert input data consisting of unique product-event pairs into a
       binary dataframe indicating which event and which product are
       associated with each other.

    Args:
        data (pd.DataFrame): A DataFrame consisting of unique product-event pairs for each row
        product_label (str, optional): If the product name is not in a column called `name`, override here. Defaults to "name".
        ae_label (str, optional): If the adverse event is not in a column called `AE`, override here.. Defaults to "AE".

    Returns:
        Container: A container with two binary dataframes. One is the X data of product names and the other is the
        y data with adverse events. Index locations are associated with the input DataFrame.

    """
    DC = Container()

    # Sanitize df to remove unnecessary information during transforms
    data = _sanitize_data(data, [product_label, ae_label, count_label])

    if use_counts:
        if not isinstance(product_label, str):
            group_list = [*product_label, ae_label]
        else:
            group_list = [product_label, ae_label]
        data = data.groupby(group_list).sum().reset_index()
        event_df = __transform_dataframe(data, count_label, ae_label)
        DC.type = "binary_count"
    else:
        if data[count_label].max() > 1 and expand_counts:
            data = __expand_dataframe(data, count_label, ae_label, product_label)
        event_df = pd.get_dummies(data[ae_label], prefix="", prefix_sep="")
        event_df = event_df.groupby(by=event_df.columns, axis=1).sum()
        DC.type = "binary"

    prod_df = pd.get_dummies(data[product_label], prefix="", prefix_sep="")
    DC.product_features = prod_df.groupby(by=prod_df.columns, axis=1).sum()

    DC.event_outcomes = event_df
    DC.N = data.shape[0]
    DC.data = data

    return DC


def convert_multi_item(df, product_label=["name"], ae_label="AE", count_label="count", min_threshold=3):
    """***WARNING*** Currently experimental and not guaranteed to perform as expected.
    Convert data with multiple product columns into a multi-item flattened dataframe for the DA methods.

    Args:
        df (pd.DataFrame): A dataframe where each row is a unique adverse event and has multiple columns
        indicating the presence of multiple devices/drugs/interventions.
        product_cols (list, optional): A list of column names associated with the co-occuring products. Defaults to ["name"].
        ae_col (str, optional): The column name that contains the adverse events. Defaults to "AE".
        min_threshold (int, optional): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        Container: A container object that holds the necessary components for DA.
    """
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
    new_df.rename(columns={ae_col: "ae_name"}, inplace=True)

    DC = Container()
    DC.contingency = compute_contingency(new_df, "product_name", "count", "ae_name", min_threshold)
    DC.data = new_df[["ae_name", "product_name", "count_across_brands", "product_aes", "events"]].drop_duplicates()
    DC.N = new_df["events"].sum()

    return DC


def count(data, rows, cols):
    """
    Convert the input contingency table to a flattened table

    Arguments:
        data (Pandas DataFrame): A contingency table of brands and events

    Returns:
        df: A Pandas DataFrame with the count information

    """
    d = {
        "events": [],
        "product_aes": [],
        "count_across_brands": [],
        "ae_name": [],
        "product_name": [],
    }
    for col, row in product(data.columns, data.index):
        n11 = data[col][row]
        if n11 > 0:
            d["count_across_brands"].append(cols[col])
            d["product_aes"].append(rows[row])
            d["events"].append(n11)
            d["product_name"].append(row)
            d["ae_name"].append(col)

    df = pd.DataFrame(d)
    return df

def _sanitize_data(df, keep_labels):
    keep = []
    for label in keep_labels:
        if not isinstance(label, str):
            keep.extend(label)
        else:
            keep.append(label)

    return df[keep].copy()


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

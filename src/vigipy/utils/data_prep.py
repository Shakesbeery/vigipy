import itertools

import numpy as np
import pandas as pd

from .Container import Container


def convert(data_frame, margin_threshold=1, product_label="name", count_label="count", ae_label="AE"):
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
    # Create a contingency table based on the brands and AEs
    data_cont = pd.pivot_table(
        data_frame, values=count_label, index=product_label, columns=ae_label, aggfunc=np.sum, fill_value=0
    )

    # Calculate empty rows/columns based on margin_threshold and remove
    cut_rows = np.where(np.sum(data_cont, axis=1) < margin_threshold)
    drop_rows = data_cont.index[cut_rows]

    cut_cols = np.where(np.sum(data_cont, axis=0) < margin_threshold)
    drop_cols = data_cont.columns[cut_cols]

    data_cont = data_cont.drop(drop_rows)
    data_cont = data_cont.drop(drop_cols, axis=1)
    col_sums = np.sum(data_cont, axis=0)
    row_sums = np.sum(data_cont, axis=1)

    # Compute the flattened table from the contingency table.
    data_df = count(data_cont, row_sums, col_sums)

    # Initialize the container object and assign the data
    DC = Container()
    DC.contingency = data_cont
    DC.data = data_df
    DC.N = data_df["events"].sum()
    return DC


def count(data, rows, cols):
    """
    Convert the input contingency table to a flattened table

    Arguments:
        data (Pandas DataFrame): A contingency table of brands and events

    Returns:
        df: A Pandas DataFrame with the count information

    """
    d = {"events": [], "product_aes": [], "count_across_brands": [], "ae_name": [], "product_name": []}
    for col, row in itertools.product(data.columns, data.index):
        n11 = data[col][row]
        if n11 > 0:
            d["count_across_brands"].append(cols[col])
            d["product_aes"].append(rows[row])
            d["events"].append(n11)
            d["product_name"].append(row)
            d["ae_name"].append(col)

    df = pd.DataFrame(d)
    return df

import pandas as pd
import traceback
from ..utils import convert, convert_binary, convert_multi_item


class LongitudinalModel:

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

    def __init__(self, dataframe, time_unit):
        """
        Initialize the longitudinal model with raw data and a time unit.

        Arguments:
            dataframe (Pandas DataFrame): A dataframe containing counts, AEs,
                                          product/brands and AE dates.

            time_unit (str): One of Pandas' time unit aliases found here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        """
        self.time_unit = time_unit
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        self.date_groups = dataframe.resample(time_unit, on="date")
        self.data = dataframe

    def _convert(self, data, conversion_type, conversion_kwargs):
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


    def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
        """
        Run the longitudinal model as initialized.

        Arguments:
            model (vigipy model): One of the supported vigipy models
                                  from this module (i.e. gps, prr, etc)

            include_gaps (bool): If a particular time slice has a
                                 zero-count sum, still run the model?

            kwargs: key word arguments can be added to this function call
                    and they will be passed into the model at run time.

        """
        self.results = []
        for timestamp, count in self.date_groups.sum()["count"].items():
            if count == 0:
                if include_gaps:
                    self.results.append((timestamp, None))
                continue

            subset = self.data.loc[self.data["date"] <= timestamp]
            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            self._run_model(model, sub_container, timestamp, include_gaps, kwargs)

    def run_disjoint(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
        """
        Run the longitudinal model as initialized.

        Arguments:
            model (vigipy model): One of the supported vigipy models
                                  from this module (i.e. gps, prr, etc)

            include_gaps (bool): If a particular time slice has a
                                 zero-count sum, still run the model?

            kwargs: key word arguments can be added to this function call
                    and they will be passed into the model at run time.

        """
        self.results = []
        for count, (timestamp, subset) in zip(
            self.date_groups.sum()["count"], self.date_groups
        ):
            if count == 0:
                if include_gaps:
                    self.results.append((timestamp, None))
                continue

            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            self._run_model(model, sub_container, timestamp, include_gaps, kwargs)

    def _run_model(self, model, sub_container, timestamp, include_gaps, kwargs):
        try:
            da_results = model(sub_container, **kwargs)
            self.results.append((timestamp, da_results))
        except ValueError as e:
            print(traceback.format_exc())
            if include_gaps:
                self.results.append((timestamp, None))
            print(f"Insufficient data for this model. Skipping this slice: {timestamp}")

    def regroup_dates(self, time_unit):
        """
        Regroup the data by a new time unit.

        Arguments:
            time_unit (str): One of Pandas' time unit aliases. (Q, A, QS, etc.)

        """
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit, on="date")

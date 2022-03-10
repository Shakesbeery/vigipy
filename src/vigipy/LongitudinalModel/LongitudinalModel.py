import pandas as pd
from ..utils import convert


class LongitudinalModel():

    def __init__(self, dataframe, time_unit):
        '''
        Initialize the longitudinal model with raw data and a time unit.

        Arguments:
            dataframe (Pandas DataFrame): A dataframe containing counts, AEs,
                                          product/brands and AE dates.

            time_unit (str): One of Pandas' time unit aliases. (Q, A, QS, etc.)

        '''
        self.time_unit = time_unit
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        self.date_groups = dataframe.resample(time_unit,
                                              on='date').sum()['count']
        self.data = dataframe

    def run(self, model, include_gaps=True, **kwargs):
        '''
        Run the longitudinal model as initialized.

        Arguments:
            model (vigipy model): One of the supported vigipy models
                                  from this module (i.e. gps, prr, etc)

            include_gaps (bool): If a particular time slice has a
                                 zero-count sum, still run the model?

            kwargs: key word arguments can be added to this function call
                    and they will be passed into the model at run time.

        '''
        self.results = []
        for timestamp, count in self.date_groups.iteritems():
            if count == 0 and not include_gaps:
                continue
            subset = self.data.loc[self.data['date'] <= timestamp]
            sub_container = convert(subset)
            try:
                da_results = model(sub_container, **kwargs)
                self.results.append((timestamp, da_results))
            except ValueError:
                print("Insufficient data for this model. Skipping this slice.")

    def regroup_dates(self, time_unit):
        '''
        Regroup the data by a new time unit.

        Arguments:
            time_unit (str): One of Pandas' time unit aliases. (Q, A, QS, etc.)

        '''
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit,
                                              on='date').sum()['count']

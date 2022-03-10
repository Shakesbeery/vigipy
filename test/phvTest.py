import unittest

import pandas as pd

from src.vigipy import bcpnn, gps, ror, rfet, prr, convert, LongitudinalModel

df = pd.read_csv("test/fixtures/sample.csv")

data = None

METRICS = ("fdr", "signals", "rank")
STATS = ("p_value", "quantile")
METHODS = ("mantel-haentzel", "negative-binomial", "poisson")


class StateOneTest(unittest.TestCase):
    def test0_Convert(self):
        global df
        global data

        data = convert(df)

    def test1_Bcpnn(self):
        global data
        print("Starting BCPNN testing...")
        for method in METHODS:
            for metric in METRICS:
                for stat in STATS:
                    print(method, metric, stat)
                    bcpnn(data, expected_method=method, decision_metric=metric, ranking_statistic=stat, min_events=3)
                    print("OK!")
        print("Finished with BCPNN testing...")

    def test2_Gps(self):
        global data

        print("Starting GPS testing...")
        for method in METHODS:
            for metric in METRICS:
                for stat in STATS:
                    print(method, metric, stat)
                    gps(data, expected_method=method, decision_metric=metric, ranking_statistic=stat, min_events=3, truncate=True)
                    print("OK!")
        print("Finished with GPS testing...")

    def test3_Ror(self):
        global data

        print("Starting ROR testing...")
        for method in METHODS:
            for metric in METRICS:
                for stat in STATS:
                    print(method, metric, stat)
                    try:
                        ror(data, expected_method=method, decision_metric=metric, ranking_statistic=stat, min_events=3)
                    except Exception as error:
                        print(error)
                    print("OK!")
        print("Finished with ROR testing...")

    def test4_Rfet(self):
        global data

        print("Starting RFET testing...")
        for method in METHODS:
            for metric in METRICS:
                for stat in STATS:
                    print(method, metric, stat)
                    rfet(data, expected_method=method, decision_metric=metric, min_events=3)
                    print("OK!")
        print("Finished with RFET testing...")

    def test5_Prr(self):
        global data

        print("Starting PRR testing...")
        for method in METHODS:
            for metric in METRICS:
                for stat in STATS:
                    print(method, metric, stat)
                    prr(data, expected_method=method, decision_metric=metric, ranking_statistic=stat, min_events=3)
                    print("OK!")
        print("Finished with PRR testing...")

    def test6_LongModel(self):
        global df

        LM = LongitudinalModel(df, "A")

        print("Starting longitudinal model testing...")
        LM.run(gps, False, decision_metric='rank', ranking_statistic='quantile')
        LM.run(bcpnn, False, decision_metric='signals', ranking_statistic='quantile')
        LM.run(prr, False, min_events=1, decision_metric='signals', ranking_statistic='p_value')
        print("Finished with longitudinal model testing...")


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(StateOneTest())

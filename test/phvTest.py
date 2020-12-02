import unittest

import pandas as pd

from src.vigipy import bcpnn, gps, ror, rfet, prr, convert, LongitudinalModel

df = pd.read_csv("test/fixtures/sample.csv")

data = None


class StateOneTest(unittest.TestCase):

    def test0_Convert(self):
        global df
        global data

        data = convert(df)

    def test1_Bcpnn(self):
        global data

        bcpnn(data, min_events=3)
        bcpnn(data, expected_method='negative-binomial', method_alpha=1.1439, min_events=3)

    def test2_Gps(self):
        global data

        gps(data, min_events=3)
        gps(data, expected_method='negative-binomial', method_alpha=1.1439, min_events=3)

    def test3_Ror(self):
        global data

        ror(data, min_events=3)
        ror(data, expected_method='negative-binomial', method_alpha=1.1439, min_events=3)

    def test4_Rfet(self):
        global data

        rfet(data, min_events=3)
        rfet(data, expected_method='negative-binomial', method_alpha=1.1439, min_events=3)

    def test5_Prr(self):
        global data

        prr(data, min_events=3)
        prr(data, expected_method='negative-binomial', method_alpha=1.1439, min_events=3)

    def test6_LongModel(self):
        global df

        LM = LongitudinalModel(df, 'A')
        LM.run(gps, False, decision_metric='rank', ranking_statistic='quantile')


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(StateOneTest())

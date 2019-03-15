import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils.lbe import lbe
from ..utils.Container import Container


def _compute_expected_counts(N, n1j, ni1):
    return n1j * ni1 / N


def ror(container, relative_risk=1, min_events=1, decision_metric='fdr',
        decision_thres=0.05, ranking_statistic='p_value'):
    '''
    Calculate the proportional reporting ratio.

    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        relative_risk (int/float): The relative risk value

        min_events: The min number of AE reports to be considered a signal

        decision_metric (str): The metric used for detecting signals:
                            {fdr = false detection rate,
                            signals = number of signals,
                            rank = ranking statistic}

        decision_thres (float): The min thres value for the decision_metric

        ranking_statistic (str): How to rank signals:
                            {'p_value' = posterior prob of the null hypothesis,
                            'CI' = 95% CI lower boundary}

    '''
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA.loc[DATA.events >= min_events]

    n11 = np.asarray(DATA['events'], dtype=np.float64)
    n1j = np.asarray(DATA['product_aes'], dtype=np.float64)
    ni1 = np.asarray(DATA['count_across_brands'], dtype=np.float64)
    num_cell = len(n11)
    expected = _compute_expected_counts(N, n1j, ni1)

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    log_ror = np.log(n11*n00 / (n10*n01))
    var_log_ror = 1./n11 + 1./n10 + 1./n01 + 1./n00
    pval_uni = 1-norm.cdf(log_ror, np.log(relative_risk), np.sqrt(var_log_ror))
    # rankstat = (log_prr - np.log(relative_risk)) / np.sqrt(var_log_prr)
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0

    results = lbe(2*np.minimum(pval_uni, 1-pval_uni))
    pi_c = results[1]

    fdr = (pi_c * np.sort(pval_uni[pval_uni <= .5]) /
           (np.arange(1, (pval_uni <= .5).sum()+1) / num_cell))

    fdr = np.concatenate((fdr, (pi_c /
                         (2 * np.arange(((pval_uni <= .5).sum()),
                                        num_cell) / num_cell)
                        + 1
                        - (pval_uni <= .5).sum() /
                        np.arange((pval_uni <= .5).sum(), num_cell))),
                        axis=None)

    FDR = np.minimum(fdr, np.ones((len(fdr),)))
    if ranking_statistic == 'CI':
        FDR = np.empty((len(n11),))

    LB = norm.ppf(.025, log_ror, np.sqrt(var_log_ror))
    if ranking_statistic == 'p_value':
        RankStat = pval_uni
    else:
        RankStat = LB

    if (decision_metric == 'fdr' and ranking_statistic == 'p_value'):
        num_signals = (FDR <= decision_thres).sum()
    elif (decision_metric == 'signals'):
        num_signals = min(decision_thres, num_cell)
    elif (decision_metric == 'rank'):
        if (ranking_statistic == 'p_value'):
            num_signals = (RankStat <= decision_thres).sum()
        else:
            num_signals = (RankStat >= decision_thres).sum()

    RC = Container()
    RC.all_signals = pd.DataFrame({'Product': DATA['product_name'],
                                   'Adverse Event': DATA['ae_name'],
                                   'Count': n11,
                                   'Expected Count': expected,
                                   'p_value': RankStat,
                                   'PRR': np.exp(log_ror),
                                   'product margin': n1j,
                                   'event margin': ni1,
                                   'FDR': FDR}).sort_values(by=['p_value'])

    if ranking_statistic == 'CI':
        RC.all_signals.rename(index=str,
                              columns={'p_value': 'lower_bound_CI(95%)'},
                              inplace=True).sort()

    RC.signals = RC.all_signals.loc[0:num_signals, ]
    RC.num_signals = num_signals
    return RC

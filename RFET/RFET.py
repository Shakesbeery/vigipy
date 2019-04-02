import numpy as np
import pandas as pd
from ..utils.lbe import lbe
from scipy.stats import fisher_exact, hypergeom
from ..utils import Container
from ..utils import calculate_expected


def rfet(container, relative_risk=1, min_events=1, decision_metric='fdr',
         decision_thres=0.05, mid_pval=False,
         expected_method='mantel-haentzel', method_alpha=1):
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

    '''
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA.loc[DATA.events >= min_events]

    n11 = np.asarray(DATA['events'], dtype=np.float64)
    n1j = np.asarray(DATA['product_aes'], dtype=np.float64)
    ni1 = np.asarray(DATA['count_across_brands'], dtype=np.float64)
    num_cell = len(n11)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method,
                                  method_alpha)

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    log_rfet = np.log(n11*n00 / (n10*n01))
    pval_fish_uni = np.empty((num_cell))
    for p in range(num_cell):
        table = [[n11[p], n10[p]],
                 [n01[p], n00[p]]]
        pval_fish_uni[p] = fisher_exact(table, alternative='greater')[1]

    if mid_pval:
        for p in range(num_cell):
            pval_fish_uni[p] = (pval_fish_uni[p] -
                                .5 *
                                hypergeom.pmf(n11[p], n11[p] + n10[p],
                                              n11[p] + n01[p],
                                              n10[p] + n00[p]))

    pval_uni = pval_fish_uni
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0
    RankStat = pval_uni

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

    if decision_metric == 'fdr':
        num_signals = (FDR <= decision_thres).sum()
    elif decision_metric == 'signals':
        num_signals = min(decision_thres, num_cell)
    elif decision_metric == 'rank':
        num_signals = (RankStat <= decision_thres).sum()

    RC = Container()
    RC.all_signals = pd.DataFrame({'Product': DATA['product_name'],
                                   'Adverse Event': DATA['ae_name'],
                                   'Count': n11,
                                   'Expected Count': expected,
                                   'p_value': RankStat,
                                   'PRR': np.exp(log_rfet),
                                   'product margin': n1j,
                                   'event margin': ni1,
                                   'FDR': FDR}).sort_values(by=['p_value'])

    RC.signals = RC.all_signals.loc[0:num_signals, ]
    RC.num_signals = num_signals
    return RC

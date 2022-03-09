import pandas as pd
import numpy as np
from scipy.special import gdtr
from scipy.optimize import minimize
from sympy.functions.special import gamma_functions

from ..utils import Container
from ..utils import calculate_expected
from ..utils.distribution_funcs.negative_binomials import dnbinom, pnbinom
from ..utils.distribution_funcs.quantile_funcs import quantiles

dnbinom = np.vectorize(dnbinom)
pnbinom = np.vectorize(pnbinom)
digamma = np.vectorize(gamma_functions.digamma)
quantiles = np.vectorize(quantiles)


def gps(
    container,
    relative_risk=1,
    min_events=1,
    decision_metric="rank",
    decision_thres=0.05,
    ranking_statistic="log2",
    truncate=False,
    truncate_thres=1,
    prior_init={"alpha1": 0.2041, "beta1": 0.05816, "alpha2": 1.415, "beta2": 1.838, "w": 0.0969},
    prior_param=None,
    expected_method="mantel-haentzel",
    method_alpha=1,
):
    """
    A multi-item gamma poisson shrinker algo for disproportionality analysis

    Arguments:

        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        relative_risk (float): The relative risk value

        min_events: The min number of AE reports to be considered a signal

        decision_metric (str): The metric used for detecting signals:
                            {fdr = false detection rate,
                            signals = number of signals,
                            rank = ranking statistic}

        decision_thres (float): The min thres value for the decision_metric

        ranking_statistic (str): How to rank signals:
                                {p-value-posterior probability,
                                quantile-5% quantile of the lambda dist,
                                log2-Posterior expectation of log2(lambda)}

        truncate: Calculate data hyperparameters with at least
                    truncate_thres notifications

        truncate_thres: Threshold for hyper parameter calculations

        prior_init (dict): The priors for multi-item gamma poisson shrinkage.
                        By default they are the priors from DuMouchel's
                        1999 paper.

        prior_param: Chosen hyper parameters. Default uses maximization
                    of marginal likelihood

        expected_method: The method of calculating the expected counts for
                        the disproportionality analysis.

        method_alpha: If the expected_method is negative-binomial, this
                    parameter is the alpha parameter of the distribution.

    """
    priors = np.asarray(
        [prior_init["alpha1"], prior_init["beta1"], prior_init["alpha2"], prior_init["beta2"], prior_init["w"]]
    )
    DATA = container.data
    N = container.N

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    p_out = True

    if prior_param is None:
        p_out = False

        if not truncate:
            data_cont = container.contingency
            n1__mat = data_cont.sum(axis=1)
            n_1_mat = data_cont.sum(axis=0)
            rep = len(n_1_mat)
            n1__c = np.tile(n1__mat.values, reps=rep)
            rep = len(n1__mat)
            n_1_c = np.repeat(n_1_mat.values, repeats=rep)
            E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
            n11_c_temp = []
            for col in data_cont:
                n11_c_temp.extend(list(data_cont[col]))
            n11_c = np.asarray(n11_c_temp)

            p_out = minimize(non_truncated_likelihood, x0=priors, args=(n11_c, E_c), options={"maxiter": 500})
        elif truncate:
            truncate = truncate_thres - 1
            p_out = minimize(
                truncated_likelihood,
                x0=priors,
                args=(n11[n11 >= truncate_thres], expected[n11 >= truncate_thres], truncate),
                options={"maxiter": 500},
            )

        prior_param = p_out.x
        code_convergence = p_out.message

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]
        expected = expected[n11 >= min_events]
        n1j = n1j[n11 >= min_events]
        ni1 = ni1[n11 >= min_events]
        n11 = n11[n11 >= min_events]

    num_cell = len(n11)
    posterior_probability = []

    # Posterior probability of the null hypothesis
    qdb1 = dnbinom(n11, size=priors[0], prob=priors[1] / (priors[1] + expected))
    qdb2 = dnbinom(n11, size=priors[2], prob=priors[3] / (priors[3] + expected))
    Qn = priors[4] * qdb1 / (priors[4] * qdb1 + (1 - priors[4]) * qdb2)

    gd1 = gdtr(relative_risk, priors[0] + n11, priors[1] + expected)
    gd2 = gdtr(relative_risk, priors[2] + n11, priors[3] + expected)
    posterior_probability = Qn * gd1 + (1 - Qn) * gd2

    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(priors[1] + expected)
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(priors[3] + expected)
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)

    # Calculation of the Lower Bound.
    LB = quantiles(0.05, Qn, priors[0] + n11, priors[1] + expected, priors[2] + n11, priors[3] + expected)

    # Assignment based on the method
    if ranking_statistic == "p_value":
        RankStat = posterior_probability
    elif ranking_statistic == "quantile":
        RankStat = LB
    elif ranking_statistic == "log2":
        RankStat = np.array([x.evalf() for x in EBlog2])

    post_cumsum = np.cumsum(posterior_probability)
    post_1_cumsum = np.cumsum(1 - posterior_probability)
    post_1_sum = sum(1 - posterior_probability)
    post_range = np.arange(1, len(posterior_probability) + 1)
    if ranking_statistic == "p_value":
        FDR = post_cumsum / np.array(post_range)
        FNR = np.array(post_1_cumsum) / ((num_cell - post_range) + 1e-7)
        Se = np.cumsum((1 - posterior_probability)) / post_1_sum
        Sp = np.array(post_cumsum) / (num_cell - post_1_sum)
    else:
        FDR = post_cumsum / post_range
        FNR = np.array(list(reversed(post_1_cumsum))) / ((num_cell - post_range) + 1e-7)
        Se = np.cumsum((1 - posterior_probability)) / post_1_sum
        Sp = np.array(list(reversed(post_cumsum))) / (num_cell - post_1_sum)

    # Number of signals according to the decision rule (pp/FDR/Nb of Signals)

    if decision_metric == "fdr":
        num_signals = np.sum(FDR <= decision_thres)
    elif decision_metric == "signals":
        num_signals = np.min((RankStat <= decision_thres).sum(), num_cell)
    elif decision_metric == "rank":
        if ranking_statistic == "p_value":
            num_signals = np.sum(RankStat <= decision_thres)
        elif ranking_statistic == "quantile":
            num_signals = np.sum(RankStat >= decision_thres)
        elif ranking_statistic == "log2":
            num_signals = np.sum(RankStat >= decision_thres)

    name = DATA["product_name"]
    ae = DATA["ae_name"]
    count = n11
    RES = Container(params=True)

    # list of the parameters used
    RES.input_param = (
        relative_risk,
        min_events,
        decision_metric,
        decision_thres,
        ranking_statistic,
        truncate,
        truncate_thres,
    )

    # vector of the final a priori parameters (if p_out=TRUE)
    if p_out:
        RES.param["prior_param"] = prior_param
    # vector of the initial a priori and final a priori parameters
    if not p_out:
        RES.param["prior_init"] = prior_init
        RES.param["prior_param"] = prior_param
        RES.param["convergence"] = code_convergence

    # SIGNALS RESULTS and presentation
    if ranking_statistic == "p_value":
        RES.all_signals = pd.DataFrame(
            {
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": expected,
                "p_value": RankStat,
                "count/expected": (count / expected),
                "product margin": n1j,
                "event margin": ni1,
                "fdr": FDR,
                "FNR": FNR,
                "Se": Se,
                "Sp": Sp,
            }
        ).sort_values(by=[ranking_statistic])

    elif ranking_statistic == "quantile":
        RES.all_signals = pd.DataFrame(
            {
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": expected,
                "quantile": RankStat,
                "count/expected": (count / expected),
                "product margin": n1j,
                "event margin": ni1,
                "fdr": FDR,
                "FNR": FNR,
                "Se": Se,
                "Sp": Sp,
                "posterior_probability": posterior_probability,
            }
        )
        RES.all_signals = RES.all_signals.sort_values(by=[ranking_statistic], ascending=False)
    else:
        RES.all_signals = pd.DataFrame(
            {
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": expected,
                "log2": RankStat,
                "count/expected": (count / expected),
                "product margin": n1j,
                "event margin": ni1,
                "fdr": FDR,
                "FNR": FNR,
                "Se": Se,
                "Sp": Sp,
                "LowerBound": LB,
                "p_value": posterior_probability,
            }
        )
        RES.all_signals = RES.all_signals.sort_values(by=[ranking_statistic], ascending=False)

    # List of Signals generated according to the method
    RES.all_signals.index = np.arange(0, len(RES.all_signals.index))
    if num_signals > 0:
        num_signals -= 1
    else:
        num_signals = 0
    RES.signals = RES.all_signals.iloc[
        0:num_signals,
    ]

    # Number of signals
    RES.num_signals = num_signals

    return RES


def non_truncated_likelihood(p, n11, E):
    dnb1 = np.nan_to_num(dnbinom(n11, prob=p[1] / (p[1] + E), size=p[0]))
    dnb2 = np.nan_to_num(dnbinom(n11, prob=p[3] / (p[3] + E), size=p[2]))
    term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
    return np.sum(-np.log(term))


def truncated_likelihood(p, n11, E, truncate):
    dnb1 = dnbinom(n11, size=p[0], prob=p[1] / (p[1] + E))
    dnb2 = dnbinom(n11, size=p[2], prob=p[3] / (p[3] + E))
    term1 = p[4] * dnb1 + (1 - p[4]) * dnb2

    pnb1 = pnbinom(truncate, size=p[0], prob=p[1] / (p[1] + E))
    pnb2 = pnbinom(truncate, size=p[2], prob=p[3] / (p[3] + E))
    term2 = 1 - (p[4] * pnb1 + (1 - p[4]) * pnb2)

    return np.sum(-np.log(term1 / term2))


class ResultContainer:
    def __init__(self, empty=False):
        self.input_param = None
        self.all_signals = None
        self.signals = None
        self.num_signals = None
        self.param = {}
        self.empty = empty

    def export(self, name, index=False):
        writer = pd.ExcelWriter(name)
        if not self.empty:
            self.signals.to_excel(writer, "Signals", index=index)
            self.all_signals.to_excel(writer, "ALL_Signals", index=index)
            writer.save()
        else:
            pd.DataFrame().to_excel(writer, "No Candidates")

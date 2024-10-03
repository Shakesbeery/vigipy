import pandas as pd
import numpy as np
import warnings
from scipy.special import gdtr
from scipy.stats import nbinom
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

EPS = np.finfo(np.float32).eps
BOUNDED_METHODS = {
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
}


def gps(
    container,
    relative_risk=1,
    min_events=1,
    decision_metric="rank",
    decision_thres=0.05,
    ranking_statistic="log2",
    truncate=False,
    truncate_thres=1,
    prior_init={
        "alpha1": 0.2041,
        "beta1": 0.05816,
        "alpha2": 1.415,
        "beta2": 1.838,
        "w": 0.0969,
    },
    prior_param=None,
    expected_method="mantel-haentzel",
    method_alpha=1,
    minimization_method="CG",
    minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1)),
    minimization_options=None,
):
    """
    Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS) using prior distributions
    for adverse event and product feature data.

    Parameters:
    -----------
    container : object
        A container object holding the input data, including event counts (`events`),
        product-event pairs (`product_aes`), and across-brand counts (`count_across_brands`).
    relative_risk : float, optional (default=1)
        The threshold for relative risk used in the posterior probability calculations.
    min_events : int, optional (default=1)
        The minimum number of events required for an adverse event to be considered in the analysis.
    decision_metric : str, optional (default="rank")
        The decision rule for signal detection. Options are 'rank', 'fdr', or 'signals'.
    decision_thres : float, optional (default=0.05)
        The threshold used in the decision rule to filter significant signals.
    ranking_statistic : str, optional (default="log2")
        The ranking statistic used to order the results. Options include 'log2', 'p_value', or 'quantile'.
    truncate : bool, optional (default=False)
        Whether to truncate likelihoods below a certain threshold for stability in signal detection.
    truncate_thres : float, optional (default=1)
        The truncation threshold for likelihood values if `truncate` is set to True.
    prior_init : dict, optional
        Initial values for the prior distributions used in Bayesian inference. Contains parameters for two Poisson
        distributions (alpha1, beta1, alpha2, beta2) and the mixture weight (w).
    prior_param : array, optional (default=None)
        Manually provided prior distribution parameters. If None, the function estimates priors using optimization.
    expected_method : str, optional (default="mantel-haentzel")
        The method used to calculate the expected event counts. Options include "mantel-haentzel", "negative-binomial" and "poisson".
    method_alpha : float, optional (default=1)
        Dispersion parameter used in the expected value calculation method.
    minimization_method : str, optional (default="CG")
        The optimization method used for estimating prior parameters if `prior_param` is None.
    minimization_bounds : tuple, optional
        Bounds on the prior parameter values for the optimization process.
    minimization_options : dict, optional
        Options for the minimization routine.

    Returns:
    --------
    RES : object
        A container object with the following attributes:
        - `param`: A dictionary of input parameters, including prior initialization and optimization results.
        - `all_signals`: A DataFrame containing detailed results of signal detection, including posterior probabilities,
          expected counts, and ranking statistics.
        - `signals`: A DataFrame of filtered signals according to the decision metric and threshold.
        - `num_signals`: The number of signals detected based on the decision rule.

    Notes:
    ------
    - This function implements a Bayesian model to calculate posterior probabilities using a mixture of two negative
      binomial distributions.
    - The function can apply different ranking statistics to order results, such as p-value, quantile, or log2.
    - The optimization process is used to estimate the prior parameters unless provided manually.
    - The function can handle truncation for numerical stability when dealing with sparse data.
    """
    input_params = locals()
    del input_params["container"]

    priors = np.asarray(
        [
            prior_init["alpha1"],
            prior_init["beta1"],
            prior_init["alpha2"],
            prior_init["beta2"],
            prior_init["w"],
        ]
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
        if minimization_method not in BOUNDED_METHODS:
            minimization_bounds = None

        if minimization_options is None:
            minimization_options = {}

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

            p_out = minimize(
                non_truncated_likelihood,
                x0=priors,
                args=(n11_c, E_c),
                options={"maxiter": 500},
                method=minimization_method,
                bounds=minimization_bounds,
                **minimization_options,
            )
        elif truncate:
            trunc = truncate_thres - 1
            p_out = minimize(
                truncated_likelihood,
                x0=priors,
                args=(
                    n11[n11 >= truncate_thres],
                    expected[n11 >= truncate_thres],
                    trunc,
                ),
                options={"maxiter": 500},
                method=minimization_method,
                bounds=minimization_bounds,
                **minimization_options,
            )

        priors = p_out.x
        if np.any(priors < 0) or priors[4] > 1:
            warnings.warn(
                f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
            )
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
    qdb1 = nbinom(n=priors[0], p=priors[1] / (priors[1] + expected)).pmf(n11)
    qdb2 = nbinom(n=priors[2], p=priors[3] / (priors[3] + expected)).pmf(n11)

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
    LB = quantiles(
        0.05,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )

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
        num_signals = min((RankStat <= decision_thres).sum(), num_cell)
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
    RES.param["input_params"] = input_params
    RES.param["prior_init"] = prior_init
    RES.param["prior_param"] = priors
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
        RES.all_signals = RES.all_signals.sort_values(
            by=[ranking_statistic], ascending=False
        )
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
        RES.all_signals = RES.all_signals.sort_values(
            by=[ranking_statistic], ascending=False
        )

    # List of Signals generated according to the method
    RES.all_signals.index = np.arange(0, len(RES.all_signals.index))
    if num_signals > 0:
        num_signals -= 1
    else:
        num_signals = 0
    RES.signals = RES.all_signals.iloc[0:num_signals,]

    # Number of signals
    RES.num_signals = num_signals

    return RES


def non_truncated_likelihood(p, n11, E):
    dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
    dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
    term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
    return np.sum(-np.log(term))


def truncated_likelihood(p, n11, E, truncate):
    dnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).pmf(n11)
    dnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).pmf(n11)
    term1 = p[4] * dnb1 + (1 - p[4]) * dnb2

    pnb1 = nbinom(n=p[0], p=p[1] / (p[1] + E)).cdf(truncate)
    pnb2 = nbinom(n=p[2], p=p[3] / (p[3] + E)).cdf(truncate)
    term2 = 1 - (p[4] * pnb1 + (1 - p[4]) * pnb2)

    return np.sum(-np.log(term1 / term2))

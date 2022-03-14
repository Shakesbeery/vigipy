import warnings

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.tools.sm_exceptions import PerfectSeparationError


def __cameron_trivedi_dispersion(df_row):
    """
    Use the dispersion test from Cameron and Trivedi's 1990 paper:

    Cameron, A. and Trivedi, Pravin, (1990), Regression-based tests for
    overdispersion in the Poisson model, Journal of Econometrics,
    46, issue 3, p. 347-364, https://doi.org/10.1016/0304-4076(90)90014-K.

    Arguments:
        df_row: A single row of a Pandas dataframe object

    Returns:
        (float) The Cameron-Trivedi response variable

    """
    y = df_row["events"]
    m = df_row["mu"]
    return ((y - m) ** 2 - y) / m


def __test_dispersion(model, data):
    """
    Apply an auxiliary formula to run linear regression for the dispersion
    test. If the lower_bound is >0, overdispersion is most likely confirmed.
    The alpha parameter can be utilized in a negative binomial family GLM to
    better model the data.

    Arguments:
       model (statsmodel model): A poisson family GLM fitted model
       data (DataFrame): A dataframe with 'events' as a column

    Returns:
        alpha: The optimized alpha term for a negative binomial model
        lower_bound: The 5% confidence interval value
        upper_bound: the 95% confidence interval value

    """
    data["mu"] = model.mu
    data["response"] = data.apply(__cameron_trivedi_dispersion, axis=1)
    results = smf.ols("response ~ mu - 1", data).fit()

    alpha_conf_int = results.conf_int(0.05).loc["mu"]
    alpha = results.params[0]
    lower_bound = alpha_conf_int.loc[0]
    upper_bound = alpha_conf_int.loc[1]

    return alpha, lower_bound, upper_bound


def __mh(N, n1j, ni1):
    """
    The Mantel-Haentzel calculation for expected counts.

    """
    return n1j * ni1 / N


def __stats_method(n1j, ni1, n11, family):
    """
    If the expected counts are calculated via a statistical model,
    this function will do so. Expected counts are considered a
    function of n1j and ni1.

    Arguments:
        n1j (iterable): All adverse events for a single product
        ni1 (iterable): Total count of a particular AE across all products
        n11 (iterable): Total count of a particular AE for a particular product
        family (statsmodel family): The GLM family

    Returns:
        The expected counts for n11

    """
    data = pd.DataFrame({"events": n11, "prod_events": n1j, "ae_events": ni1})
    model = smf.glm(formula="events ~ prod_events+ae_events", data=data, family=family)
    model = model.fit()

    if isinstance(family, sm.families.Poisson):
        dispersion = model.pearson_chi2 / model.df_resid
        if dispersion > 2:
            alpha, lb, ub = __test_dispersion(model, data)
            warning_string = f"""Variance does not equal the mean! Data likely overdispersed...\n
                                Consider utilizing the negative-binomial family instead of poisson.\n
                                Cameron-Trivedi alpha: {alpha:5.4f}, CI: ({lb}, {ub})"""
            warnings.warn(warning_string)

    return model.predict(data[["prod_events", "ae_events"]]).values


def calculate_expected(N, n1j, ni1, n11, method="mantel-haentzel", alpha=1):
    """
    Calculate the expected counts for disproportionality analysis.

    Arguments:
        N (int): THe total number of adverse events across all products
        n1j (iterable): All adverse events for a single product
        ni1 (iterable): Total count of a particular AE across all products
        n11 (iterable): Total count of a particular AE for a particular product
        method (str): The method for calculating the expected counts
        alpha (float): If using a NegativeBinomial family, alpha is the
                       parameter of that distribution.

    Returns:
        The expected counts for n11

    """
    try:
        assert method in ("mantel-haentzel", "negative-binomial", "poisson")
    except AssertionError:
        err_msg = "{0} not a supported method. Please choose from {1}"
        raise AssertionError(err_msg.format(method, ("mantel-haentzel", "negative-binomial", "poisson")))

    if method == "mantel-haentzel":
        return __mh(N, n1j, ni1)
    elif method == "negative-binomial":
        try:
            return __stats_method(n1j, ni1, n11, sm.families.NegativeBinomial(alpha=alpha))
        except PerfectSeparationError:
            print("Perfect separation of data detected. Defaulting to Mantel-Haentzel estimation.")
            return __mh(N, n1j, ni1)
    elif method == "poisson":
        try:
            return __stats_method(n1j, ni1, n11, sm.families.Poisson())
        except PerfectSeparationError:
            print("Perfect separation of data detected. Defaulting to Mantel-Haentzel estimation.")
            return __mh(N, n1j, ni1)

import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoLars, LassoLarsIC
import numpy as np

from ..utils import Container


def lasso(
    container,
    lasso_thresh=0,
    alpha=0.5,
    min_events=3,
    num_bootstrap=10,
    ci=95,
    use_lars=False,
    use_IC=False,
    IC_criterion="bic",
    lasso_kwargs=None,
    use_glm=False,
    nb_alpha = 1
):
    """
    Applies LASSO regression or its variants to detect signals between product features and adverse events, 
    optionally using bootstrap confidence intervals.

    Parameters:
    -----------
    container : object
        A container object holding product features (`product_features`) and event outcomes (`event_outcomes`) in separate attributes.
    lasso_thresh : float, optional (default=0)
        The threshold for filtering out LASSO coefficients. Coefficients below this value are ignored in the final results.
    alpha : float, optional (default=0.5)
        The regularization strength for LASSO. Higher values lead to stronger regularization.
    min_events : int, optional (default=3)
        The minimum number of events required for an adverse event to be considered in the analysis.
    num_bootstrap : int, optional (default=10)
        The number of bootstrap resamples to use for computing confidence intervals for LASSO coefficients.
    ci : int, optional (default=95)
        The confidence interval percentage for the bootstrapped LASSO coefficients (e.g., 95% CI).
    use_lars : bool, optional (default=False)
        Whether to use LASSO-LARS (Least Angle Regression) instead of regular LASSO.
    use_IC : bool, optional (default=False)
        Whether to use LASSO with Information Criterion (LassoLarsIC) for model selection.
    IC_criterion : str, optional (default="bic")
        The information criterion to be used if `use_IC` is True. Choices are "aic" (Akaike) or "bic" (Bayesian).
    lasso_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the LASSO model.
    use_glm : bool, optional (default=False)
        If True, use Generalized Linear Model (GLM) with L1 regularization instead of LASSO.
    nb_alpha : float, optional (default=1)
        Dispersion parameter for Negative Binomial GLM if `use_glm` is True.

    Returns:
    --------
    RES : object
        A container object that holds the following attributes:
        - `param`: A dictionary of input parameters used for the LASSO.
        - `all_signals`: A DataFrame containing all computed LASSO coefficients, confidence intervals, and related information.
        - `signals`: A filtered DataFrame of signals where LASSO coefficients exceed the `lasso_thresh`.
        - `num_signals`: The number of signals detected.

    Notes:
    ------
    - When `use_glm` is True, GLM with Negative Binomial regression is applied instead of LASSO.
    - Confidence intervals for the LASSO coefficients are generated via bootstrapping iff `use_glm` is False.
    - The function iterates over adverse events, using product features as predictors, and applies the chosen LASSO model to find associations.
    """
    input_params = locals()
    del input_params["container"]
    X = container.product_features
    ys = container.event_outcomes
    res = defaultdict(list)

    if lasso_kwargs is None:
        lasso_kwargs = dict()

    # Set type of LASSO per user inputs
    if use_glm:
        pass
    elif use_IC:
        lasso = LassoLarsIC(criterion=IC_criterion, **lasso_kwargs)
    elif use_lars:
        lasso = LassoLars(alpha=alpha, **lasso_kwargs)
    else:
        lasso = Lasso(alpha=alpha, **lasso_kwargs)

    # Iterate over adverse events using product as features for DA
    for column in ys.columns:
        y = ys[column].values
        if y.sum() < min_events:
            for product in X.columns:
                res["Product"].append(product)
                res["Adverse Event"].append(column)
                res["LASSO Coefficient"].append(0)
                res["CI Lower"].append(0)
                res["CI Upper"].append(0)
            continue

        if use_glm:
            nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=nb_alpha))
            results = nb.fit_regularized(L1_wt=1)
            all_coefs = np.clip(results.params.values.copy(), 0, None)
            ci_lower = np.zeros(len(all_coefs))
            ci_upper = np.zeros(len(all_coefs))
        else:
            lasso.fit(X, y)
            all_coefs = lasso.coef_.copy()

            # Initialize a list to store bootstrap coefficients
            bootstrap_coefficients = []

            # Bootstrap resampling
            for _ in range(num_bootstrap):
                # Sample with replacement
                bootstrap_sample_indices = np.random.choice(
                    range(len(ys)), size=len(ys), replace=True
                )
                X_bootstrap = X.iloc[bootstrap_sample_indices]
                y_bootstrap = y[bootstrap_sample_indices]

                # Fit LASSO model to bootstrap sample
                if use_glm:
                    nb = sm.GLM(y_bootstrap, X_bootstrap, family=sm.families.NegativeBinomial(alpha=nb_alpha))
                    results = nb.fit_regularized(L1_wt=1)
                    boot_coefs = results.params.values.copy()
                else:
                    lasso.fit(X_bootstrap, y_bootstrap)
                    boot_coefs = lasso.coef_.copy()
                bootstrap_coefficients.append(boot_coefs)

            bootstrap_coefficients = np.array(bootstrap_coefficients)

            # Calculate confidence intervals for each coefficient
            ci_lower = np.percentile(bootstrap_coefficients, (100 - ci) / 2.0, axis=0)
            ci_upper = np.percentile(bootstrap_coefficients, 100 - (100 - ci) / 2.0 , axis=0)

        for product, co, ci_u, ci_l in zip(X.columns, all_coefs, ci_upper, ci_lower):
            res["Product"].append(product)
            res["Adverse Event"].append(column)
            res["LASSO Coefficient"].append(co)
            res["CI Lower"].append(ci_l)
            res["CI Upper"].append(ci_u)

    RES = Container(params=True)

    # list of the parameters used
    RES.param = input_params
    RES.all_signals = pd.DataFrame(res).sort_values(by="LASSO Coefficient", ascending=False)
    RES.signals = RES.all_signals.loc[RES.all_signals["LASSO Coefficient"] > lasso_thresh]

    # Number of signals
    RES.num_signals = len(RES.signals)

    return RES

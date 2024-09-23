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
    """The LASSO algorithm for product-event pair signal detection. This function supports vanilla LASSO, LASSO using LARS and
    LASSO using the IC to determine alpha values.

    Args:
        container (Container): A binary container from `convert_binary`
        alpha (float): The alpha parameter for the LASSO model
        min_events (int, optional): Minimum number of adverse events of a particular type to attempt a product-ADR analysis. Defaults to 3.
        num_bootstrap (int, optional): How many iterations of bootstrapping to run for CI calculations. Defaults to 10.
        ci (int, optional): Upper confidence interval %. The lower bound is considered as 100 - ci. Defaults to 95.
        use_lars (bool, optional): Use LASSO with least angle regression. Defaults to False.
        use_IC (bool, optional): Use information criterion to determine alpha. Defaults to False.
        IC_criterion (str, optional): Either Akaike (AIC) or Bayes (BIC) information criterion. Defaults to "bic".

    Returns:
        _type_: _description_
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
            all_coefs = results.params.values.copy()
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
            ci_lower = np.percentile(bootstrap_coefficients, (100 - ci), axis=0)
            ci_upper = np.percentile(bootstrap_coefficients, ci, axis=0)

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

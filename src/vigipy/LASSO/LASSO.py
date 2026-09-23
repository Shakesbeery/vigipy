from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import (
    Lasso,
    LassoLars,
    LassoLarsIC,
    LogisticRegression,
    LogisticRegressionCV,
)

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params


def lasso(
    container: DataContainer,
    lasso_thresh: float = 0,
    alpha: float = 0.5,
    min_events: int = 3,
    num_bootstrap: int = 10,
    ci: int = 95,
    use_lars: bool = False,
    use_IC: bool = False,
    IC_criterion: Literal["aic", "bic"] = "bic",
    lasso_kwargs: dict | None = None,
    use_glm: bool = False,
    nb_alpha: float = 1,
    lasso_alpha: float = 1e-9,
    family: Literal["logistic", "linear", "negative_binomial"] = "logistic",
    C: float = 1.0,
    decision_metric: Literal["lower_bound", "coefficient"] = "lower_bound",
    use_cv: bool = False,
    cv: int = 3,
    use_bootstrap: bool = False,
) -> AnalysisResult:
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
    lasso_alpha : float, optional (default=1e-9)
        ElasticNet alpha parameter for GLM if `use_glm` is True.

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
    if use_glm:
        resolved_family = "negative_binomial"
    elif use_lars or use_IC:
        resolved_family = "linear"
    else:
        resolved_family = family

    input_params = {
        "lasso_thresh": lasso_thresh,
        "alpha": alpha,
        "min_events": min_events,
        "num_bootstrap": num_bootstrap,
        "ci": ci,
        "use_lars": use_lars,
        "use_IC": use_IC,
        "IC_criterion": IC_criterion,
        "use_glm": use_glm,
        "nb_alpha": nb_alpha,
        "lasso_alpha": lasso_alpha,
        "family": resolved_family,
        "C": C,
        "decision_metric": decision_metric,
        "use_cv": use_cv,
        "cv": cv,
        "use_bootstrap": use_bootstrap,
    }
    X = container.product_features
    ys = container.event_outcomes
    X_arr = np.ascontiguousarray(X.values, dtype=np.float64)
    n_samples, n_features = X_arr.shape
    products = list(X.columns)
    rng = np.random.default_rng(42)
    z_crit = float(stats.norm.ppf(1.0 - (100.0 - ci) / 200.0))
    res = defaultdict(list)

    if lasso_kwargs is None:
        lasso_kwargs = dict()

    # Pre-instantiate linear model if family == 'linear'
    if resolved_family == "linear":
        if use_IC:
            lin_model = LassoLarsIC(criterion=IC_criterion, **lasso_kwargs)
        elif use_lars:
            lin_model = LassoLars(alpha=alpha, **lasso_kwargs)
        else:
            lin_model = Lasso(alpha=alpha, **lasso_kwargs)

    # Iterate over adverse events using product features for DA
    for column in ys.columns:
        y = np.ascontiguousarray(ys[column].values, dtype=np.float64)
        total_events = float(np.sum(y))

        # Calculate co-occurrence count for each product
        counts = np.sum(X_arr * y[:, None], axis=0).astype(int)

        if total_events < min_events:
            for idx, product in enumerate(products):
                res["Product"].append(product)
                res["Adverse Event"].append(column)
                res["Count"].append(int(counts[idx]))
                res["LASSO Coefficient"].append(0.0)
                res["CI Lower"].append(0.0)
                res["CI Upper"].append(0.0)
                if resolved_family == "logistic":
                    res["aROR"].append(1.0)
                    res["aROR Lower"].append(1.0)
                    res["aROR Upper"].append(1.0)
                    res["SE"].append(0.0)
                    res["p_value"].append(1.0)
            continue

        if resolved_family == "logistic":
            pos_cases = int(np.sum(y))
            neg_cases = len(y) - pos_cases
            if use_cv and min(pos_cases, neg_cases) >= cv:
                cv_kwargs = dict(
                    Cs=[0.05, 0.1, 0.5, 1.0, 2.0],
                    cv=cv,
                    penalty="l1",
                    solver="liblinear",
                    scoring="roc_auc",
                    fit_intercept=True,
                    random_state=42,
                )
                if lasso_kwargs:
                    cv_kwargs.update(lasso_kwargs)
                clf = LogisticRegressionCV(**cv_kwargs)
            else:
                log_kwargs = dict(
                    penalty="l1",
                    solver="liblinear",
                    C=C,
                    fit_intercept=True,
                    random_state=42,
                )
                if lasso_kwargs:
                    log_kwargs.update(lasso_kwargs)
                clf = LogisticRegression(**log_kwargs)

            clf.fit(X_arr, y)
            coefs = clf.coef_[0]

            if use_bootstrap:
                boot_coefs_list = []
                for _ in range(num_bootstrap):
                    b_idx = rng.choice(n_samples, size=n_samples, replace=True)
                    y_b = y[b_idx]
                    if len(np.unique(y_b)) < 2:
                        boot_coefs_list.append(np.zeros(n_features))
                        continue
                    clf_b = LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=C,
                        fit_intercept=True,
                        random_state=42,
                        **lasso_kwargs,
                    )
                    clf_b.fit(X_arr[b_idx], y_b)
                    boot_coefs_list.append(clf_b.coef_[0].copy())
                boot_arr = np.array(boot_coefs_list)
                ci_l = np.percentile(boot_arr, (100.0 - ci) / 2.0, axis=0)
                ci_u = np.percentile(boot_arr, 100.0 - (100.0 - ci) / 2.0, axis=0)
                se_vec = np.std(boot_arr, axis=0)
                z_sc = np.abs(coefs) / np.maximum(se_vec, 1e-9)
                p_vec = np.clip(2.0 * (1.0 - stats.norm.cdf(z_sc)), 0.0, 1.0)
            else:
                # Fast, exact post-LASSO Fisher Information standard errors
                p_pred = clf.predict_proba(X_arr)[:, 1]
                p_pred = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
                w = p_pred * (1.0 - p_pred)

                active_mask = np.abs(coefs) > 1e-6
                active_indices = np.where(active_mask)[0]

                ci_l = np.zeros(n_features)
                ci_u = np.zeros(n_features)
                se_vec = np.zeros(n_features)
                p_vec = np.ones(n_features)

                if len(active_indices) > 0:
                    is_partition = bool(np.allclose(np.sum(X_arr[:, active_indices], axis=1), 1.0))
                    if is_partition:
                        X_sub = X_arr[:, active_indices]
                        H = X_sub.T @ (w[:, None] * X_sub) + 1e-4 * np.eye(len(active_indices))
                        offset = 0
                    else:
                        X_sub = np.column_stack([np.ones(n_samples), X_arr[:, active_indices]])
                        H = X_sub.T @ (w[:, None] * X_sub) + 1e-4 * np.eye(len(active_indices) + 1)
                        offset = 1

                    try:
                        V = np.linalg.inv(H)
                        active_se = np.sqrt(np.maximum(np.diag(V)[offset:], 1e-8))
                        se_vec[active_indices] = active_se
                        ci_l[active_indices] = coefs[active_indices] - z_crit * active_se
                        ci_u[active_indices] = coefs[active_indices] + z_crit * active_se
                        z_scores = np.abs(coefs[active_indices]) / active_se
                        p_vec[active_indices] = np.clip(2.0 * (1.0 - stats.norm.cdf(z_scores)), 0.0, 1.0)
                    except np.linalg.LinAlgError:
                        diag_w = np.sum(w[:, None] * (X_arr[:, active_indices] ** 2), axis=0) + 1e-4
                        active_se = 1.0 / np.sqrt(diag_w)
                        se_vec[active_indices] = active_se
                        ci_l[active_indices] = coefs[active_indices] - z_crit * active_se
                        ci_u[active_indices] = coefs[active_indices] + z_crit * active_se
                        z_scores = np.abs(coefs[active_indices]) / active_se
                        p_vec[active_indices] = np.clip(2.0 * (1.0 - stats.norm.cdf(z_scores)), 0.0, 1.0)

            aror = np.exp(coefs)
            aror_l = np.exp(ci_l)
            aror_u = np.exp(ci_u)

            for idx, product in enumerate(products):
                res["Product"].append(product)
                res["Adverse Event"].append(column)
                res["Count"].append(int(counts[idx]))
                res["LASSO Coefficient"].append(float(coefs[idx]))
                res["CI Lower"].append(float(ci_l[idx]))
                res["CI Upper"].append(float(ci_u[idx]))
                res["aROR"].append(float(aror[idx]))
                res["aROR Lower"].append(float(aror_l[idx]))
                res["aROR Upper"].append(float(aror_u[idx]))
                res["SE"].append(float(se_vec[idx]))
                res["p_value"].append(float(p_vec[idx]))

        elif resolved_family == "negative_binomial":
            nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=nb_alpha))
            results = nb.fit_regularized(L1_wt=1, alpha=lasso_alpha)
            all_coefs = np.clip(results.params.values.copy(), 0, None)
            ci_l = np.zeros(len(all_coefs))
            ci_u = np.zeros(len(all_coefs))

            for idx, product in enumerate(products):
                res["Product"].append(product)
                res["Adverse Event"].append(column)
                res["Count"].append(int(counts[idx]))
                res["LASSO Coefficient"].append(float(all_coefs[idx]))
                res["CI Lower"].append(float(ci_l[idx]))
                res["CI Upper"].append(float(ci_u[idx]))

        else:
            # Linear LASSO
            lin_model.fit(X_arr, y)
            all_coefs = lin_model.coef_.copy()

            bootstrap_coefficients = []
            for _ in range(num_bootstrap):
                b_idx = rng.choice(n_samples, size=n_samples, replace=True)
                lin_model.fit(X_arr[b_idx], y[b_idx])
                bootstrap_coefficients.append(lin_model.coef_.copy())

            boot_arr = np.array(bootstrap_coefficients)
            ci_l = np.percentile(boot_arr, (100.0 - ci) / 2.0, axis=0)
            ci_u = np.percentile(boot_arr, 100.0 - (100.0 - ci) / 2.0, axis=0)

            for idx, product in enumerate(products):
                res["Product"].append(product)
                res["Adverse Event"].append(column)
                res["Count"].append(int(counts[idx]))
                res["LASSO Coefficient"].append(float(all_coefs[idx]))
                res["CI Lower"].append(float(ci_l[idx]))
                res["CI Upper"].append(float(ci_u[idx]))

    all_signals = pd.DataFrame(res).sort_values(by="LASSO Coefficient", ascending=False)
    all_signals.reset_index(drop=True, inplace=True)

    if resolved_family == "logistic":
        if decision_metric == "lower_bound":
            signals = all_signals.loc[
                (all_signals["CI Lower"] > lasso_thresh) & (all_signals["Count"] >= min_events)
            ].copy()
        else:
            signals = all_signals.loc[
                (all_signals["LASSO Coefficient"] > lasso_thresh) & (all_signals["Count"] >= min_events)
            ].copy()
    else:
        signals = all_signals.loc[all_signals["LASSO Coefficient"] > lasso_thresh].copy()

    signals.reset_index(drop=True, inplace=True)

    return AnalysisResult(
        all_signals=all_signals,
        signals=signals,
        num_signals=len(signals),
        params=build_params("lasso", input_params),
    )

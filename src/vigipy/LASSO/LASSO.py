from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.sparse import issparse, hstack
from sklearn.linear_model import (
    Lasso,
    LassoLars,
    LassoLarsIC,
    LogisticRegression,
    LogisticRegressionCV,
)

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params


def _extract_y_vector(series):
    """Extract a dense 1D float array from a Series, safely handling sparse types and NaNs."""
    if hasattr(series, "to_numpy"):
        arr = series.to_numpy(dtype=np.float64, na_value=0.0)
    else:
        arr = np.ascontiguousarray(series.values, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


def _fit_single_adverse_event(
    column_name,
    y,
    X_mat,
    products,
    n_features,
    z_crit,
    C,
    use_cv,
    cv,
    use_bootstrap,
    num_bootstrap,
    lasso_kwargs,
    relaxed,
    Z_mat,
    rng_seed,
    min_events,
    resolved_family,
    lin_model,
    nb_alpha,
    lasso_alpha,
    ci=95,
):
    rng = np.random.default_rng(rng_seed)
    total_events = float(np.sum(y))

    if issparse(X_mat):
        counts_raw = X_mat.T.dot(y)
        if isinstance(counts_raw, np.matrix):
            counts = np.nan_to_num(counts_raw.A1, nan=0.0).astype(int)
        elif hasattr(counts_raw, "toarray"):
            counts = np.nan_to_num(counts_raw.toarray().flatten(), nan=0.0).astype(int)
        else:
            counts = np.nan_to_num(np.array(counts_raw).flatten(), nan=0.0).astype(int)
    else:
        counts = np.nan_to_num(np.sum(X_mat * y[:, None], axis=0), nan=0.0).astype(int)

    res_dict = {
        "Product": [],
        "Adverse Event": [],
        "Count": [],
    }
    
    if resolved_family == "logistic":
        if relaxed:
            res_dict["L1 Coefficient"] = []
        res_dict["LASSO Coefficient"] = []
        res_dict["CI Lower"] = []
        res_dict["CI Upper"] = []
        res_dict["aROR"] = []
        res_dict["aROR Lower"] = []
        res_dict["aROR Upper"] = []
        res_dict["SE"] = []
        res_dict["p_value"] = []
    else:
        res_dict["LASSO Coefficient"] = []
        res_dict["CI Lower"] = []
        res_dict["CI Upper"] = []

    if total_events < min_events:
        for idx, product in enumerate(products):
            res_dict["Product"].append(product)
            res_dict["Adverse Event"].append(column_name)
            res_dict["Count"].append(int(counts[idx]))
            if resolved_family == "logistic":
                if relaxed:
                    res_dict["L1 Coefficient"].append(0.0)
                res_dict["LASSO Coefficient"].append(0.0)
                res_dict["CI Lower"].append(0.0)
                res_dict["CI Upper"].append(0.0)
                res_dict["aROR"].append(1.0)
                res_dict["aROR Lower"].append(1.0)
                res_dict["aROR Upper"].append(1.0)
                res_dict["SE"].append(0.0)
                res_dict["p_value"].append(1.0)
            else:
                res_dict["LASSO Coefficient"].append(0.0)
                res_dict["CI Lower"].append(0.0)
                res_dict["CI Upper"].append(0.0)
        return res_dict

    n_samples = X_mat.shape[0]

    if resolved_family == "logistic":
        pos_cases = int(np.sum(y))
        neg_cases = len(y) - pos_cases

        if relaxed and Z_mat is not None:
            if issparse(X_mat):
                X_fit = hstack([Z_mat, X_mat]).tocsr()
            else:
                X_fit = np.column_stack([Z_mat, X_mat])
        else:
            X_fit = X_mat

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

        clf.fit(X_fit, y)
        coefs_all = clf.coef_[0]

        if relaxed and Z_mat is not None:
            coefs = coefs_all[-n_features:]
        else:
            coefs = coefs_all

        if relaxed:
            l1_coefs = coefs.copy()
            active_mask = np.abs(coefs) > 1e-6

            if not np.any(active_mask):
                for idx, product in enumerate(products):
                    res_dict["Product"].append(product)
                    res_dict["Adverse Event"].append(column_name)
                    res_dict["Count"].append(int(counts[idx]))
                    res_dict["L1 Coefficient"].append(float(l1_coefs[idx]))
                    res_dict["LASSO Coefficient"].append(0.0)
                    res_dict["CI Lower"].append(0.0)
                    res_dict["CI Upper"].append(0.0)
                    res_dict["aROR"].append(1.0)
                    res_dict["aROR Lower"].append(1.0)
                    res_dict["aROR Upper"].append(1.0)
                    res_dict["SE"].append(0.0)
                    res_dict["p_value"].append(1.0)
                return res_dict

            if issparse(X_mat):
                X_active = X_mat[:, active_mask]
                is_partition = bool(np.allclose(X_active.sum(axis=1), 1.0))
            else:
                X_active = X_mat[:, active_mask]
                is_partition = bool(np.allclose(np.sum(X_active, axis=1), 1.0))

            fit_int = not is_partition

            if Z_mat is not None:
                if issparse(X_active):
                    M_stage2 = hstack([Z_mat, X_active]).tocsr()
                else:
                    M_stage2 = np.column_stack([Z_mat, X_active])
            else:
                M_stage2 = X_active

            clf_relaxed = LogisticRegression(
                penalty="l2", C=1e4, solver="lbfgs", fit_intercept=fit_int, max_iter=1000
            )
            clf_relaxed.fit(M_stage2, y)

            relaxed_coefs_all = clf_relaxed.coef_[0]
            if Z_mat is not None:
                relaxed_coefs = relaxed_coefs_all[-np.sum(active_mask):]
            else:
                relaxed_coefs = relaxed_coefs_all

            if issparse(M_stage2):
                M_dense = M_stage2.toarray()
            else:
                M_dense = M_stage2

            p_pred = np.clip(clf_relaxed.predict_proba(M_dense)[:, 1], 1e-6, 1 - 1e-6)
            w = p_pred * (1 - p_pred)

            if fit_int:
                M_inf = np.column_stack([np.ones(n_samples), M_dense])
                offset = 1 + (Z_mat.shape[1] if Z_mat is not None else 0)
            else:
                M_inf = M_dense
                offset = Z_mat.shape[1] if Z_mat is not None else 0

            H = M_inf.T @ (w[:, None] * M_inf)
            V = np.linalg.pinv(H, rcond=1e-7)
            active_se = np.sqrt(np.maximum(np.diag(V)[offset:], 1e-8))

            ci_l = np.zeros(n_features)
            ci_u = np.zeros(n_features)
            se_vec = np.zeros(n_features)
            p_vec = np.ones(n_features)
            final_coefs = np.zeros(n_features)

            active_indices = np.where(active_mask)[0]
            final_coefs[active_indices] = relaxed_coefs
            se_vec[active_indices] = active_se
            ci_l[active_indices] = relaxed_coefs - z_crit * active_se
            ci_u[active_indices] = relaxed_coefs + z_crit * active_se
            z_scores = np.abs(relaxed_coefs) / active_se
            p_vec[active_indices] = np.clip(2.0 * (1.0 - stats.norm.cdf(z_scores)), 0.0, 1.0)

            aror = np.exp(final_coefs)
            aror_l = np.exp(ci_l)
            aror_u = np.exp(ci_u)

            for idx, product in enumerate(products):
                res_dict["Product"].append(product)
                res_dict["Adverse Event"].append(column_name)
                res_dict["Count"].append(int(counts[idx]))
                res_dict["L1 Coefficient"].append(float(l1_coefs[idx]))
                res_dict["LASSO Coefficient"].append(float(final_coefs[idx]))
                res_dict["CI Lower"].append(float(ci_l[idx]))
                res_dict["CI Upper"].append(float(ci_u[idx]))
                res_dict["aROR"].append(float(aror[idx]))
                res_dict["aROR Lower"].append(float(aror_l[idx]))
                res_dict["aROR Upper"].append(float(aror_u[idx]))
                res_dict["SE"].append(float(se_vec[idx]))
                res_dict["p_value"].append(float(p_vec[idx]))

        else:
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
                        **lasso_kwargs if lasso_kwargs else {},
                    )
                    if issparse(X_mat):
                        X_b = X_mat[b_idx]
                    else:
                        X_b = X_mat[b_idx]
                    clf_b.fit(X_b, y_b)
                    boot_coefs_list.append(clf_b.coef_[0].copy())
                boot_arr = np.array(boot_coefs_list)
                ci_l = np.percentile(boot_arr, (100.0 - ci) / 2.0, axis=0)
                ci_u = np.percentile(boot_arr, 100.0 - (100.0 - ci) / 2.0, axis=0)
                se_vec = np.std(boot_arr, axis=0)
                z_sc = np.abs(coefs) / np.maximum(se_vec, 1e-9)
                p_vec = np.clip(2.0 * (1.0 - stats.norm.cdf(z_sc)), 0.0, 1.0)
            else:
                if issparse(X_mat):
                    X_arr_dense = X_mat.toarray()
                else:
                    X_arr_dense = X_mat

                p_pred = clf.predict_proba(X_arr_dense)[:, 1]
                p_pred = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
                w = p_pred * (1.0 - p_pred)

                active_mask = np.abs(coefs) > 1e-6
                active_indices = np.where(active_mask)[0]

                ci_l = np.zeros(n_features)
                ci_u = np.zeros(n_features)
                se_vec = np.zeros(n_features)
                p_vec = np.ones(n_features)

                if len(active_indices) > 0:
                    is_partition = bool(np.allclose(np.sum(X_arr_dense[:, active_indices], axis=1), 1.0))
                    if is_partition:
                        X_sub = X_arr_dense[:, active_indices]
                        H = X_sub.T @ (w[:, None] * X_sub) + 1e-4 * np.eye(len(active_indices))
                        offset = 0
                    else:
                        X_sub = np.column_stack([np.ones(n_samples), X_arr_dense[:, active_indices]])
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
                        diag_w = np.sum(w[:, None] * (X_arr_dense[:, active_indices] ** 2), axis=0) + 1e-4
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
                res_dict["Product"].append(product)
                res_dict["Adverse Event"].append(column_name)
                res_dict["Count"].append(int(counts[idx]))
                res_dict["LASSO Coefficient"].append(float(coefs[idx]))
                res_dict["CI Lower"].append(float(ci_l[idx]))
                res_dict["CI Upper"].append(float(ci_u[idx]))
                res_dict["aROR"].append(float(aror[idx]))
                res_dict["aROR Lower"].append(float(aror_l[idx]))
                res_dict["aROR Upper"].append(float(aror_u[idx]))
                res_dict["SE"].append(float(se_vec[idx]))
                res_dict["p_value"].append(float(p_vec[idx]))

    elif resolved_family == "negative_binomial":
        # Convert sparse to dense if needed for statsmodels
        if issparse(X_mat):
            X_arr_dense = X_mat.toarray()
        else:
            X_arr_dense = X_mat
        nb = sm.GLM(y, X_arr_dense, family=sm.families.NegativeBinomial(alpha=nb_alpha))
        results = nb.fit_regularized(L1_wt=1, alpha=lasso_alpha)
        all_coefs = np.clip(results.params.copy(), 0, None)
        ci_l = np.zeros(len(all_coefs))
        ci_u = np.zeros(len(all_coefs))

        for idx, product in enumerate(products):
            res_dict["Product"].append(product)
            res_dict["Adverse Event"].append(column_name)
            res_dict["Count"].append(int(counts[idx]))
            res_dict["LASSO Coefficient"].append(float(all_coefs[idx]))
            res_dict["CI Lower"].append(float(ci_l[idx]))
            res_dict["CI Upper"].append(float(ci_u[idx]))

    else:
        # Linear LASSO
        if issparse(X_mat):
            X_arr_dense = X_mat.toarray()
        else:
            X_arr_dense = X_mat
            
        lin_model.fit(X_arr_dense, y)
        all_coefs = lin_model.coef_.copy()

        bootstrap_coefficients = []
        for _ in range(num_bootstrap):
            b_idx = rng.choice(n_samples, size=n_samples, replace=True)
            lin_model.fit(X_arr_dense[b_idx], y[b_idx])
            bootstrap_coefficients.append(lin_model.coef_.copy())

        boot_arr = np.array(bootstrap_coefficients)
        ci_l = np.percentile(boot_arr, (100.0 - ci) / 2.0, axis=0)
        ci_u = np.percentile(boot_arr, 100.0 - (100.0 - ci) / 2.0, axis=0)

        for idx, product in enumerate(products):
            res_dict["Product"].append(product)
            res_dict["Adverse Event"].append(column_name)
            res_dict["Count"].append(int(counts[idx]))
            res_dict["LASSO Coefficient"].append(float(all_coefs[idx]))
            res_dict["CI Lower"].append(float(ci_l[idx]))
            res_dict["CI Upper"].append(float(ci_u[idx]))
            
    return res_dict


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
    relaxed: bool = True,
    n_jobs: int = 1,
) -> AnalysisResult:
    """
    Applies LASSO regression or its variants to detect signals between product features and adverse events,
    optionally using bootstrap confidence intervals.
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
        "relaxed": relaxed,
        "n_jobs": n_jobs,
    }

    X = container.product_features
    ys = container.event_outcomes
    products = list(X.columns)

    if hasattr(X, "sparse") or issparse(X):
        if hasattr(X, "sparse"):
            X_mat = X.sparse.to_coo().tocsr()
        else:
            X_mat = X.tocsr()
    else:
        X_mat = np.ascontiguousarray(X.values, dtype=np.float64)

    if hasattr(container, "covariates") and container.covariates is not None:
        Z_mat = np.ascontiguousarray(container.covariates.values, dtype=np.float64)
    else:
        Z_mat = None

    n_samples = X_mat.shape[0]
    n_features = X_mat.shape[1]
    rng = np.random.default_rng(42)
    z_crit = float(stats.norm.ppf(1.0 - (100.0 - ci) / 200.0))
    res = defaultdict(list)

    if lasso_kwargs is None:
        lasso_kwargs = dict()

    lin_model = None
    if resolved_family == "linear":
        if use_IC:
            lin_model = LassoLarsIC(criterion=IC_criterion, **lasso_kwargs)
        elif use_lars:
            lin_model = LassoLars(alpha=alpha, **lasso_kwargs)
        else:
            lin_model = Lasso(alpha=alpha, **lasso_kwargs)

    if n_jobs != 1 and len(ys.columns) > 1:
        from joblib import Parallel, delayed

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_adverse_event)(
                column_name=column,
                y=_extract_y_vector(ys[column]),
                X_mat=X_mat,
                products=products,
                n_features=n_features,
                z_crit=z_crit,
                C=C,
                use_cv=use_cv,
                cv=cv,
                use_bootstrap=use_bootstrap,
                num_bootstrap=num_bootstrap,
                lasso_kwargs=lasso_kwargs,
                relaxed=relaxed,
                Z_mat=Z_mat,
                rng_seed=42 + i,
                min_events=min_events,
                resolved_family=resolved_family,
                lin_model=lin_model,
                nb_alpha=nb_alpha,
                lasso_alpha=lasso_alpha,
                ci=ci,
            )
            for i, column in enumerate(ys.columns)
        )

        for r_dict in results_list:
            for k, v in r_dict.items():
                res[k].extend(v)
    else:
        for i, column in enumerate(ys.columns):
            y = _extract_y_vector(ys[column])
            r_dict = _fit_single_adverse_event(
                column_name=column,
                y=y,
                X_mat=X_mat,
                products=products,
                n_features=n_features,
                z_crit=z_crit,
                C=C,
                use_cv=use_cv,
                cv=cv,
                use_bootstrap=use_bootstrap,
                num_bootstrap=num_bootstrap,
                lasso_kwargs=lasso_kwargs,
                relaxed=relaxed,
                Z_mat=Z_mat,
                rng_seed=42 + i,
                min_events=min_events,
                resolved_family=resolved_family,
                lin_model=lin_model,
                nb_alpha=nb_alpha,
                lasso_alpha=lasso_alpha,
                ci=ci,
            )
            for k, v in r_dict.items():
                res[k].extend(v)

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

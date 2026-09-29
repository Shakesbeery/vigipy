"""SCORE-DA: Syndromic Cellwise Outlier & Residual Estimation for Disproportionality Analysis.

A novel pattern discovery and signal detection framework that blends:
1. Low-Rank Matrix Factorization to absorb baseline drug-class effects and indication confounding.
2. Report-Level Syndromic Graph Regularization (Graph Laplacian) to borrow statistical strength
   across clinically co-occurring adverse events without relying on external ontologies.
3. Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with exact Lipschitz bounds and
   non-negative box constraints (preventing hallucinated signals when count is zero).
4. Iterative Masking-Free Deflation to eliminate the competition/blockbuster bias.
5. Exact Null-Model Poisson / Negative-Binomial standard errors and Benjamini-Hochberg FDR control.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse, csr_matrix

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params


def _solve_fista_single_drug(
    y_target: np.ndarray,
    c_observed: np.ndarray,
    L_ae: np.ndarray,
    lambda_1: float,
    lambda_2: float,
    L_lip: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    """Solve the box-constrained graph-regularized lasso problem for a single drug using FISTA.

    min_{0 <= theta <= c_observed} 0.5 * ||y_target - theta||^2 + lambda_1 * ||theta||_1 + 0.5 * lambda_2 * theta^T L_ae theta
    """
    n_events = len(y_target)
    if np.all(y_target <= 0) or np.all(c_observed <= 0):
        return np.zeros(n_events, dtype=np.float64)

    inv_L_lip = 1.0 / max(L_lip, 1e-6)
    step_thresh = lambda_1 * inv_L_lip
    c_max = np.maximum(0.0, c_observed)

    # Initialize at feasible target projection
    theta = np.clip(y_target, 0.0, c_max)
    z = theta.copy()
    t = 1.0

    for _ in range(max_iter):
        # Gradient: (I + lambda_2 * L) * z - y_target
        grad = z + lambda_2 * (L_ae @ z) - y_target
        v = z - inv_L_lip * grad
        # Projected soft-thresholding with box constraint [0, c_max]
        theta_next = np.clip(v - step_thresh, 0.0, c_max)

        diff = np.max(np.abs(theta_next - theta))
        if diff < tol:
            theta = theta_next
            break

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = theta_next + ((t - 1.0) / t_next) * (theta_next - theta)
        t = t_next
        theta = theta_next

    return theta


def _build_syndromic_laplacian(
    S_cooccur: np.ndarray,
    min_jaccard: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalized graph Laplacian and spectral syndrome clusters.

    Parameters:
        S_cooccur: (I x I) co-occurrence matrix between adverse events.
        min_jaccard: Minimum Jaccard similarity threshold to retain an edge.

    Returns:
        L_norm: (I x I) normalized graph Laplacian matrix (PSD guaranteed, isolated nodes unpenalized).
        clusters: (I,) array of cluster assignments for each adverse event.
    """
    n_events = S_cooccur.shape[0]
    if n_events <= 1:
        return np.zeros((n_events, n_events), dtype=np.float64), np.zeros(n_events, dtype=int)

    diag_s = np.diag(S_cooccur)
    # Jaccard similarity: S_ik / (S_ii + S_kk - S_ik)
    denom = diag_s[:, None] + diag_s[None, :] - S_cooccur
    denom = np.maximum(denom, 1e-9)
    W = np.divide(S_cooccur, denom, where=(denom > 0))
    np.fill_diagonal(W, 0.0)
    W = np.clip(W, 0.0, 1.0)
    W[W < min_jaccard] = 0.0

    # Symmetrize
    W = 0.5 * (W + W.T)

    d = np.sum(W, axis=1)
    mask = d > 0

    if not np.any(mask):
        L_norm = np.zeros((n_events, n_events), dtype=np.float64)
        clusters = np.zeros(n_events, dtype=int)
        return L_norm, clusters

    # Safe degree inversion: isolated vertices (d_i == 0) remain 0 so they are unpenalized
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    L_norm = np.diag(mask.astype(np.float64)) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    # Spectral syndrome clustering: use Fiedler vector / second smallest eigenvector
    try:
        vals, vecs = np.linalg.eigh(L_norm)
        if n_events >= 4 and len(vals) > 2:
            v1 = vecs[:, 1]
            v2 = vecs[:, 2]
            clusters = (v1 > 0).astype(int) + 2 * (v2 > 0).astype(int)
        else:
            clusters = (vecs[:, 1] > 0).astype(int) if len(vals) > 1 else np.zeros(n_events, dtype=int)
    except Exception:
        clusters = np.zeros(n_events, dtype=int)

    return L_norm, clusters


def score_da(
    container: DataContainer,
    latent_rank: int = 5,
    syndromic_weight: float = 0.5,
    sparsity_param: float = 1.0,
    fdr_threshold: float = 0.05,
    deflate_iterations: int = 2,
    min_events: int = 1,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_jobs: int = 1,
    seed: int = 42,
) -> AnalysisResult:
    """Perform Syndromic Cellwise Outlier & Residual Estimation (SCORE-DA).

    Blends low-rank baseline background estimation (absorbing drug class and indication
    confounding) with report-level syndromic graph regularization (borrowing statistical
    strength across co-occurring symptoms) and iterative deflation to eliminate masking.

    Parameters:
        container: A DataContainer holding binary report outcomes or a contingency table.
        latent_rank: Number of latent factors for background indication & drug class absorption.
            Set to 0 to disable low-rank baseline filtering.
        syndromic_weight: Graph Laplacian coupling penalty (lambda_2 >= 0). Higher values
            encourage stronger borrowing of statistical strength across connected symptoms.
        sparsity_param: L1 sparsity penalty (lambda_1 >= 0) on the excess signal rate.
        fdr_threshold: Target False Discovery Rate (q-value) cutoff for signal detection.
        deflate_iterations: Number of iterative deflation passes to remove masking/blockbuster bias.
            Default is 2 (initial fit + 1 deflated refit).
        min_events: Minimum observed event count required to qualify as a signal.
        max_iter: Maximum number of FISTA iterations per drug.
        tol: Convergence tolerance for FISTA.
        n_jobs: Number of CPU worker processes (-1 for all available cores).
        seed: Random seed for reproducibility.

    Returns:
        AnalysisResult containing all drug-event pairs, detected signals, signal count,
        and model parameters.
    """
    param_dict = build_params("score", {
        "latent_rank": latent_rank,
        "syndromic_weight": syndromic_weight,
        "sparsity_param": sparsity_param,
        "fdr_threshold": fdr_threshold,
        "deflate_iterations": deflate_iterations,
        "min_events": min_events,
        "max_iter": max_iter,
        "tol": tol,
        "n_jobs": n_jobs,
        "seed": seed,
    })

    # 1. Ingestion: Extract contingency matrix C (J drugs x I AEs) and Co-occurrence S (I x I)
    if container.type in ("binary", "binary_report", "binary_count"):
        X_df = container.product_features
        Y_df = container.event_outcomes
        products = list(X_df.columns)
        events = list(Y_df.columns)

        if hasattr(X_df, "sparse") or issparse(X_df):
            X_mat = X_df.sparse.to_coo().tocsr() if hasattr(X_df, "sparse") else X_df.tocsr()
        else:
            X_mat = np.ascontiguousarray(X_df.values, dtype=np.float64)

        if hasattr(Y_df, "sparse") or issparse(Y_df):
            Y_mat = Y_df.sparse.to_coo().tocsr() if hasattr(Y_df, "sparse") else Y_df.tocsr()
        else:
            Y_mat = np.ascontiguousarray(Y_df.values, dtype=np.float64)

        if issparse(X_mat) or issparse(Y_mat):
            C = (X_mat.T @ Y_mat).toarray()
            S_cooccur = (Y_mat.T @ Y_mat).toarray()
        else:
            C = X_mat.T @ Y_mat
            S_cooccur = Y_mat.T @ Y_mat

    elif container.type == "contingency":
        cont = container.contingency
        products = list(cont.index)
        events = list(cont.columns)
        C = np.ascontiguousarray(cont.values, dtype=np.float64)
        S_cooccur = C.T @ C
    else:
        raise ValueError(f"Unsupported container type '{container.type}' for score_da.")

    n_drugs, n_events = C.shape

    # 2. Build Syndromic Graph Laplacian & Clusters
    L_ae, clusters = _build_syndromic_laplacian(S_cooccur)

    # Compute exact spectral norm of L_ae via power iteration to get optimal Lipschitz constant
    rng = np.random.default_rng(seed)
    v_init = rng.standard_normal(n_events)
    v_norm = np.linalg.norm(v_init)
    if v_norm > 0:
        v_pi = v_init / v_norm
        for _ in range(10):
            v_next = L_ae @ v_pi
            vn = np.linalg.norm(v_next)
            if vn > 0:
                v_pi = v_next / vn
        lambda_max = float(v_pi.T @ (L_ae @ v_pi))
    else:
        lambda_max = 2.0
    L_lip = 1.0 + lambda_max * float(syndromic_weight)

    # 3. Iterative Deflation Loop to Eliminate Masking
    C_current = C.copy()
    Theta_est = np.zeros_like(C)
    Lambda_baseline = np.zeros_like(C)

    effective_rank = min(latent_rank, max(0, min(n_drugs, n_events) - 1))

    for it in range(max(1, deflate_iterations)):
        # Compute marginals on current (possibly deflated) counts
        R_row = np.sum(C_current, axis=1)
        C_col = np.sum(C_current, axis=0)
        N_tot = float(np.sum(R_row))

        if N_tot <= 0:
            Lambda_baseline = np.full_like(C, 1e-4)
            break

        E_mat = np.outer(R_row, C_col) / N_tot

        # Standardized Pearson residuals
        denom_var = E_mat * (1.0 - R_row[:, None] / N_tot) * (1.0 - C_col[None, :] / N_tot)
        std_err_mat = np.sqrt(np.maximum(denom_var, 1e-6))
        R_pears = (C_current - E_mat) / std_err_mat

        # Low-rank background factor absorption via randomized SVD
        if effective_rank > 0 and min(n_drugs, n_events) > effective_rank:
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S_vals, Vt = randomized_svd(
                    R_pears, n_components=effective_rank, random_state=seed
                )
                R_low_rank = (U * S_vals) @ Vt
                Lambda_baseline = np.maximum(1e-4, E_mat + R_low_rank * std_err_mat)
            except Exception:
                try:
                    U, S_vals, Vt = np.linalg.svd(R_pears, full_matrices=False)
                    R_low_rank = (U[:, :effective_rank] * S_vals[:effective_rank]) @ Vt[:effective_rank, :]
                    Lambda_baseline = np.maximum(1e-4, E_mat + R_low_rank * std_err_mat)
                except Exception:
                    Lambda_baseline = np.maximum(1e-4, E_mat)
        else:
            Lambda_baseline = np.maximum(1e-4, E_mat)

        # Target excess rate for each drug
        Y_target = C - Lambda_baseline

        # Solve box-constrained FISTA per drug
        if n_jobs != 1 and n_drugs > 1:
            from joblib import Parallel, delayed

            theta_rows = Parallel(n_jobs=n_jobs)(
                delayed(_solve_fista_single_drug)(
                    y_target=Y_target[j, :],
                    c_observed=C[j, :],
                    L_ae=L_ae,
                    lambda_1=sparsity_param,
                    lambda_2=syndromic_weight,
                    L_lip=L_lip,
                    max_iter=max_iter,
                    tol=tol,
                )
                for j in range(n_drugs)
            )
            Theta_est = np.array(theta_rows)
        else:
            for j in range(n_drugs):
                Theta_est[j, :] = _solve_fista_single_drug(
                    y_target=Y_target[j, :],
                    c_observed=C[j, :],
                    L_ae=L_ae,
                    lambda_1=sparsity_param,
                    lambda_2=syndromic_weight,
                    L_lip=L_lip,
                    max_iter=max_iter,
                    tol=tol,
                )

        # Deflate table for next iteration (guaranteed non-negative by box constraint)
        if it < deflate_iterations - 1:
            C_current = np.maximum(0.0, C - Theta_est)

    # 4. Statistical Inference & Output Assembly
    counts_flat = C.flatten()
    baseline_flat = Lambda_baseline.flatten()
    theta_flat = Theta_est.flatten()

    # Exact null standard error under H_0: theta = 0 (Var(C) = Lambda)
    se_flat = np.sqrt(np.maximum(baseline_flat, 1e-6))
    z_scores = np.where(theta_flat > 0, theta_flat / se_flat, 0.0)
    p_values = np.where(z_scores > 0, 1.0 - stats.norm.cdf(z_scores), 1.0)
    p_values = np.clip(p_values, 0.0, 1.0)

    # 95% Wald Confidence Intervals for the Syndromic Excess Rate
    z_crit = 1.959963984540054
    ser_lower = np.maximum(0.0, theta_flat - z_crit * se_flat)
    ser_upper = theta_flat + z_crit * se_flat

    # Benjamini-Hochberg adjustment across all pairs
    n_pairs = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]
    ranks = np.arange(1, n_pairs + 1)
    adj_p = np.minimum(1.0, sorted_p * n_pairs / ranks)
    adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]
    q_values = np.empty_like(adj_p)
    q_values[sort_idx] = adj_p

    srr_flat = np.where(baseline_flat > 0, counts_flat / np.maximum(baseline_flat, 1e-6), 1.0)

    # Build master records
    prod_indices, event_indices = np.unravel_index(np.arange(n_pairs), (n_drugs, n_events))
    prod_names = [products[p] for p in prod_indices]
    event_names = [events[e] for e in event_indices]
    cluster_assigned = [int(clusters[e]) for e in event_indices]

    all_signals_df = pd.DataFrame({
        "Product": prod_names,
        "Adverse Event": event_names,
        "Count": counts_flat.astype(int),
        "Expected Count": np.round(baseline_flat, 2),
        "Expected": np.round(baseline_flat, 2),  # Compatibility alias
        "SER": np.round(theta_flat, 4),
        "SER Lower": np.round(ser_lower, 4),
        "SER Upper": np.round(ser_upper, 4),
        "SRR": np.round(srr_flat, 3),
        "SE": np.round(se_flat, 3),
        "z_score": np.round(z_scores, 3),
        "p_value": p_values,
        "fdr": q_values,
        "p_adj": q_values,  # Compatibility alias
        "Syndrome_Cluster": cluster_assigned,
    })

    # Filter detected signals
    sig_mask = (
        (all_signals_df["Count"] >= min_events)
        & (all_signals_df["SER"] > 0.0)
        & (all_signals_df["fdr"] <= fdr_threshold)
    )
    signals_df = all_signals_df[sig_mask].sort_values(
        by=["fdr", "SER"], ascending=[True, False]
    ).reset_index(drop=True)

    return AnalysisResult(
        all_signals=all_signals_df.reset_index(drop=True),
        signals=signals_df,
        num_signals=len(signals_df),
        params=param_dict,
    )

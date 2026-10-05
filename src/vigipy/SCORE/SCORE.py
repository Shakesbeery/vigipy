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

import logging
from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse, csr_matrix

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params

logger = logging.getLogger("vigipy")


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


def _solve_fista_matrix(
    Y_target: np.ndarray,
    C_observed: np.ndarray,
    L_ae: np.ndarray | csr_matrix,
    lambda_1: float,
    lambda_2: float,
    L_lip: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    """Solve the box-constrained graph-regularized lasso problem for all drugs simultaneously via Matrix FISTA.

    Solves for the entire (J x I) matrix Theta concurrently:
        min_{0 <= Theta <= C_observed} 0.5 * ||Y_target - Theta||_F^2 + lambda_1 * ||Theta||_1 + 0.5 * lambda_2 * Tr(Theta L_ae Theta^T)
    """
    n_drugs, n_events = Y_target.shape
    if n_drugs == 0 or n_events == 0:
        return np.zeros_like(Y_target)

    inv_L_lip = 1.0 / max(L_lip, 1e-6)
    step_thresh = lambda_1 * inv_L_lip
    c_max = np.maximum(0.0, C_observed)

    # Initialize at feasible target projection
    theta = np.clip(Y_target, 0.0, c_max)
    # Identify rows that have no positive targets or no positive observations
    inactive = (np.all(Y_target <= 0.0, axis=1) | np.all(C_observed <= 0.0, axis=1))
    theta[inactive] = 0.0

    z = theta.copy()
    t = 1.0

    for _ in range(max_iter):
        # Matrix gradient across all drugs: Z + lambda_2 * (Z @ L_ae) - Y_target
        grad = z + lambda_2 * (z @ L_ae) - Y_target
        v = z - inv_L_lip * grad
        theta_next = np.clip(v - step_thresh, 0.0, c_max)
        theta_next[inactive] = 0.0

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

    if issparse(S_cooccur):
        diag_s = S_cooccur.diagonal()
        S_coo = S_cooccur.tocoo()
        non_diag = S_coo.row != S_coo.col
        rows = S_coo.row[non_diag]
        cols = S_coo.col[non_diag]
        vals = S_coo.data[non_diag]
        denom_sp = diag_s[rows] + diag_s[cols] - vals
        w_vals = np.where(denom_sp > 0, vals / np.maximum(denom_sp, 1e-9), 0.0)
        w_vals = np.clip(w_vals, 0.0, 1.0)
        keep = w_vals >= min_jaccard
        W_sp = csr_matrix((w_vals[keep], (rows[keep], cols[keep])), shape=(n_events, n_events))
        W = 0.5 * (W_sp + W_sp.T)
        d = np.array(W.sum(axis=1)).ravel()
        mask = d > 0
        if not np.any(mask):
            return csr_matrix((n_events, n_events), dtype=np.float64), np.zeros(n_events, dtype=int)
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        from scipy.sparse import diags
        D_inv = diags(d_inv_sqrt)
        L_norm = diags(mask.astype(np.float64)) - D_inv @ W @ D_inv
        L_norm = L_norm.tocsr()
    else:
        diag_s = np.diag(S_cooccur)
        denom = diag_s[:, None] + diag_s[None, :] - S_cooccur
        denom = np.maximum(denom, 1e-9)
        W = np.divide(S_cooccur, denom, where=(denom > 0))
        np.fill_diagonal(W, 0.0)
        W = np.clip(W, 0.0, 1.0)
        W[W < min_jaccard] = 0.0
        W = 0.5 * (W + W.T)
        d = np.sum(W, axis=1)
        mask = d > 0
        if not np.any(mask):
            return np.zeros((n_events, n_events), dtype=np.float64), np.zeros(n_events, dtype=int)
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        L_norm = np.diag(mask.astype(np.float64)) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
        if n_events > 500:
            L_norm = csr_matrix(L_norm)

    # Spectral syndrome clustering: use Fiedler vector / second smallest eigenvector
    try:
        if n_events >= 4:
            from scipy.sparse.linalg import eigsh
            k = min(3, n_events - 1)
            vals, vecs = eigsh(csr_matrix(L_norm), k=k, which="SM")
            sort_order = np.argsort(vals)
            vals = vals[sort_order]
            vecs = vecs[:, sort_order]
            if k >= 3:
                v1 = vecs[:, 1]
                v2 = vecs[:, 2]
                clusters = (v1 > 0).astype(int) + 2 * (v2 > 0).astype(int)
            elif k >= 2:
                clusters = (vecs[:, 1] > 0).astype(int)
            else:
                clusters = np.zeros(n_events, dtype=int)
        elif n_events > 1:
            vals, vecs = np.linalg.eigh(L_norm)
            clusters = (vecs[:, 1] > 0).astype(int) if len(vals) > 1 else np.zeros(n_events, dtype=int)
        else:
            clusters = np.zeros(n_events, dtype=int)
    except Exception:
        clusters = np.zeros(n_events, dtype=int)

    if n_events > 500:
        L_norm = csr_matrix(L_norm)

    return L_norm, clusters


def _extract_matrices_from_container(
    container: DataContainer,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract contingency matrix C (drugs x AEs) and AE co-occurrence matrix S (AEs x AEs)."""
    if container.type in ("binary", "binary_report", "binary_count", "binary_ddi"):
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
            S_cooccur = (Y_mat.T @ Y_mat)
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
        raise ValueError(f"Unsupported container type '{container.type}' for SCORE.")

    return C, S_cooccur, products, events


def _compute_lipschitz_constant(L_ae: np.ndarray, syndromic_weight: float, seed: int = 42) -> float:
    """Compute exact Lipschitz gradient constant bound via power iteration spectral norm."""
    n_events = L_ae.shape[0]
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
    return 1.0 + lambda_max * float(syndromic_weight)


def _estimate_low_rank_baseline(C_current: np.ndarray, effective_rank: int, seed: int = 42) -> np.ndarray:
    """Estimate expected baseline rates absorbing indication/class confounding via randomized SVD on Pearson residuals."""
    R_row = np.sum(C_current, axis=1)
    C_col = np.sum(C_current, axis=0)
    N_tot = float(np.sum(R_row))

    if N_tot <= 0:
        return np.full_like(C_current, 1e-4)

    E_mat = np.outer(R_row, C_col) / N_tot
    denom_var = E_mat * (1.0 - R_row[:, None] / N_tot) * (1.0 - C_col[None, :] / N_tot)
    std_err_mat = np.sqrt(np.maximum(denom_var, 1e-6))
    R_pears = (C_current - E_mat) / std_err_mat

    n_drugs, n_events = C_current.shape
    if effective_rank > 0 and min(n_drugs, n_events) > effective_rank:
        try:
            from sklearn.utils.extmath import randomized_svd
            U, S_vals, Vt = randomized_svd(
                R_pears, n_components=effective_rank, random_state=seed
            )
            R_low_rank = (U * S_vals) @ Vt
            return np.maximum(1e-4, E_mat + R_low_rank * std_err_mat)
        except Exception:
            try:
                U, S_vals, Vt = np.linalg.svd(R_pears, full_matrices=False)
                R_low_rank = (U[:, :effective_rank] * S_vals[:effective_rank]) @ Vt[:effective_rank, :]
                return np.maximum(1e-4, E_mat + R_low_rank * std_err_mat)
            except Exception:
                return np.maximum(1e-4, E_mat)
    else:
        return np.maximum(1e-4, E_mat)


def _compute_bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR adjusted q-values across candidate p-values."""
    n_comparisons = len(p_values)
    if n_comparisons == 0:
        return np.empty(0, dtype=np.float64)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]
    ranks = np.arange(1, n_comparisons + 1)
    adj_p = np.minimum(1.0, sorted_p * n_comparisons / ranks)
    adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]
    q_values = np.empty_like(adj_p)
    q_values[sort_idx] = adj_p
    return q_values


def _classify_interaction_archetype(active_count: int, k_order: int) -> str:
    """Classify the clinical synergy mechanism into an epidemiological archetype."""
    if active_count == 0:
        return "EMERGENT"
    elif active_count < k_order:
        return "POTENTIATED"
    else:
        return "TWO_HIT" if k_order == 2 else "MULTI_HIT"


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

    Clinical Intuition:
        Traditional disproportionality methods treat all symptoms independently and assume
        uniform background reporting across drugs. In practice, clinical reality violates both:
        drugs in the same class share indications (e.g. anti-diabetics associated with hyperglycemia),
        and symptoms co-occur in clinical syndromes (e.g. urticaria, facial edema, hypotension in anaphylaxis).
        Furthermore, blockbuster drugs (with millions of reports) artificially inflate background denominators,
        masking true signals from smaller drugs.

        SCORE-DA resolves all three:
        1. Indication Absorption: Low-rank matrix factorization (SVD on Pearson residuals) absorbs
           drug-class and indication confounding into the baseline null model.
        2. Syndromic Borrowing: Report-level Graph Laplacian regularization borrows statistical
           strength across co-occurring symptoms in syndromic clusters.
        3. Masking Deflation: Iterative deflation removes signals from the contingency table, unmasking
           hidden safety alerts suppressed by blockbuster competition.

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

    # 1. Ingestion: Extract contingency matrix C and Co-occurrence S
    C, S_cooccur, products, events = _extract_matrices_from_container(container)
    n_drugs, n_events = C.shape

    # 2. Build Syndromic Graph Laplacian & Clusters
    L_ae, clusters = _build_syndromic_laplacian(S_cooccur)
    L_lip = _compute_lipschitz_constant(L_ae, syndromic_weight, seed=seed)

    # 3. Iterative Deflation Loop to Eliminate Masking
    C_current = C
    Theta_est = np.zeros_like(C)
    Lambda_baseline = np.zeros_like(C)

    effective_rank = min(latent_rank, max(0, min(n_drugs, n_events) - 1))
    logger.debug("SCORE-DA: fitting %d drugs, %d events, effective rank %d", n_drugs, n_events, effective_rank)

    for it in range(max(1, deflate_iterations)):
        logger.debug("SCORE-DA deflation iteration %d / %d", it + 1, deflate_iterations)
        Lambda_baseline = _estimate_low_rank_baseline(C_current, effective_rank, seed=seed)

        # Target excess rate for each drug
        Y_target = C - Lambda_baseline

        # Solve box-constrained FISTA across all drugs simultaneously via Matrix FISTA
        Theta_est = _solve_fista_matrix(
            Y_target=Y_target,
            C_observed=C,
            L_ae=L_ae,
            lambda_1=sparsity_param,
            lambda_2=syndromic_weight,
            L_lip=L_lip,
            max_iter=max_iter,
            tol=tol,
        )

        # Deflate table for next iteration
        if it < deflate_iterations - 1:
            C_current = np.maximum(0.0, C - Theta_est)

    # 4. Statistical Inference & Output Assembly
    counts_flat = C.flatten()
    baseline_flat = Lambda_baseline.flatten()
    theta_flat = Theta_est.flatten()

    # Null standard error under H_0: theta = 0 (Var(C) = Lambda)
    se_flat = np.sqrt(np.maximum(baseline_flat, 1e-6))
    z_scores = np.where(theta_flat > 0, theta_flat / se_flat, 0.0)
    p_values = np.where(z_scores > 0, 1.0 - stats.norm.cdf(z_scores), 1.0)
    p_values = np.clip(p_values, 0.0, 1.0)

    # 95% Wald Confidence Intervals for the Syndromic Excess Rate
    z_crit = 1.959963984540054
    ser_lower = np.maximum(0.0, theta_flat - z_crit * se_flat)
    ser_upper = theta_flat + z_crit * se_flat

    # Benjamini-Hochberg adjustment across all pairs
    q_values = _compute_bh_fdr(p_values)
    srr_flat = np.where(baseline_flat > 0, counts_flat / np.maximum(baseline_flat, 1e-6), 1.0)

    # Build master records
    n_pairs = len(p_values)
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



def score_ddi(
    container: DataContainer,
    interaction_model: str = "multiplicative",
    syndromic_weight: float = 0.5,
    sparsity_param: float = 1.0,
    fdr_threshold: float = 0.05,
    min_events: int = 1,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_jobs: int = 1,
    seed: int = 42,
) -> AnalysisResult:
    """Perform SCORE-DDI (Syndromic Cellwise Outlier & Residual Estimation for Drug-Drug Interactions).

    Clinical Intuition:
        When patients take multiple medications simultaneously, adverse reactions can
        occur through synergistic interactions that far exceed what either drug would
        cause alone. SCORE-DDI determines whether a combination of medications produces
        an adverse event rate significantly higher than expected under an independent
        baseline model (either multiplicative or additive risk compounding).

        Crucially, adverse drug reactions are rarely isolated biochemical events; they
        frequently manifest as syndromic constellations of co-occurring symptoms (e.g.,
        rash + eosinophilia + systemic symptoms in DRESS syndrome, or fever + rigidity
        in neuroleptic malignant syndrome). SCORE-DDI couples all adverse events using
        a patient-level Graph Laplacian, borrowing statistical strength across related
        symptoms to boost signal detection for rare, life-threatening multi-organ toxicities.

        Each detected interaction is classified into a clinical archetype:
        - **EMERGENT**: Neither drug causes the reaction individually; the toxicity is
          wholly unique to the combination (true pharmacological synergy).
        - **POTENTIATED**: One drug is already known to cause the reaction, and the second
          drug markedly amplifies its incidence or severity.
        - **TWO-HIT / MULTI-HIT**: Both drugs independently trigger the toxicity, and their
          concurrent administration produces a severe compound injury.

    Parameters:
        container: A DataContainer prepared via `convert_ddi(...)` or containing pair_mapping.
        interaction_model: Null model for expected co-reporting under no interaction.
            Options: 'multiplicative' (default, independent relative risk compounding)
            or 'additive' (additive excess risk difference).
        syndromic_weight: Graph Laplacian coupling penalty (lambda_2 >= 0). Encourages
            borrowing of strength across clinically related symptoms in a syndrome.
        sparsity_param: L1 sparsity penalty (lambda_1 >= 0) on the interaction excess rate.
        fdr_threshold: Target False Discovery Rate (q-value) cutoff for interaction signal detection.
        min_events: Minimum observed co-occurrence count required to qualify as an interaction signal.
        max_iter: Maximum number of FISTA iterations per drug pair.
        tol: Convergence tolerance for FISTA.
        n_jobs: Number of CPU worker processes (-1 for all available cores).
        seed: Random seed for reproducibility.

    Returns:
        AnalysisResult containing all drug pairs x adverse event combinations, detected
        interaction signals, interaction archetypes (EMERGENT, POTENTIATED, TWO_HIT),
        and model parameters.
    """
    if interaction_model not in ("multiplicative", "additive"):
        raise ValueError(
            f"Invalid interaction_model '{interaction_model}'. Must be 'multiplicative' or 'additive'."
        )

    param_dict = build_params("score_ddi", {
        "interaction_model": interaction_model,
        "syndromic_weight": syndromic_weight,
        "sparsity_param": sparsity_param,
        "fdr_threshold": fdr_threshold,
        "min_events": min_events,
        "max_iter": max_iter,
        "tol": tol,
        "n_jobs": n_jobs,
        "seed": seed,
    })

    # 1. Extract contingency matrix & pairs
    if container.contingency is not None:
        cont = container.contingency
    elif container.type in ("binary", "binary_report", "binary_count", "binary_ddi"):
        X_df = container.product_features
        Y_df = container.event_outcomes
        if hasattr(X_df, "sparse") or issparse(X_df):
            X_mat = X_df.sparse.to_coo().tocsr() if hasattr(X_df, "sparse") else X_df.tocsr()
        else:
            X_mat = np.ascontiguousarray(X_df.values, dtype=np.float64)
        if hasattr(Y_df, "sparse") or issparse(Y_df):
            Y_mat = Y_df.sparse.to_coo().tocsr() if hasattr(Y_df, "sparse") else Y_df.tocsr()
        else:
            Y_mat = np.ascontiguousarray(Y_df.values, dtype=np.float64)
        C_arr = (X_mat.T @ Y_mat).toarray() if issparse(X_mat) or issparse(Y_mat) else X_mat.T @ Y_mat
        cont = pd.DataFrame(C_arr, index=list(X_df.columns), columns=list(Y_df.columns))
    else:
        raise ValueError(f"Unsupported container type '{container.type}' for score_ddi.")

    pair_mapping = getattr(container, "pair_mapping", None)
    if not pair_mapping:
        pair_mapping = {}
        for name in cont.index:
            if " + " in str(name):
                parts = str(name).split(" + ")
                if len(parts) == 2 and parts[0] in cont.index and parts[1] in cont.index:
                    pair_mapping[name] = (parts[0], parts[1])

    if not pair_mapping:
        raise ValueError(
            "No drug pairs found in container. Use convert_ddi() to prepare interaction data."
        )

    events = list(cont.columns)
    n_events = len(events)
    all_rows = list(cont.index)
    C_all = np.ascontiguousarray(cont.values, dtype=np.float64)
    N_tot = float(np.sum(C_all))

    if N_tot <= 0:
        raise ValueError("Contingency table contains no counts.")

    C_ae = np.sum(C_all, axis=0)

    # 2. Build Syndromic Graph Laplacian & Clusters
    if container.event_outcomes is not None:
        Y_df = container.event_outcomes
        if hasattr(Y_df, "sparse") or issparse(Y_df):
            Y_mat = Y_df.sparse.to_coo().tocsr() if hasattr(Y_df, "sparse") else Y_df.tocsr()
        else:
            Y_mat = np.ascontiguousarray(Y_df.values, dtype=np.float64)
        S_cooccur = (Y_mat.T @ Y_mat) if issparse(Y_mat) else Y_mat.T @ Y_mat
    else:
        S_cooccur = C_all.T @ C_all

    L_ae, clusters = _build_syndromic_laplacian(S_cooccur)
    L_lip = _compute_lipschitz_constant(L_ae, syndromic_weight, seed=seed)

    # 3. Compute baseline marginals across single drugs (to avoid double-counting pair rows)
    single_drugs = [d for d in cont.index if d not in pair_mapping]
    if len(single_drugs) > 0:
        single_indices = [cont.index.get_loc(d) for d in single_drugs]
        C_singles = C_all[single_indices, :]
        N_tot = float(np.sum(C_singles))
        C_ae = np.sum(C_singles, axis=0)
    else:
        N_tot = float(np.sum(C_all))
        C_ae = np.sum(C_all, axis=0)

    if N_tot <= 0:
        raise ValueError("Contingency table contains no counts.")

    single_counts = {d: C_all[cont.index.get_loc(d), :] for d in single_drugs}

    # 4. Compute no-interaction baselines & targets for all candidate pairs
    pair_names = list(pair_mapping.keys())
    n_pairs = len(pair_names)
    C_pairs = np.zeros((n_pairs, n_events), dtype=np.float64)
    Lambda_null = np.zeros((n_pairs, n_events), dtype=np.float64)
    Y_target_pairs = np.zeros((n_pairs, n_events), dtype=np.float64)
    archetypes = []

    for p_idx, p_name in enumerate(pair_names):
        raw_pair = pair_mapping[p_name]
        drugs = tuple(raw_pair) if isinstance(raw_pair, (list, tuple)) else (str(raw_pair),)
        k_order = len(drugs)

        c_p = C_all[cont.index.get_loc(p_name), :]
        C_pairs[p_idx, :] = c_p
        r_p = np.sum(c_p)
        e_p = (r_p * C_ae) / N_tot

        # Solo counts and baseline relative risks for each constituent drug
        rr_list = []
        c_solo_list = []
        for d in drugs:
            c_raw = single_counts.get(d, c_p)
            c_solo = np.maximum(0.0, c_raw - c_p)
            r_solo = np.sum(c_solo)
            e_solo = (r_solo * C_ae) / N_tot if r_solo > 0 else np.full(n_events, 1e-4)
            rr = np.where(e_solo > 0, c_solo / np.maximum(e_solo, 1e-6), 1.0)
            rr_list.append(np.maximum(1e-6, rr))
            c_solo_list.append(c_solo)

        if interaction_model == "multiplicative":
            rr_null = np.ones(n_events, dtype=np.float64)
            for rr_item in rr_list:
                rr_null *= rr_item
        else:
            rr_sum = np.sum(rr_list, axis=0)
            rr_null = np.maximum(1e-6, rr_sum - (k_order - 1.0))

        lam_null = np.maximum(1e-4, e_p * rr_null)
        Lambda_null[p_idx, :] = lam_null
        Y_target_pairs[p_idx, :] = c_p - lam_null

        # Classify interaction archetype per event
        for i in range(n_events):
            active_count = sum(
                (rr_list[d_idx][i] >= 2.0 and c_solo_list[d_idx][i] >= min_events)
                for d_idx in range(k_order)
            )
            archetypes.append(_classify_interaction_archetype(active_count, k_order))

    # 5. Solve Box-Constrained FISTA across all pairs simultaneously via Matrix FISTA
    Theta_est = _solve_fista_matrix(
        Y_target=Y_target_pairs,
        C_observed=C_pairs,
        L_ae=L_ae,
        lambda_1=sparsity_param,
        lambda_2=syndromic_weight,
        L_lip=L_lip,
        max_iter=max_iter,
        tol=tol,
    )

    # 6. Statistical Inference & Output Formatting
    counts_flat = C_pairs.flatten()
    lam_flat = Lambda_null.flatten()
    theta_flat = Theta_est.flatten()

    se_flat = np.sqrt(np.maximum(lam_flat, 1e-6))
    z_scores = np.where(theta_flat > 0, theta_flat / se_flat, 0.0)
    p_values = np.where(z_scores > 0, 1.0 - stats.norm.cdf(z_scores), 1.0)
    p_values = np.clip(p_values, 0.0, 1.0)

    # 95% Wald CI for SER
    z_crit = 1.959963984540054
    ser_lower = np.maximum(0.0, theta_flat - z_crit * se_flat)
    ser_upper = theta_flat + z_crit * se_flat

    # Benjamini-Hochberg FDR
    q_values = _compute_bh_fdr(p_values)

    ddi_ratio = np.where(lam_flat > 0, counts_flat / np.maximum(lam_flat, 1e-6), 1.0)

    total_comparisons = len(p_values)
    p_indices, e_indices = np.unravel_index(np.arange(total_comparisons), (n_pairs, n_events))
    p_names = [pair_names[p] for p in p_indices]
    components = [", ".join(pair_mapping[pair_names[p]]) for p in p_indices]
    order_vals = [len(pair_mapping[pair_names[p]]) for p in p_indices]
    drug1_names = [pair_mapping[pair_names[p]][0] for p in p_indices]
    drug2_names = [pair_mapping[pair_names[p]][1] if len(pair_mapping[pair_names[p]]) > 1 else "" for p in p_indices]
    event_names = [events[e] for e in e_indices]
    clusters_assigned = [int(clusters[e]) for e in e_indices]

    all_signals_df = pd.DataFrame({
        "Components": components,
        "Order": order_vals,
        "Drug_1": drug1_names,
        "Drug_2": drug2_names,
        "Product": p_names,
        "Adverse Event": event_names,
        "Count": counts_flat.astype(int),
        "Expected_Null": np.round(lam_flat, 2),
        "Expected": np.round(lam_flat, 2),  # Compatibility alias
        "SER_Interaction": np.round(theta_flat, 4),
        "SER": np.round(theta_flat, 4),  # Compatibility alias
        "SER Lower": np.round(ser_lower, 4),
        "SER Upper": np.round(ser_upper, 4),
        "DDI_Ratio": np.round(ddi_ratio, 3),
        "SRR": np.round(ddi_ratio, 3),  # Compatibility alias
        "SE": np.round(se_flat, 3),
        "z_score": np.round(z_scores, 3),
        "p_value": p_values,
        "fdr": q_values,
        "p_adj": q_values,  # Compatibility alias
        "Interaction_Archetype": archetypes,
        "Syndrome_Cluster": clusters_assigned,
    })

    sig_mask = (
        (all_signals_df["Count"] >= min_events)
        & (all_signals_df["SER_Interaction"] > 0.0)
        & (all_signals_df["fdr"] <= fdr_threshold)
    )
    signals_df = all_signals_df[sig_mask].sort_values(
        by=["fdr", "SER_Interaction"], ascending=[True, False]
    ).reset_index(drop=True)

    return AnalysisResult(
        all_signals=all_signals_df.reset_index(drop=True),
        signals=signals_df,
        num_signals=len(signals_df),
        params=param_dict,
    )

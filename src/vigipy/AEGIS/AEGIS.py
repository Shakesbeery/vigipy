"""AEGIS: Anytime Empirical Graph-Informed Shrinkage for Disproportionality Analysis.

Implements a pharmacovigilance signal detection model that addresses four structural
failure modes of classical disproportionality analysis:

1. Counterfactual baseline estimation via leave-one-cell-out (LOCO) initialization and
   bias-corrected Huber Winsorization (with optional low-rank SVD residual filtering)
   to prevent blockbuster competition masking and self-cloaking.
2. Positive Normalized Pointwise Mutual Information (NPMI) symptom graph construction
   with conjugate Poisson-Gamma neighborhood pooling and covariate-gated empirical Bayes
   shrinkage.
3. Exposure-normalized Dual-EWMA surge velocity to separate drug-wide stimulated
   reporting (media notoriety) from focal biological acceleration.
4. Anytime-valid Wald-Poisson mixture martingale e-values with cross-graph product
   martingale coupling (Vovk & Wang, 2021) for sequential surveillance.
"""

from __future__ import annotations

import logging
from typing import Literal
import numpy as np
import pandas as pd
from scipy.special import digamma, gdtr, gammaln, expit, logit, ndtr
from scipy.optimize import minimize
from scipy.sparse import issparse, csr_matrix

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params
from ..utils.distribution_funcs.quantile_funcs import quantiles

logger = logging.getLogger("vigipy")

# Calibrated fallback priors: (alpha1, beta1, alpha2, beta2, pi_base)
# Component 1 (null): mean = 0.85, rate = 3.0 -> shape = 2.55
# Component 2 (signal): mean = 3.20, rate = 0.8 -> shape = 2.56
_DEFAULT_PRIOR_PARAMS: tuple[float, float, float, float, float] = (2.55, 3.00, 2.56, 0.80, 0.12)

# Grid of alternative relative risks and prior weights for the Wald-Poisson mixture martingale
_MARTINGALE_GRID_RR = np.array([1.3, 1.6, 2.0, 2.8, 4.0, 6.0], dtype=np.float64)
_MARTINGALE_GRID_WEIGHTS = np.array([0.20, 0.25, 0.25, 0.15, 0.10, 0.05], dtype=np.float64)

# Thresholds for context gating and graph regularization
_DEGREE_REGULARIZER = 0.03
_COHERENCE_GATE_MIN = 0.25
_VELOCITY_GATE_MIN = 0.45
_ISOLATED_LOGIT_OFFSET = -0.45
_MAX_NEIGHBOR_E_CAP = 12.0


def _normal_upper_clip_bias(delta: float) -> float:
    """Analytical downward bias E[Z - min(Z, delta)] = phi(delta) - delta * Phi(-delta) for Z ~ N(0, 1)."""
    d = max(float(delta), 0.1)
    phi_d = np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
    sf_d = float(ndtr(-d))
    return float(max(0.0, phi_d - d * sf_d))


def _extract_aegis_matrices(
    container: DataContainer,
    fast_window_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float, list[str], list[str], np.ndarray | None]:
    """Extract contingency matrix C (J x I), AE co-occurrence matrix S (I x I), total reports N_rep,
    and exponentially weighted fast-window contingency matrix C_fast (J x I).
    """
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

        n_rep = float(X_mat.shape[0])
        if issparse(X_mat) or issparse(Y_mat):
            C_raw = X_mat.T @ Y_mat
            C = C_raw.toarray() if issparse(C_raw) else np.asarray(C_raw, dtype=np.float64)
            S_raw = Y_mat.T @ Y_mat
            S_cooccur = S_raw.toarray() if issparse(S_raw) else np.asarray(S_raw, dtype=np.float64)
        else:
            C = X_mat.T @ Y_mat
            S_cooccur = Y_mat.T @ Y_mat

        # Compute exponentially weighted fast-window counts C_fast = X^T diag(w_fast) Y
        n_rows = X_mat.shape[0]
        if n_rows >= 20 and 0.05 <= fast_window_fraction < 1.0:
            u = np.linspace(1.0 / n_rows, 1.0, n_rows, dtype=np.float64)
            # Exponential kernel concentrated on recent reports, normalized to target window mass
            tau = max(float(fast_window_fraction) * 0.75, 0.05)
            w_raw = np.exp(-(1.0 - u) / tau)
            w_fast = w_raw * (float(fast_window_fraction) * n_rows / max(float(np.sum(w_raw)), 1e-12))

            if issparse(X_mat) or issparse(Y_mat):
                X_weighted = csr_matrix(X_mat).multiply(w_fast[:, None])
                C_fast_raw = X_weighted.T @ csr_matrix(Y_mat)
                C_fast = C_fast_raw.toarray() if issparse(C_fast_raw) else np.asarray(C_fast_raw, dtype=np.float64)
            else:
                C_fast = (X_mat * w_fast[:, None]).T @ Y_mat
        else:
            C_fast = None

    elif container.contingency is not None:
        cont = container.contingency
        products = list(cont.index)
        events = list(cont.columns)
        C = np.ascontiguousarray(cont.values, dtype=np.float64)
        n_rep = float(max(container.N, np.sum(C), 1.0))

        row_sums = np.maximum(np.sum(C, axis=1, keepdims=True), 1.0)
        P_cond = C / row_sums
        col_sums = np.sum(C, axis=0)
        S_cooccur = (P_cond.T * col_sums[:, None]) @ P_cond
        np.fill_diagonal(S_cooccur, col_sums)

        C_fast = getattr(container, "fast_contingency", None)
        if C_fast is not None and isinstance(C_fast, pd.DataFrame):
            C_fast = np.ascontiguousarray(
                C_fast.reindex(index=products, columns=events, fill_value=0.0).values,
                dtype=np.float64,
            )
    else:
        raise ValueError(f"Unsupported container type '{container.type}' for AEGIS.")

    return C, S_cooccur, n_rep, products, events, C_fast


def _sinkhorn_huber_baseline(
    C: np.ndarray,
    huber_delta: float = 2.0,
    sinkhorn_iters: int = 5,
    latent_rank: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate counterfactual baseline expectations via LOCO initialization and
    bias-corrected Huber Winsorization, with optional low-rank SVD residual filtering.

    Returns:
        B_counterfactual: (J x I) unmasked counterfactual baseline expectation matrix.
        E_naive: (J x I) standard marginal expectation matrix R_j * K_i / N.
    """
    r_naive = np.sum(C, axis=1)
    c_naive = np.sum(C, axis=0)
    n_tot_naive = float(np.sum(r_naive))
    if n_tot_naive <= 0:
        B_min = np.full_like(C, 1e-4)
        return B_min, B_min

    E_naive = np.maximum(np.outer(r_naive, c_naive) / n_tot_naive, 1e-6)

    # Leave-one-cell-out (LOCO) marginal initialization breaks single-cell self-cloaking in O(1)
    r_loco = np.maximum(r_naive[:, None] - C, 0.0)
    c_loco = np.maximum(c_naive[None, :] - C, 0.0)
    n_loco = np.maximum(n_tot_naive - r_naive[:, None] - c_naive[None, :] + C, 1.0)
    B_loco = (r_loco * c_loco) / n_loco
    B = np.where((r_loco > 0) & (c_loco > 0), np.minimum(E_naive, np.maximum(B_loco, 1e-6)), E_naive)

    # Analytical Gaussian tail bias correction restores null unbiasedness under upper Winsorization
    bias_corr = _normal_upper_clip_bias(huber_delta)

    for _ in range(max(1, sinkhorn_iters)):
        std_b = np.sqrt(B + 1.0)
        cap = B + huber_delta * std_b
        excess = np.maximum(0.0, C - B)
        col_share = excess / np.maximum(c_naive[None, :], 1.0)
        row_share = excess / np.maximum(r_naive[:, None], 1.0)
        hijack_mask = (C > 2.0 * B) & (C > cap) & ((col_share > 0.15) | ((row_share > 0.50) & (C > 5.0 * B)))
        C_winsor = np.where(hijack_mask, cap + bias_corr * std_b, C)
        r_cf = np.sum(C_winsor, axis=1)
        c_cf = np.sum(C_winsor, axis=0)
        n_cf = float(np.sum(r_cf))
        if n_cf <= 0:
            break
        B = np.maximum(np.outer(r_cf, c_cf) / n_cf, 1e-6)

    # Optional low-rank SVD absorption of positive diffuse drug-class/indication structure
    n_drugs, n_events = C.shape
    eff_rank = min(int(latent_rank), max(0, min(n_drugs, n_events) - 1))
    if eff_rank > 0:
        std_mat = np.sqrt(B + 1.0)
        cap = B + huber_delta * std_mat
        C_winsor = np.minimum(C, cap)
        R_win = np.clip((C_winsor - B) / std_mat, -huber_delta, huber_delta)
        try:
            from sklearn.utils.extmath import randomized_svd

            U, s_vals, Vt = randomized_svd(R_win, n_components=eff_rank, random_state=seed)
            R_lr = np.maximum(0.0, (U * s_vals) @ Vt)
            B = np.maximum(1e-4, B + R_lr * std_mat)
        except Exception as exc:
            logger.debug("randomized_svd fallback in _sinkhorn_huber_baseline: %s", exc)
            U, s_vals, Vt = np.linalg.svd(R_win, full_matrices=False)
            R_lr = np.maximum(0.0, (U[:, :eff_rank] * s_vals[:eff_rank]) @ Vt[:eff_rank, :])
            B = np.maximum(1e-4, B + R_lr * std_mat)

    return np.maximum(B, 1e-5), E_naive


def _build_npmi_syndrome_graph(
    S_cooccur: np.ndarray,
    n_rep: float,
    min_npmi: float = 0.05,
    top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a top-K pruned positive Normalized Pointwise Mutual Information (NPMI) graph.

    Returns:
        S_bar: (I x I) degree-regularized row-normalized diffusion matrix with zero diagonal.
        clusters: (I,) integer array of spectral cluster assignments.
    """
    n_events = S_cooccur.shape[0]
    if n_events <= 1:
        return np.zeros((n_events, n_events), dtype=np.float64), np.zeros(n_events, dtype=int)

    S_dense = S_cooccur.toarray() if issparse(S_cooccur) else np.asarray(S_cooccur, dtype=np.float64)
    diag_counts = np.maximum(np.diag(S_dense), 0.0)
    N_eff = max(float(n_rep), float(np.max(diag_counts)) + 1.0, 1.0)

    p_i = np.clip(diag_counts / N_eff, 1e-12, 1.0 - 1e-12)
    p_ik = np.clip(S_dense / N_eff, 1e-12, 1.0 - 1e-12)
    p_indep = np.outer(p_i, p_i)

    min_cooccur = 1.5 if np.max(diag_counts) >= 5.0 else 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (S_dense >= min_cooccur) & (p_ik > p_indep),
            np.log(p_ik / np.maximum(p_indep, 1e-24)),
            0.0,
        )
        neg_log_pik = np.maximum(-np.log(p_ik), 1e-12)
        npmi = np.clip(pmi / neg_log_pik, 0.0, 1.0)

    np.fill_diagonal(npmi, 0.0)
    npmi[npmi < min_npmi] = 0.0
    W_sym = 0.5 * (npmi + npmi.T)

    # Retain top-K neighbors per event with squared NPMI weights
    W = np.zeros_like(W_sym)
    k_eff = min(max(1, top_k), n_events - 1)
    for i in range(n_events):
        row = W_sym[i]
        if np.any(row > 0):
            if np.count_nonzero(row > 0) > k_eff:
                top_idx = np.argpartition(row, -k_eff)[-k_eff:]
                W[i, top_idx] = row[top_idx] ** 2
            else:
                W[i] = row ** 2

    row_deg = np.sum(W, axis=1)
    S_bar = np.divide(
        W,
        np.maximum(row_deg[:, None] + _DEGREE_REGULARIZER, 1e-12),
        where=(row_deg[:, None] > 0),
    )
    S_bar[row_deg <= 0] = 0.0

    # Spectral cluster assignment from normalized Laplacian
    W_clust = 0.5 * (W + W.T)
    deg_c = np.sum(W_clust, axis=1)
    clusters = np.zeros(n_events, dtype=int)
    mask = deg_c > 0
    if np.sum(mask) >= 2:
        try:
            d_inv_sqrt = np.zeros_like(deg_c)
            d_inv_sqrt[mask] = 1.0 / np.sqrt(deg_c[mask])
            L_norm = np.diag(mask.astype(np.float64)) - (d_inv_sqrt[:, None] * W_clust * d_inv_sqrt[None, :])
            if n_events >= 4:
                from scipy.sparse.linalg import eigsh

                k = min(3, n_events - 1)
                vals, vecs = eigsh(csr_matrix(L_norm), k=k, which="SM")
                order = np.argsort(vals)
                vecs = vecs[:, order]
                if k >= 3:
                    clusters = (vecs[:, 1] > 0).astype(int) + 2 * (vecs[:, 2] > 0).astype(int)
                else:
                    clusters = (vecs[:, 1] > 0).astype(int)
            else:
                vals, vecs = np.linalg.eigh(L_norm)
                if len(vals) > 1:
                    clusters = (vecs[:, 1] > 0).astype(int)
        except Exception as exc:
            logger.debug("Spectral clustering fallback in _build_npmi_syndrome_graph: %s", exc)
            clusters = np.zeros(n_events, dtype=int)

    return S_bar, clusters


def _compute_dual_ewma_velocity(
    C: np.ndarray,
    B: np.ndarray,
    C_fast: np.ndarray | None,
    kappa: float = 0.5,
) -> np.ndarray:
    """Compute exposure-normalized Dual-EWMA surge velocity.

    Normalizes exponentially weighted recent counts C_fast against the drug's recent
    reporting volume share so drug-wide media notoriety waves cancel out.
    """
    if C_fast is None:
        return np.zeros_like(C)

    r_fast = np.sum(C_fast, axis=1)
    n_fast = float(np.sum(r_fast))
    if n_fast <= 0:
        return np.zeros_like(C)

    r_total = np.maximum(np.sum(C, axis=1), 1e-6)
    drug_fast_share = np.clip(r_fast / r_total, 1e-4, 1.0 - 1e-4)
    B_fast = B * drug_fast_share[:, None]

    C_slow = np.maximum(0.0, C - C_fast)
    B_slow = np.maximum(1e-5, B - B_fast)

    fast_rr_log2 = np.log2((C_fast + kappa) / (B_fast + kappa))
    slow_rr_log2 = np.log2((C_slow + kappa) / (B_slow + kappa))
    return np.where(C > 0, fast_rr_log2 - slow_rr_log2, 0.0)


def _fit_empirical_bayes_slab_priors(
    C_flat: np.ndarray,
    B_flat: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Fit identifiable 2-Gamma empirical Bayes prior parameters (alpha1, beta1, alpha2, beta2, pi_base)
    parameterized via component means (mu1, beta1, mu2, beta2, pi_base) with mu1 < 1 < mu2.
    """
    pos_mask = C_flat >= 1.0
    n_pos = C_flat[pos_mask]
    E_pos = B_flat[pos_mask]

    if len(n_pos) < 15:
        return _DEFAULT_PRIOR_PARAMS

    # Winsorize extreme cell counts during hyperprior estimation so single blockbuster spikes do not dominate
    n_fit = np.minimum(n_pos, E_pos + 15.0 * np.sqrt(E_pos + 1.0))
    gammaln_n_1 = gammaln(n_fit + 1.0)

    def _nll(theta: np.ndarray) -> float:
        mu1, b1, mu2, b2, pi_sig = theta
        r1 = mu1 * b1
        r2 = mu2 * b2
        p1 = np.clip(b1 / (b1 + E_pos + 1e-10), 1e-10, 1.0 - 1e-10)
        p2 = np.clip(b2 / (b2 + E_pos + 1e-10), 1e-10, 1.0 - 1e-10)

        log_nb1 = gammaln(n_fit + r1) - gammaln_n_1 - gammaln(r1) + r1 * np.log(p1) + n_fit * np.log(1.0 - p1)
        log_nb2 = gammaln(n_fit + r2) - gammaln_n_1 - gammaln(r2) + r2 * np.log(p2) + n_fit * np.log(1.0 - p2)
        max_l = np.maximum(log_nb1, log_nb2)
        mix = (1.0 - pi_sig) * np.exp(log_nb1 - max_l) + pi_sig * np.exp(log_nb2 - max_l)
        term1_log = max_l + np.log(np.maximum(mix, 1e-300))

        p0_1 = np.exp(r1 * np.log(p1))
        p0_2 = np.exp(r2 * np.log(p2))
        prob_pos = np.maximum(1.0 - ((1.0 - pi_sig) * p0_1 + pi_sig * p0_2), 1e-7)
        return float(np.sum(-term1_log + np.log(prob_pos)))

    bounds = (
        (0.60, 0.96),  # mu1: null component mean
        (2.00, 8.00),  # beta1: null component rate
        (2.40, 6.50),  # mu2: signal component mean
        (0.40, 1.80),  # beta2: signal component rate
        (0.04, 0.30),  # pi_base: baseline signal weight
    )
    try:
        res = minimize(
            _nll,
            x0=np.array([0.85, 3.0, 3.2, 0.80, 0.12], dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150},
        )
        mu1, b1, mu2, b2, pi_sig = res.x
        return float(mu1 * b1), float(b1), float(mu2 * b2), float(b2), float(pi_sig)
    except Exception as exc:
        logger.debug("Empirical Bayes prior optimization fallback: %s", exc)
        return _DEFAULT_PRIOR_PARAMS


def _compute_anytime_martingale_evalues(
    C_mat: np.ndarray,
    B_mat: np.ndarray,
    S_bar: np.ndarray,
    martingale_coupling: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sided Wald-Poisson mixture martingale e-values with cross-graph product
    martingale coupling (Vovk & Wang, 2021) and e-BH FDR adjustment (Wang & Ramdas, 2022).

    Mathematical Validity:
        1. For each cell (j, i), the raw mixture martingale is evaluated on actual counts (C_{ji}, B_{ji}):
               E^{raw}_{ji} = sum_m w_m exp(C_{ji} ln(lambda_m) - (lambda_m - 1) B_{ji})
           Under H_0: E[C_{ji}] <= B_{ji}, E_{H0}[E^{raw}_{ji}] <= 1.
        2. Because S_bar has a strictly zero diagonal (S_bar[i, i] = 0) and row sums d_i <= 1,
           the leave-one-out neighborhood e-value
               E^{neigh}_{ji} = (1 - d_i) + sum_{k != i} S_bar[i, k] * E^{raw}_{jk}
           satisfies E_{H0}[E^{neigh}_{ji}] <= 1 by linearity of expectation and depends only on
           columns k != i.
        3. Under cell-wise null independence across columns, E^{raw}_{ji} and E^{neigh}_{ji} are
           independent, so the coupled e-value
               E_{ji} = (1 - gamma) * E^{raw}_{ji} + gamma * E^{raw}_{ji} * min(E^{neigh}_{ji}, kappa_max)
           satisfies E_{H0}[E_{ji}] <= 1 by Ville's and Vovk-Wang's product martingale property.
    """
    C_flat = C_mat.ravel()
    B_flat = B_mat.ravel()

    weights = _MARTINGALE_GRID_WEIGHTS / np.sum(_MARTINGALE_GRID_WEIGHTS)
    log_grid = np.log(_MARTINGALE_GRID_RR)
    drift_grid = _MARTINGALE_GRID_RR - 1.0

    c_vec = C_flat[:, None]
    b_vec = B_flat[:, None]
    log_martingales = np.where(
        c_vec > b_vec,
        c_vec * log_grid[None, :] - b_vec * drift_grid[None, :],
        0.0,
    )
    log_martingales = np.clip(log_martingales, -50.0, 500.0)
    max_lm = np.max(log_martingales, axis=1, keepdims=True)
    mix_scaled = np.sum(weights[None, :] * np.exp(log_martingales - max_lm), axis=1)
    log_e_raw = np.squeeze(max_lm, axis=1) + np.log(np.maximum(mix_scaled, 1e-300))
    e_raw_flat = np.where(C_flat > B_flat, np.exp(np.minimum(log_e_raw, 300.0)), 1.0)

    # Leave-one-out neighborhood e-value across S_bar (strictly zero diagonal)
    E_raw_mat = e_raw_flat.reshape(C_mat.shape)
    E_Raw_capped = np.minimum(E_raw_mat, _MAX_NEIGHBOR_E_CAP)
    row_deg = np.clip(np.sum(S_bar, axis=1), 0.0, 1.0)
    E_neigh_mat = (1.0 - row_deg[None, :]) + (E_Raw_capped @ S_bar.T)
    e_neigh_flat = np.minimum(E_neigh_mat.ravel(), _MAX_NEIGHBOR_E_CAP)

    gamma = float(np.clip(martingale_coupling, 0.0, 1.0))
    e_values = np.where(
        C_flat > B_flat,
        (1.0 - gamma) * e_raw_flat + gamma * e_raw_flat * e_neigh_flat,
        1.0,
    )
    p_anytime = np.where(C_flat > B_flat, np.minimum(1.0, 1.0 / np.maximum(e_values, 1.0)), 1.0)

    # e-BH procedure (Wang & Ramdas, 2022)
    n_total = len(e_values)
    testable_idx = np.flatnonzero(C_flat > B_flat)
    m_testable = max(len(testable_idx), 1)
    e_fdr = np.ones(n_total, dtype=np.float64)
    if len(testable_idx) > 0:
        e_sub = e_values[testable_idx]
        order = np.argsort(-e_sub, kind="mergesort")
        sorted_e = e_sub[order]
        ranks = np.arange(1, len(testable_idx) + 1)
        adj = np.minimum(1.0, m_testable / np.maximum(ranks * sorted_e, 1e-12))
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        q_sub = np.empty_like(adj)
        q_sub[order] = adj
        e_fdr[testable_idx] = q_sub

    return e_values, p_anytime, e_fdr


def aegis(
    container: DataContainer,
    min_events: int = 3,
    fdr_threshold: float = 0.05,
    lb_threshold: float = 1.0,
    decision_metric: Literal["combined", "lb", "fdr", "e_value"] = "combined",
    null_rr: float = 1.0,
    huber_delta: float = 2.0,
    sinkhorn_iters: int = 5,
    latent_rank: int = 2,
    min_npmi: float = 0.05,
    syndromic_gain: float = 1.50,
    syndromic_coupling: float = 0.60,
    velocity_gain: float = 0.85,
    fast_window_fraction: float = 0.25,
    seed: int = 42,
) -> AnalysisResult:
    """Perform AEGIS (Anytime Empirical Graph-Informed Shrinkage) disproportionality analysis.

    Parameters:
        container: DataContainer holding binary report matrices or a contingency table.
        min_events: Minimum observed count required to flag a drug-event pair.
        fdr_threshold: Target false discovery rate threshold (default: 0.05).
        lb_threshold: Minimum 5th-percentile posterior relative risk threshold (default: 1.0).
        decision_metric: Decision rule ('combined', 'lb', 'fdr', or 'e_value').
        null_rr: Null relative risk benchmark (default: 1.0).
        huber_delta: Huber Winsorization threshold in standard deviations for baseline estimation.
        sinkhorn_iters: Number of Winsorized marginal refinement iterations.
        latent_rank: Rank of truncated SVD on Winsorized residuals for class/indication filtering.
        min_npmi: Minimum positive NPMI threshold to retain an edge in the symptom graph.
        syndromic_gain: Logistic gate slope for syndromic neighborhood coherence.
        syndromic_coupling: Conjugate Poisson-Gamma neighborhood precision pooling weight.
        velocity_gain: Logistic gate slope for Dual-EWMA surge velocity.
        fast_window_fraction: Effective fraction of recent reports weighted in Dual-EWMA velocity.
        seed: Random seed for randomized SVD reproducibility.

    Returns:
        AnalysisResult containing all drug-event pairs, detected signals, signal count,
        and fitted model parameters.
    """
    param_dict = build_params(
        "aegis",
        {
            "min_events": min_events,
            "fdr_threshold": fdr_threshold,
            "lb_threshold": lb_threshold,
            "decision_metric": decision_metric,
            "null_rr": null_rr,
            "huber_delta": huber_delta,
            "sinkhorn_iters": sinkhorn_iters,
            "latent_rank": latent_rank,
            "min_npmi": min_npmi,
            "syndromic_gain": syndromic_gain,
            "syndromic_coupling": syndromic_coupling,
            "velocity_gain": velocity_gain,
            "fast_window_fraction": fast_window_fraction,
            "seed": seed,
        },
    )

    # 1. Extract contingency matrix C, co-occurrence S, and exponentially weighted fast window C_fast
    C, S_cooccur, n_rep, products, events, C_fast = _extract_aegis_matrices(
        container, fast_window_fraction=fast_window_fraction
    )
    n_drugs, n_events = C.shape

    # 2. Counterfactual baseline estimation (LOCO + bias-corrected Huber Winsorization + low-rank SVD)
    B_cf, E_naive = _sinkhorn_huber_baseline(
        C,
        huber_delta=huber_delta,
        sinkhorn_iters=sinkhorn_iters,
        latent_rank=latent_rank,
        seed=seed,
    )
    unmasking_ratio = E_naive / np.maximum(B_cf, 1e-6)

    # 3. Positive NPMI symptom graph & leave-one-out neighborhood coherence
    S_bar, clusters = _build_npmi_syndrome_graph(S_cooccur, n_rep=n_rep, min_npmi=min_npmi)
    z0_mat = np.clip((C - B_cf) / np.sqrt(B_cf + 0.5), 0.0, 8.0)
    z_syn_mat = z0_mat @ S_bar.T

    # 4. Exposure-normalized Dual-EWMA surge velocity
    velocity_mat = _compute_dual_ewma_velocity(C, B_cf, C_fast)

    # 5. Conjugate Poisson-Gamma neighborhood & velocity pooling
    # Pools both pseudo-counts (dC) and pseudo-exposure (dB) so posterior mean remains a convex
    # shrinkage estimator while posterior precision increases for coherent syndromes.
    coherence_active = (C > B_cf) & (z_syn_mat > _COHERENCE_GATE_MIN) & (syndromic_coupling > 0.0)
    vel_active = (C > B_cf) & (velocity_mat > _VELOCITY_GATE_MIN) & (velocity_gain > 0.0)

    obs_rr = C / np.maximum(B_cf, 1e-6)
    neigh_rr = np.maximum(obs_rr, 1.0 + z_syn_mat / np.sqrt(B_cf + 0.5))

    tau_syn = np.where(coherence_active, float(syndromic_coupling) * np.minimum(z_syn_mat, 2.5), 0.0)
    tau_vel = np.where(vel_active, 0.30 * np.maximum(0.0, velocity_mat - 0.35), 0.0)
    tau_pool = tau_syn + tau_vel

    dB_mat = tau_pool * B_cf
    dC_mat = dB_mat * neigh_rr

    C_eff_mat = C + dC_mat
    B_eff_mat = B_cf + dB_mat

    # 6. Covariate-gated 2-Gamma empirical Bayes shrinkage
    C_flat = C.ravel()
    C_eff_flat = C_eff_mat.ravel()
    B_flat = B_cf.ravel()
    B_eff_flat = B_eff_mat.ravel()
    E_naive_flat = E_naive.ravel()
    unmask_flat = unmasking_ratio.ravel()
    z_syn_flat = z_syn_mat.ravel()
    vel_flat = velocity_mat.ravel()

    r1, b1, r2, b2, pi_base = _fit_empirical_bayes_slab_priors(C_flat, B_flat)
    base_logit = float(logit(np.clip(pi_base, 1e-4, 1.0 - 1e-4)))

    isolation_offset = np.where(
        (z_syn_flat >= _COHERENCE_GATE_MIN) | (vel_flat >= _COHERENCE_GATE_MIN),
        0.0,
        _ISOLATED_LOGIT_OFFSET,
    )
    gate_eta = base_logit + isolation_offset + syndromic_gain * z_syn_flat + velocity_gain * np.maximum(0.0, vel_flat)
    pi_context = np.where(C_flat > 0, expit(np.clip(gate_eta, -10.0, 10.0)), 0.0)
    w_null_prior = 1.0 - pi_context

    # Conjugate posterior mixture update at (C_eff, B_eff)
    p_post1 = np.clip(b1 / (b1 + B_eff_flat + 1e-10), 1e-10, 1.0 - 1e-10)
    p_post2 = np.clip(b2 / (b2 + B_eff_flat + 1e-10), 1e-10, 1.0 - 1e-10)
    gammaln_ceff_1 = gammaln(C_eff_flat + 1.0)

    log_q1 = (
        gammaln(C_eff_flat + r1)
        - gammaln_ceff_1
        - gammaln(r1)
        + r1 * np.log(p_post1)
        + C_eff_flat * np.log(1.0 - p_post1)
    )
    log_q2 = (
        gammaln(C_eff_flat + r2)
        - gammaln_ceff_1
        - gammaln(r2)
        + r2 * np.log(p_post2)
        + C_eff_flat * np.log(1.0 - p_post2)
    )
    max_lq = np.maximum(log_q1, log_q2)
    q1_s = np.exp(log_q1 - max_lq)
    q2_s = np.exp(log_q2 - max_lq)

    denom_q = w_null_prior * q1_s + pi_context * q2_s
    Qn = np.where(denom_q > 0, w_null_prior * q1_s / denom_q, 1.0)

    # Posterior null probability P(lambda <= null_rr | data)
    gd1 = gdtr(null_rr, r1 + C_eff_flat, np.maximum(b1 + B_eff_flat, 1e-10))
    gd2 = gdtr(null_rr, r2 + C_eff_flat, np.maximum(b2 + B_eff_flat, 1e-10))
    post_null_prob = np.where(C_flat > 0, np.clip(Qn * gd1 + (1.0 - Qn) * gd2, 0.0, 1.0), 1.0)
    post_sig_prob = 1.0 - post_null_prob

    # Posterior geometric mean (AEGIS_EBGM) and exact 5th / 95th quantiles
    dg1 = digamma(r1 + C_eff_flat) - np.log(np.maximum(b1 + B_eff_flat, 1e-10))
    dg2 = digamma(r2 + C_eff_flat) - np.log(np.maximum(b2 + B_eff_flat, 1e-10))
    aegis_log2 = np.where(C_flat > 0, (Qn * dg1 + (1.0 - Qn) * dg2) / np.log(2.0), 0.0)
    aegis_ebgm = np.where(C_flat > 0, np.power(2.0, aegis_log2), 0.0)

    aegis_lb = np.zeros_like(C_flat)
    aegis_ub = np.zeros_like(C_flat)
    pos_idx = np.flatnonzero(C_flat > 0)
    if len(pos_idx) > 0:
        aegis_lb[pos_idx] = quantiles(
            0.05,
            Qn[pos_idx],
            r1 + C_eff_flat[pos_idx],
            b1 + B_eff_flat[pos_idx],
            r2 + C_eff_flat[pos_idx],
            b2 + B_eff_flat[pos_idx],
        )
        aegis_ub[pos_idx] = quantiles(
            0.95,
            Qn[pos_idx],
            r1 + C_eff_flat[pos_idx],
            b1 + B_eff_flat[pos_idx],
            r2 + C_eff_flat[pos_idx],
            b2 + B_eff_flat[pos_idx],
        )

    # 7. Anytime-valid mixture martingale e-values on true counts (C, B_cf) with leave-one-out graph coupling
    e_values, p_anytime, e_fdr = _compute_anytime_martingale_evalues(
        C_mat=C,
        B_mat=B_cf,
        S_bar=S_bar,
    )

    # Cumulative Bayesian Posterior Error Probability (PEP) FDR
    bayesian_fdr = np.ones_like(C_flat)
    if len(pos_idx) > 0:
        p_null_pos = post_null_prob[pos_idx]
        ord_pos = np.argsort(p_null_pos, kind="mergesort")
        cum_fdr = np.cumsum(p_null_pos[ord_pos]) / np.arange(1, len(pos_idx) + 1)
        cum_fdr = np.clip(np.maximum.accumulate(cum_fdr), 0.0, 1.0)
        bfdr_sub = np.empty_like(cum_fdr)
        bfdr_sub[ord_pos] = cum_fdr
        bayesian_fdr[pos_idx] = bfdr_sub

    # Assemble result DataFrame
    n_pairs = len(C_flat)
    prod_indices, event_indices = np.unravel_index(np.arange(n_pairs), (n_drugs, n_events))
    prod_names = [products[p] for p in prod_indices]
    event_names = [events[e] for e in event_indices]
    cluster_assigned = [int(clusters[e]) for e in event_indices]
    is_int_counts = bool(np.all(np.equal(np.mod(C_flat, 1), 0)))
    count_col = C_flat.astype(int) if is_int_counts else np.round(C_flat, 4)

    all_signals_df = pd.DataFrame({
        "Product": prod_names,
        "Adverse Event": event_names,
        "Count": count_col,
        "Effective_Count": np.round(C_eff_flat, 3),
        "Expected Count": np.round(B_flat, 4),
        "Expected": np.round(B_flat, 4),
        "B_Counterfactual": np.round(B_flat, 4),
        "Naive_Expected": np.round(E_naive_flat, 4),
        "Unmasking_Ratio": np.round(unmask_flat, 3),
        "AEGIS_EBGM": np.round(aegis_ebgm, 4),
        "AEGIS_LB": np.round(aegis_lb, 4),
        "AEGIS_UB": np.round(aegis_ub, 4),
        "EBGM": np.round(aegis_ebgm, 4),
        "LowerBound": np.round(aegis_lb, 4),
        "UpperBound": np.round(aegis_ub, 4),
        "Context_Prior": np.round(pi_context, 4),
        "Posterior_Sig_Prob": np.round(post_sig_prob, 4),
        "Syndromic_Coherence": np.round(z_syn_flat, 4),
        "Surge_Velocity": np.round(vel_flat, 4),
        "E_Value": np.round(e_values, 4),
        "e_value": np.round(e_values, 4),
        "p_value": post_null_prob,
        "anytime_p_value": p_anytime,
        "fdr": bayesian_fdr,
        "e_fdr": e_fdr,
        "Syndrome_Cluster": cluster_assigned,
    })

    all_signals_df = all_signals_df.sort_values(
        by=["AEGIS_LB", "AEGIS_EBGM", "E_Value", "Product", "Adverse Event"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    count_ok = all_signals_df["Count"] >= min_events
    if decision_metric == "lb":
        sig_mask = count_ok & (all_signals_df["AEGIS_LB"] > lb_threshold)
    elif decision_metric == "fdr":
        sig_mask = count_ok & (all_signals_df["fdr"] <= fdr_threshold) & (all_signals_df["AEGIS_EBGM"] > null_rr)
    elif decision_metric == "e_value":
        sig_mask = count_ok & (all_signals_df["E_Value"] >= (1.0 / max(fdr_threshold, 1e-6)))
    else:
        e_floor = max(4.0, 0.35 / max(fdr_threshold, 1e-6))
        sig_mask = (
            count_ok
            & (all_signals_df["AEGIS_LB"] > lb_threshold)
            & (all_signals_df["fdr"] <= fdr_threshold)
            & (all_signals_df["E_Value"] >= e_floor)
        )

    signals_df = all_signals_df.loc[sig_mask].reset_index(drop=True)

    param_dict["prior_slab"] = {
        "alpha1": r1,
        "beta1": b1,
        "alpha2": r2,
        "beta2": b2,
        "pi_base": pi_base,
    }

    return AnalysisResult(
        signals=signals_df,
        num_signals=len(signals_df),
        all_signals=all_signals_df,
        params=param_dict,
    )

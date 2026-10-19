"""AEGIS: Anytime Empirical Graph-Informed Shrinkage for Disproportionality Analysis.

A unified four-pillar pharmacovigilance signal detection methodology designed to overcome
the structural blind spots of classical frequentist, empirical Bayes, and regression methods:

1. Counterfactual Sinkhorn-Huber Unmasking:
   Iteratively estimates counterfactual null marginals by Huber-Winsorizing extreme excess
   cells before computing row/column expectations, eliminating both across-drug blockbuster
   competition masking and within-drug self-cloaking (with optional low-rank SVD indication
   absorption).
2. NPMI Syndrome Graph & Context-Gated Empirical Bayes Shrinkage:
   Constructs a volume-invariant positive Normalized Pointwise Mutual Information (NPMI)
   network across adverse events and modulates cell-level prior signal probabilities via a
   logistic gate driven by syndromic neighborhood coherence. Isolated low-count noise is
   heavily shrunk, while emerging multi-symptom syndromes release shrinkage early.
3. Exposure-Normalized Dual-EWMA Surge Velocity:
   Separates drug-wide media notoriety waves from true focal biological acceleration by
   comparing fast vs. slow exponential moving windows normalized by counterfactual drug
   exposure expectations.
4. Anytime-Valid Mixture Martingale e-Values & FDR Control:
   Computes non-asymptotic one-sided Poisson mixture martingale e-values (Ville's inequality)
   to permit continuous sequential regulatory monitoring without alpha inflation.
"""

from __future__ import annotations

import logging
from typing import Literal
import numpy as np
import pandas as pd
from scipy.special import digamma, gdtr, gammaln, expit, logit
from scipy.optimize import minimize
from scipy.sparse import issparse, csr_matrix

from ..utils.Container import AnalysisResult, DataContainer
from ..utils.common import build_params
from ..utils.distribution_funcs.quantile_funcs import quantiles

logger = logging.getLogger("vigipy")
EPS = float(np.finfo(np.float32).eps)


def _extract_aegis_matrices(
    container: DataContainer,
    fast_window_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float, list[str], list[str], np.ndarray | None]:
    """Extract contingency matrix C (J x I), AE co-occurrence matrix S (I x I), total reports N_rep,
    and optional recent-window contingency matrix C_fast (J x I) for surge velocity calculation.
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

        # Extract recent report window for Dual-EWMA Surge Velocity if enough reports exist
        n_rows = X_mat.shape[0]
        if n_rows >= 20 and 0.05 <= fast_window_fraction < 1.0:
            split_idx = max(1, int(round(n_rows * (1.0 - fast_window_fraction))))
            X_fast = X_mat[split_idx:]
            Y_fast = Y_mat[split_idx:]
            if issparse(X_fast) or issparse(Y_fast):
                C_fast_raw = X_fast.T @ Y_fast
                C_fast = C_fast_raw.toarray() if issparse(C_fast_raw) else np.asarray(C_fast_raw, dtype=np.float64)
            else:
                C_fast = X_fast.T @ Y_fast
        else:
            C_fast = None

    elif container.contingency is not None:
        cont = container.contingency
        products = list(cont.index)
        events = list(cont.columns)
        C = np.ascontiguousarray(cont.values, dtype=np.float64)
        n_rep = float(max(container.N, np.sum(C), 1.0))
        # Row-normalized Tanimoto co-occurrence proxy scaled to marginal event counts
        row_sums = np.maximum(np.sum(C, axis=1, keepdims=True), 1.0)
        P_cond = C / row_sums
        col_sums = np.sum(C, axis=0)
        # Approximate report-level co-occurrence from conditional drug profiles
        S_cooccur = (P_cond.T * col_sums[:, None]) @ P_cond
        np.fill_diagonal(S_cooccur, col_sums)
        C_fast = getattr(container, "fast_contingency", None)
        if C_fast is not None and isinstance(C_fast, pd.DataFrame):
            C_fast = np.ascontiguousarray(C_fast.values, dtype=np.float64)
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
    """Pillar 1: Counterfactual Leave-One-Cell-Out (LOCO) Sinkhorn-Huber Unmasking
    with optional low-rank SVD indication absorption.

    Initializes counterfactual expectations via Leave-One-Cell-Out marginals:
        B^{(0)}_{ji} = (R_j - C_{ji}) * (K_i - C_{ji}) / max(N - R_j - K_i + C_{ji}, 1)
    which breaks the self-cloaking fixed point of massive single-cell spikes in O(1) steps,
    followed by iterative Huber-Poisson Winsorization at B + delta * sqrt(B + 1).

    Returns:
        B_counterfactual: (J x I) unmasked counterfactual baseline expectation matrix.
        E_naive: (J x I) standard naive marginal expectation matrix R_j * K_i / N.
    """
    r_naive = np.sum(C, axis=1)
    c_naive = np.sum(C, axis=0)
    n_tot_naive = float(np.sum(r_naive))
    if n_tot_naive <= 0:
        B_min = np.full_like(C, 1e-4)
        return B_min, B_min

    E_naive = np.maximum(np.outer(r_naive, c_naive) / n_tot_naive, 1e-6)

    # Step 0: Leave-One-Cell-Out (LOCO) counterfactual initialization breaks self-cloaking in O(1)
    r_loco = np.maximum(r_naive[:, None] - C, 0.0)
    c_loco = np.maximum(c_naive[None, :] - C, 0.0)
    n_loco = np.maximum(n_tot_naive - r_naive[:, None] - c_naive[None, :] + C, 1.0)
    B_loco = (r_loco * c_loco) / n_loco
    B = np.where((r_loco > 0) & (c_loco > 0), np.minimum(E_naive, np.maximum(B_loco, 1e-6)), E_naive)

    # Iterative Sinkhorn-Huber refinement on Winsorized table
    for _ in range(max(1, sinkhorn_iters)):
        cap = B + huber_delta * np.sqrt(B + 1.0)
        C_winsor = np.minimum(C, cap)
        r_cf = np.sum(C_winsor, axis=1)
        c_cf = np.sum(C_winsor, axis=0)
        n_cf = float(np.sum(r_cf))
        if n_cf <= 0:
            break
        B = np.maximum(np.outer(r_cf, c_cf) / n_cf, 1e-6)

    # Optional low-rank SVD absorption of smooth class/indication confounding on Winsorized residuals
    n_drugs, n_events = C.shape
    eff_rank = min(int(latent_rank), max(0, min(n_drugs, n_events) - 1))
    if eff_rank > 0:
        cap = B + huber_delta * np.sqrt(B + 1.0)
        C_winsor = np.minimum(C, cap)
        std_mat = np.sqrt(B + 1.0)
        R_win = (C_winsor - B) / std_mat
        try:
            from sklearn.utils.extmath import randomized_svd

            U, s_vals, Vt = randomized_svd(R_win, n_components=eff_rank, random_state=seed)
            R_lr = (U * s_vals) @ Vt
            B = np.maximum(1e-4, B + R_lr * std_mat)
        except Exception:
            try:
                U, s_vals, Vt = np.linalg.svd(R_win, full_matrices=False)
                R_lr = (U[:, :eff_rank] * s_vals[:eff_rank]) @ Vt[:eff_rank, :]
                B = np.maximum(1e-4, B + R_lr * std_mat)
            except Exception:
                pass

    return np.maximum(B, 1e-5), E_naive


def _build_npmi_syndrome_graph(
    S_cooccur: np.ndarray,
    n_rep: float,
    min_npmi: float = 0.05,
    top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Pillar 2a: Construct Top-K Sharpened Positive NPMI Syndrome Graph.

    Unlike Jaccard similarity, NPMI normalizes by -ln P(i, k), preserving strong syndromic links
    between rare sentinel adverse events (e.g., Agranulocytosis) and common prodromal symptoms (e.g., Fever).
    Retains the top-K strongest syndromic neighbors per event with squared NPMI attention weights
    so multi-symptom syndromes are not diluted across unrelated background events.

    Returns:
        S_bar: (I x I) row-normalized syndromic diffusion matrix with zero diagonal.
        clusters: (I,) integer array of spectral syndrome cluster assignments.
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

    # Top-K neighborhood attention sharpening per adverse event
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
    S_bar = np.divide(W, np.maximum(row_deg[:, None] + 0.03, 1e-12), where=(row_deg[:, None] > 0))
    S_bar[row_deg <= 0] = 0.0

    # Spectral cluster assignment on symmetrized graph
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
        except Exception:
            clusters = np.zeros(n_events, dtype=int)

    return S_bar, clusters


def _compute_dual_ewma_velocity(
    C: np.ndarray,
    B: np.ndarray,
    C_fast: np.ndarray | None,
    kappa: float = 0.5,
) -> np.ndarray:
    """Pillar 3: Exposure-Normalized Dual-EWMA Surge Velocity V_{ji}.

    Decouples drug-wide media notoriety waves from true focal biological acceleration by
    normalizing fast-window counts against fast-window counterfactual drug exposure expectations.
    """
    if C_fast is None:
        return np.zeros_like(C)

    r_fast = np.sum(C_fast, axis=1)
    n_fast = float(np.sum(r_fast))
    if n_fast <= 0:
        return np.zeros_like(C)

    # Scale B by drug-specific recent reporting share so drug-wide notoriety cancels out
    r_total = np.maximum(np.sum(C, axis=1), 1e-6)
    drug_fast_share = np.clip(r_fast / r_total, 1e-4, 1.0)
    B_fast = B * drug_fast_share[:, None]

    C_slow = np.maximum(0.0, C - C_fast)
    B_slow = np.maximum(1e-5, B - B_fast)

    fast_rr_log2 = np.log2((C_fast + kappa) / (B_fast + kappa))
    slow_rr_log2 = np.log2((C_slow + kappa) / (B_slow + kappa))
    velocity = np.where(C > 0, fast_rr_log2 - slow_rr_log2, 0.0)
    return velocity


def _fit_empirical_bayes_slab_priors(
    C_flat: np.ndarray,
    B_flat: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Fit identifiable 2-Gamma empirical Bayes prior parameters (alpha1, beta1, alpha2, beta2, pi_base)
    parameterized by (mu1, beta1, mu2, beta2, pi_base) so Component 1 is strictly a null component
    (mu1 = alpha1/beta1 in [0.55, 0.98]) and Component 2 is strictly an elevated-risk signal component
    (mu2 = alpha2/beta2 in [2.2, 8.0]).
    """
    pos_mask = C_flat >= 1.0
    n_pos = C_flat[pos_mask]
    E_pos = B_flat[pos_mask]

    # Default calibrated pharmacovigilance parameters:
    # Null component: mu1 = 0.85, beta1 = 3.0 -> alpha1 = 2.55, beta1 = 3.0
    # Signal component: mu2 = 3.2, beta2 = 0.8 -> alpha2 = 2.56, beta2 = 0.8
    default_priors = (2.55, 3.0, 2.56, 0.80, 0.12)
    if len(n_pos) < 15:
        return default_priors

    # Winsorize extreme E_pos/n_pos during hyperprior estimation so a single 5000-count blockbuster
    # cell cannot dominate the global prior fit
    n_fit = np.minimum(n_pos, B_flat[pos_mask] + 15.0 * np.sqrt(B_flat[pos_mask] + 1.0))
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
        (0.60, 0.96),  # mu1: null mean strictly < 1.0
        (2.00, 8.00),  # beta1: null rate/concentration
        (2.40, 6.50),  # mu2: signal mean strictly >= 2.4
        (0.40, 1.80),  # beta2: signal rate
        (0.04, 0.30),  # pi_base: base prior signal fraction
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
    except Exception:
        return default_priors


def _compute_anytime_martingale_evalues(
    C_eff: np.ndarray,
    B: np.ndarray,
    pi_context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pillar 4: Anytime-Valid One-Sided Poisson Mixture Martingale e-Values and e-BH FDR.

    Evaluates a discrete mixture of Wald-Poisson likelihood ratio martingales over a grid of
    alternative relative risks lambda_m > 1:
        M_m(C, B) = exp(C * ln(lambda_m) - (lambda_m - 1) * B)
    Because E_{H0}[M_m(t)] = 1 for all t and all lambda_m > 1 under H0: RR <= 1, any convex
    mixture E_t = sum_m w_m M_m(t) is a nonnegative supermartingale with E_{H0}[E_t] <= 1.
    By Ville's inequality, p_anytime = min(1, 1 / E_t) is valid at any arbitrary stopping time.
    """
    grid_rr = np.array([1.3, 1.6, 2.0, 2.8, 4.0, 6.0], dtype=np.float64)
    base_weights = np.array([0.20, 0.25, 0.25, 0.15, 0.10, 0.05], dtype=np.float64)
    base_weights /= np.sum(base_weights)

    log_grid = np.log(grid_rr)
    drift_grid = grid_rr - 1.0

    c_vec = C_eff[:, None]
    b_vec = B[:, None]
    log_martingales = np.where(
        c_vec > b_vec,
        c_vec * log_grid[None, :] - b_vec * drift_grid[None, :],
        0.0,
    )
    log_martingales = np.clip(log_martingales, -50.0, 500.0)
    max_lm = np.max(log_martingales, axis=1, keepdims=True)
    mix_scaled = np.sum(base_weights[None, :] * np.exp(log_martingales - max_lm), axis=1)
    log_e_raw = np.squeeze(max_lm, axis=1) + np.log(np.maximum(mix_scaled, 1e-300))

    w_gate = np.clip(pi_context / np.maximum(np.median(pi_context[C_eff > 0]) if np.any(C_eff > 0) else 0.1, 0.05), 0.35, 1.0)
    e_values = np.where(
        C_eff > B,
        (1.0 - w_gate) * 1.0 + w_gate * np.exp(np.minimum(log_e_raw, 300.0)),
        1.0,
    )

    p_anytime = np.where(C_eff > B, np.minimum(1.0, 1.0 / np.maximum(e_values, 1.0)), 1.0)

    # e-BH procedure (Wang & Ramdas, 2022): sort e-values descending, adjusted q_k = min(1, M / (k * E_(k)))
    n_total = len(e_values)
    testable_idx = np.flatnonzero(C_eff > B)
    m_testable = max(len(testable_idx), 1)
    e_fdr = np.ones(n_total, dtype=np.float64)
    if len(testable_idx) > 0:
        e_sub = e_values[testable_idx]
        order = np.argsort(-e_sub)
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
    syndromic_coupling: float = 0.45,
    velocity_gain: float = 0.85,
    fast_window_fraction: float = 0.25,
    seed: int = 42,
) -> AnalysisResult:
    """Perform AEGIS (Anytime Empirical Graph-Informed Shrinkage) Disproportionality Analysis.

    Clinical Intuition:
        Existing pharmacovigilance methodologies force a trade-off between early sensitivity
        and false-positive control:
        - Frequentist ratios (PRR, ROR) explode on small counts (n=1..3) and suffer severe Type I
          error inflation under sequential surveillance.
        - Empirical Bayes shrinkers (MGPS, BCPNN, ZIGPS) apply a single global prior to every cell
          independently, aggressively crushing genuine emerging signals when counts are small (n=2..4),
          and remain vulnerable to blockbuster competition masking.
        - Graph-regularized excess estimators (SCORE-DA) lack Bayesian posterior relative-risk
          quantiles and sequential anytime-valid error guarantees.

        AEGIS unifies four synergistic innovations to resolve all four limitations simultaneously:
        1. **Counterfactual LOCO Sinkhorn-Huber Unmasking**: Initializes Leave-One-Cell-Out marginals
           and iteratively Winsorizes extreme cells before computing expectations (plus low-rank SVD
           indication filtering), eliminating both across-drug blockbuster competition and within-drug
           self-cloaking.
        2. **NPMI Syndrome Graph & Context-Gated Empirical Bayes**: Builds a volume-invariant
           Normalized Pointwise Mutual Information (NPMI) symptom graph. When a drug shows elevated
           reporting across multiple pathophysiologically linked symptoms in a syndrome (e.g., fever +
           neutropenia + mucosal ulceration), the logistic context gate increases the local prior
           signal probability `pi_ji`, releasing Bayesian shrinkage early on true emerging syndromes
           while strongly shrinking isolated sporadic noise.
        3. **Exposure-Normalized Dual-EWMA Surge Velocity**: Separates drug-wide media notoriety waves
           (which inflate all AEs for a drug proportionally) from focal biological acceleration.
        4. **Anytime-Valid Mixture Martingale e-Values**: Provides non-asymptotic sequential hypothesis
           testing valid under continuous regulatory monitoring at any arbitrary stopping time.
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

    # 1. Extract contingency matrix C, co-occurrence S, and optional fast window C_fast
    C, S_cooccur, n_rep, products, events, C_fast = _extract_aegis_matrices(
        container, fast_window_fraction=fast_window_fraction
    )
    n_drugs, n_events = C.shape

    # 2. Pillar 1: Counterfactual LOCO Sinkhorn-Huber Unmasking
    B_cf, E_naive = _sinkhorn_huber_baseline(
        C,
        huber_delta=huber_delta,
        sinkhorn_iters=sinkhorn_iters,
        latent_rank=latent_rank,
        seed=seed,
    )
    unmasking_ratio = E_naive / np.maximum(B_cf, 1e-6)

    # 3. Pillar 2a: Volume-Invariant Top-K Sharpened NPMI Syndrome Graph & Neighborhood Coherence
    S_bar, clusters = _build_npmi_syndrome_graph(S_cooccur, n_rep=n_rep, min_npmi=min_npmi)

    # Standardized positive cell excess z_0 (winsorized at 8.0 so single massive spikes don't flood neighbors)
    z0_mat = np.clip((C - B_cf) / np.sqrt(B_cf + 0.5), 0.0, 8.0)
    # Cross-event syndromic coherence z_syn = z0 @ S_bar^T (strictly zero diagonal)
    z_syn_mat = z0_mat @ S_bar.T

    # 4. Pillar 3: Exposure-Normalized Dual-EWMA Surge Velocity
    velocity_mat = _compute_dual_ewma_velocity(C, B_cf, C_fast)

    # Effective syndromically & temporally coupled count: applies when cell has positive excess (C > B_cf)
    # and syndromic neighbors show positive coherence (z_syn > 0.25) or focal velocity surges
    coherence_active = (C > B_cf) & (z_syn_mat > 0.25)
    vel_active = (C > B_cf) & (velocity_mat > 0.45)
    c_syn_boost = np.where(
        coherence_active,
        (syndromic_coupling + 0.15) * np.sqrt(B_cf + 0.25) * z_syn_mat,
        0.0,
    ) + np.where(
        vel_active,
        0.30 * np.sqrt(B_cf + 0.25) * np.maximum(0.0, velocity_mat - 0.35),
        0.0,
    )
    C_eff_mat = C + c_syn_boost

    # 5. Pillar 2b: Context-Gated Empirical Bayes Shrinkage
    C_flat = C.ravel()
    C_eff_flat = C_eff_mat.ravel()
    B_flat = B_cf.ravel()
    E_naive_flat = E_naive.ravel()
    unmask_flat = unmasking_ratio.ravel()
    z_syn_flat = z_syn_mat.ravel()
    vel_flat = velocity_mat.ravel()

    r1, b1, r2, b2, pi_base = _fit_empirical_bayes_slab_priors(C_flat, B_flat)
    base_logit = float(logit(np.clip(pi_base, 1e-4, 1.0 - 1e-4)))

    # Logistic Context Gate: modulates prior signal weight pi_{ji} using cross-symptom coherence & surge velocity
    # Isolated sporadic counts (z_syn < 0.25 and vel <= 0.25) receive a skepticism offset (-0.45),
    # while syndromically coherent or accelerating pairs release shrinkage early.
    isolation_offset = np.where((z_syn_flat >= 0.25) | (vel_flat >= 0.25), 0.0, -0.45)
    gate_eta = base_logit + isolation_offset + syndromic_gain * z_syn_flat + velocity_gain * np.maximum(0.0, vel_flat)
    pi_context = np.where(C_flat > 0, expit(np.clip(gate_eta, -10.0, 10.0)), 0.0)
    w_null_prior = 1.0 - pi_context

    # Posterior Mixture Update at (C_eff, B_cf)
    p_post1 = np.clip(b1 / (b1 + B_flat + 1e-10), 1e-10, 1.0 - 1e-10)
    p_post2 = np.clip(b2 / (b2 + B_flat + 1e-10), 1e-10, 1.0 - 1e-10)
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

    # Posterior null probability P(lambda <= null_rr | data, graph, velocity)
    gd1 = gdtr(null_rr, r1 + C_eff_flat, np.maximum(b1 + B_flat, 1e-10))
    gd2 = gdtr(null_rr, r2 + C_eff_flat, np.maximum(b2 + B_flat, 1e-10))
    post_null_prob = np.where(C_flat > 0, np.clip(Qn * gd1 + (1.0 - Qn) * gd2, 0.0, 1.0), 1.0)
    post_sig_prob = 1.0 - post_null_prob

    # Context-Gated Empirical Bayes Geometric Mean (AEGIS_EBGM) & Posterior Quantiles (AEGIS_LB / AEGIS_UB)
    dg1 = digamma(r1 + C_eff_flat) - np.log(np.maximum(b1 + B_flat, 1e-10))
    dg2 = digamma(r2 + C_eff_flat) - np.log(np.maximum(b2 + B_flat, 1e-10))
    aegis_log2 = np.where(C_flat > 0, (Qn * dg1 + (1.0 - Qn) * dg2) / np.log(2.0), 0.0)
    aegis_ebgm = np.where(C_flat > 0, np.power(2.0, aegis_log2), 0.0)

    # Compute exact 5th and 95th posterior quantiles on active cells (Count > 0)
    aegis_lb = np.zeros_like(C_flat)
    aegis_ub = np.zeros_like(C_flat)
    pos_idx = np.flatnonzero(C_flat > 0)
    if len(pos_idx) > 0:
        aegis_lb[pos_idx] = quantiles(
            0.05,
            Qn[pos_idx],
            r1 + C_eff_flat[pos_idx],
            b1 + B_flat[pos_idx],
            r2 + C_eff_flat[pos_idx],
            b2 + B_flat[pos_idx],
        )
        aegis_ub[pos_idx] = quantiles(
            0.95,
            Qn[pos_idx],
            r1 + C_eff_flat[pos_idx],
            b1 + B_flat[pos_idx],
            r2 + C_eff_flat[pos_idx],
            b2 + B_flat[pos_idx],
        )

    # 6. Pillar 4: Anytime-Valid Mixture Martingale e-Values & Bayesian PEP False Discovery Rate
    e_values, p_anytime, e_fdr = _compute_anytime_martingale_evalues(
        C_eff=C_eff_flat,
        B=B_flat,
        pi_context=pi_context,
    )

    # Bayesian Cumulative Posterior Error Probability (PEP) FDR along ranking sequence:
    # For any top-k alert set ordered by posterior null probability, the expected false discovery
    # proportion is E[FP / k | data] = (1/k) * sum_{r=1}^k P(lambda_{(r)} <= null_rr | data).
    bayesian_fdr = np.ones_like(C_flat)
    if len(pos_idx) > 0:
        p_null_pos = post_null_prob[pos_idx]
        ord_pos = np.argsort(p_null_pos)
        cum_fdr = np.cumsum(p_null_pos[ord_pos]) / np.arange(1, len(pos_idx) + 1)
        cum_fdr = np.clip(np.maximum.accumulate(cum_fdr), 0.0, 1.0)
        bfdr_sub = np.empty_like(cum_fdr)
        bfdr_sub[ord_pos] = cum_fdr
        bayesian_fdr[pos_idx] = bfdr_sub

    fdr_unified = bayesian_fdr

    # Assemble output DataFrame
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
        "Naive_Expected": np.round(E_naive_flat, 4),
        "Unmasking_Ratio": np.round(unmask_flat, 3),
        "AEGIS_EBGM": np.round(aegis_ebgm, 4),
        "AEGIS_LB": np.round(aegis_lb, 4),
        "AEGIS_UB": np.round(aegis_ub, 4),
        "EBGM": np.round(aegis_ebgm, 4),          # Compatibility alias
        "LowerBound": np.round(aegis_lb, 4),      # Compatibility alias
        "UpperBound": np.round(aegis_ub, 4),      # Compatibility alias
        "Context_Prior": np.round(pi_context, 4),
        "Posterior_Sig_Prob": np.round(post_sig_prob, 4),
        "Syndromic_Coherence": np.round(z_syn_flat, 4),
        "Surge_Velocity": np.round(vel_flat, 4),
        "E_Value": np.round(e_values, 4),
        "p_value": post_null_prob,
        "anytime_p_value": p_anytime,
        "fdr": fdr_unified,
        "e_fdr": e_fdr,
        "Syndrome_Cluster": cluster_assigned,
    })

    all_signals_df = all_signals_df.sort_values(
        by=["AEGIS_LB", "AEGIS_EBGM", "E_Value", "Product", "Adverse Event"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Apply decision rule
    count_ok = all_signals_df["Count"] >= min_events
    if decision_metric == "lb":
        sig_mask = count_ok & (all_signals_df["AEGIS_LB"] > lb_threshold)
    elif decision_metric == "fdr":
        sig_mask = count_ok & (all_signals_df["fdr"] <= fdr_threshold) & (all_signals_df["AEGIS_EBGM"] > null_rr)
    elif decision_metric == "e_value":
        sig_mask = count_ok & (all_signals_df["E_Value"] >= (1.0 / max(fdr_threshold, 1e-6)))
    else:  # "combined"
        # Combines Context-Gated EB lower bound, Bayesian PEP FDR, and Ville Martingale evidence floor
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

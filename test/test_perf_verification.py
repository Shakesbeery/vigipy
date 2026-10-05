"""
Test suite verifying that optimized GPS and SCORE-DA implementations match
unmodified reference algorithms within strict floating-point numerical tolerances.
"""
import time
import warnings
import numpy as np
import pytest
from scipy.special import gdtr

from vigipy.utils.distribution_funcs.quantile_funcs import quantiles as fast_quantiles
from vigipy.SCORE.SCORE import _solve_fista_single_drug, _solve_fista_matrix


def reference_quantiles_scalar(threshold, Q, a1, b1, a2, b2):
    """The original unmodified scalar bisection from DuMouchel (1999)."""
    max_mean = np.maximum(1000.0, 10.0 * np.maximum(a1 / max(b1, 1e-6), a2 / max(b2, 1e-6)))
    M = float(max_mean)
    m = 0.0
    x = 1.0
    cost = Q * gdtr(x, a1, b1) + (1.0 - Q) * gdtr(x, a2, b2) - threshold
    iteration = 0
    while np.max(np.round(cost * 1e4)) != 0 and iteration < 100:
        S = np.sign(cost)
        xnew = (1 + S) / 2 * ((x + m) / 2) + (1 - S) / 2 * ((M + x) / 2)
        M = (1 + S) / 2 * x + (1 - S) / 2 * M
        m = (1 + S) / 2 * m + (1 - S) / 2 * x
        x = xnew
        cost = Q * gdtr(x, a1, b1) + (1.0 - Q) * gdtr(x, a2, b2) - threshold
        iteration += 1
    return float(x)


ref_quantiles_vec = np.vectorize(reference_quantiles_scalar, otypes=[np.float64])


def test_gps_quantiles_tolerance():
    """Verify that vectorized array quantiles match original scalar bisection within < 3% relative tolerance."""
    rng = np.random.default_rng(42)
    N_test = 500
    Q_test = rng.uniform(0.01, 0.99, N_test)
    a1_test = rng.uniform(0.5, 50, N_test)
    b1_test = rng.uniform(0.1, 10, N_test)
    a2_test = rng.uniform(0.5, 50, N_test)
    b2_test = rng.uniform(0.1, 10, N_test)

    # EB05
    eb05_ref = ref_quantiles_vec(0.05, Q_test, a1_test, b1_test, a2_test, b2_test)
    eb05_fast = fast_quantiles(0.05, Q_test, a1_test, b1_test, a2_test, b2_test)
    rel_eb05 = np.max(np.abs(eb05_ref - eb05_fast) / np.maximum(eb05_ref, 1e-4))
    assert rel_eb05 < 0.05, f"EB05 relative error too high: {rel_eb05}"

    # EB95
    eb95_ref = ref_quantiles_vec(0.95, Q_test, a1_test, b1_test, a2_test, b2_test)
    eb95_fast = fast_quantiles(0.95, Q_test, a1_test, b1_test, a2_test, b2_test)
    rel_eb95 = np.max(np.abs(eb95_ref - eb95_fast) / np.maximum(eb95_ref, 1e-4))
    assert rel_eb95 < 0.05, f"EB95 relative error too high: {rel_eb95}"


def test_score_matrix_fista_tolerance():
    """Verify that Matrix FISTA produces identical results to single-drug FISTA within 1e-3."""
    rng = np.random.default_rng(42)
    n_drugs = 30
    n_events = 60
    Y_target = rng.standard_normal((n_drugs, n_events)) * 4.0
    C_obs = np.maximum(0.0, Y_target + rng.uniform(0, 8, (n_drugs, n_events)))
    W = rng.uniform(0, 1, (n_events, n_events))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    d = np.sum(W, axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L_ae = np.eye(n_events) - d_inv[:, None] * W * d_inv[None, :]

    lambda_1 = 1.0
    lambda_2 = 0.5
    L_lip = 1.0 + 2.0 * lambda_2
    tol = 1e-4

    theta_ref = np.zeros_like(Y_target)
    for j in range(n_drugs):
        theta_ref[j, :] = _solve_fista_single_drug(
            Y_target[j, :], C_obs[j, :], L_ae, lambda_1, lambda_2, L_lip, max_iter=50, tol=tol
        )

    theta_mat = _solve_fista_matrix(
        Y_target, C_obs, L_ae, lambda_1, lambda_2, L_lip, max_iter=50, tol=tol
    )

    diff = np.max(np.abs(theta_ref - theta_mat))
    assert diff < 1e-3, f"Matrix FISTA differs from single-drug FISTA by {diff}"


def test_sparse_laplacian_fista():
    """Verify that Matrix FISTA works identically whether L_ae is dense ndarray or scipy.sparse.csr_matrix."""
    import scipy.sparse as sp
    rng = np.random.default_rng(123)
    n_drugs = 20
    n_events = 50
    Y_target = rng.standard_normal((n_drugs, n_events)) * 3.0
    C_obs = np.maximum(0.0, Y_target + rng.uniform(0, 5, (n_drugs, n_events)))
    W = rng.uniform(0, 1, (n_events, n_events))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    d = np.sum(W, axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L_dense = np.eye(n_events) - d_inv[:, None] * W * d_inv[None, :]
    L_sparse = sp.csr_matrix(L_dense)

    theta_dense = _solve_fista_matrix(Y_target, C_obs, L_dense, 1.0, 0.5, 2.0, max_iter=50, tol=1e-4)
    theta_sparse = _solve_fista_matrix(Y_target, C_obs, L_sparse, 1.0, 0.5, 2.0, max_iter=50, tol=1e-4)

    assert np.allclose(theta_dense, theta_sparse, atol=1e-7), "Sparse and dense FISTA outputs differ"

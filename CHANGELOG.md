# Changelog

## 3.5.0

### Added
- **AEGIS (`aegis`, `AEGISConfig`)**: Anytime Empirical Graph-Informed Shrinkage — a novel 4-pillar disproportionality methodology combining:
  1. Counterfactual Leave-One-Cell-Out (LOCO) Sinkhorn-Huber Winsorization and Huber-clipped SVD baseline estimation (`B_Counterfactual`, `Unmasking_Ratio`) to eliminate across-drug blockbuster competition and within-drug self-cloaking.
  2. Volume-invariant Top-$K$ sharpened Normalized Pointwise Mutual Information (NPMI) syndrome graph and logistic context-gated 2-Gamma Empirical Bayes shrinkage (`AEGIS_EBGM`, `AEGIS_LB`, `AEGIS_UB`, `Posterior_Null_Prob`, `Syndrome_Coherence`).
  3. Exposure-normalized Dual-EWMA surge velocity (`Surge_Velocity`) that distinguishes drug-wide media notoriety waves from focal biological acceleration.
  4. Non-asymptotic Gamma-Poisson mixture martingale $e$-values (`e_value`, `p_anytime`, `e_fdr`) and cumulative Bayesian Posterior Error Probability (`fdr`) for anytime-valid sequential surveillance under repeated quarterly peeking.
- **Zero-Inflated Gamma-Poisson Shrinker (`zigps`, `ZIGPSConfig`)**: 3-component Spike-and-Slab Empirical Bayes mixture model ($\pi_0 \delta_0(\lambda) + (1-\pi_0)[P \cdot \text{Gamma}_1 + (1-P) \cdot \text{Gamma}_2]$) with profile/bounded L-BFGS-B optimization, isolating structural drug–event zeros from the continuous relative-risk slab (`Pi0_Structural`, `Slab_Weight`, `EBGM`, `LowerBound`).
- **Sequential Change-Point Surveillance (`maxsprt`, `MaxSPRTConfig`, `cusum_da`, `CUSUMConfig`)**:
  - Kulldorff's Maximized Sequential Probability Ratio Test (`maxsprt`) supporting both Poisson (`model="poisson"`) and conditioned Binomial (`model="binomial"`) log-likelihood ratios, upper-cap truncation (`upper_rr_cap`), and exact Kulldorff critical boundary (`CV_Boundary`, `Max_LLR`, `Anytime_e_value`).
  - Page/Lucas Poisson Cumulative Sum (`cusum_da`) change-point detector with exact reference drift $k = (\theta_1 - \theta_0)/\ln(\theta_1/\theta_0)$ and Siegmund's average-run-length ($\text{ARL}_0$) decision boundary calibration (`CUSUM_Stat`, `Threshold_h`, `Excess_Ratio`).
- Full integration of `aegis`, `zigps`, `maxsprt`, and `cusum_da` across `analyze()`, `consensus_analysis()`, and `LongitudinalModel` (including hyperprior warm-starting for `ZIGPS`).

### Changed
- **SCORE-DA (`score_da`) Mathematical & Calibration Hardening**:
  - Replaced raw Jaccard thresholding in `_build_syndromic_laplacian()` with scale-invariant chance-corrected Jaccard lift and top-$K$ ($K=4$) neighbor sharpening per adverse event so isolated background AEs are not shrunk by dense noise edges.
  - Added LOCO initialization with iterative Winsorization and Huber-clipped Pearson residuals in `_estimate_low_rank_baseline()` to prevent focal signal self-absorption and multi-cell blockbuster masking.
  - Added cooperative syndromic neighbor pooling into the FISTA target and debiased active-support inference (`z_score`, `fdr`).
- **Vectorized Quantile Bisection (`quantile_funcs.py`)**: Added active unconverged index masking (`active = active[not_conv]`) so `scipy.stats.gamma.cdf` only evaluates unconverged indices during iterative root-finding.
- **LASSO (`LASSO.py`) Pre-Allocation**: Hoisted sparse/dense feature matrix assembly (`X_fit`) and contingency cross-product (`X_mat.T @ Y_mat`) outside the per-event regression loop.

### Fixed
- **Extreme Tail Underflow (`common.py`, `LASSO.py`, `SCORE.py`)**: Replaced `1.0 - stats.norm.cdf(z)` with `stats.norm.sf(z)` to prevent IEEE-754 float64 cancellation to `0.0` for $z > 8.29$ (preserving exact $p$-values down to $10^{-308}$) and enforced deterministic `kind="mergesort"` ranking.
- **Fractional Time-Decayed Counts in `RFET` & `SCORE-DA`**: Rounded fractional exponentially decayed counts before evaluating discrete hypergeometric PMF/SF in `RFET.py` and added non-integer count formatting support in `SCORE.py`.
- **Large-Count Log-Space Stability in `GPS.py`**: Rewrote `non_truncated_likelihood`, `truncated_likelihood`, and posterior `Qn` in log-space (`gammaln` + log-sum-exp) to prevent overflow/underflow `NaN`s when cell counts exceed $1{,}000$.

## 3.4.2

### Added
- Continuous exponential time decay (`decay_half_life`) in `LongitudinalModel` (supports duration strings like `'1Y'`, `'365D'`, `'6M'`, or numeric days).

### Changed
- Incremental $O(N)$ accumulation for longitudinal surveillance: pre-factorized string indices and semigroup exponential decay scaling (`C_k = C_{k-1} * 2^(-dt/half_life) + dC_k`).
- True $O(1)$ constant slice memory architecture for sequential `run()` and `run_disjoint()`.
- Lazy task generator streaming for multi-core `n_jobs != 1` execution, capping concurrent memory at $O(\text{n\_jobs})$.
- Vectorized contingency matrix construction using `pd.factorize(sort=True)` and `np.bincount()` in `compute_contingency()`.
- Fast zero-copy panel concatenation in `to_dataframe()`.

### Fixed
- Robust handling of missing (`NaN`) values in `LongitudinalModel` base index preparation without falling back to slow Pandas pivoting.
- Replaced hardcoded nanosecond constant with resolution-agnostic `np.timedelta64(1, 'D')` division.
- Non-deterministic quicksort tie-breaking in PRR and ROR regression tests.

## 0.2.1

### Changed
- All nbinom functions are now scipy native, resulting in massive speed improvements for `gps()`

### Fixed
- `pnbinom()` provided unstable distribution estimates as previously implemented. New implementation resolves this.

## 0.2.0

### Added
- New LASSO option to use an NB-GLM backend with L1 regularization
- Binary conversion forces expansion of summed counts to be compatible with LASSO
- Binary conversion requires `use_counts` flag to skip expansion

### Changed
- Added new sections to the readme for all new features
- setup.py file much cleaner and easy to read/use
- Improved documentation for GPS and LASSO
- Final signal detection for LASSO now based on LASSO threshold parameter

## 0.1.4

### Added
- New LASSO implementation for more DA options
- Exposed the scipy minimization function to user arguments
- Default boundaries for the GPS algorithm
- Data conversion tools for binary matrices and multi-item matrices

### Fixed
- Wrong GPS hyperparameters were being used
- Errors with BCPNN indexing during signal output

## 0.1.3
- Added better error handling in the LBE function to avoid crashing when no objects meet threshold criteria
- Changed PRR argument to accept `fdr_threshold` argument which is now passed to `lbe()`
- Fixed a bug where returned signal indices were strings instead of ints

## 0.1.2
- Added the ability to run a disjoint longitudinal model that does not use cumulative reports during disproportionality analysis
- Exposed a `test_dispersion` function from vigipy.utils that determines dispersion and alpha values for any data

## 0.1.1
- Fixed bug related to bad logic flow when using `signal` as a ranking statistic
- BREAKING: Changed output column names in the signal results to better match function argument conventions

## 0.1.0
- Initial Release

- - - - -
Following [Semantic Versioning](https://semver.org/)
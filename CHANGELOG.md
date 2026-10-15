# Changelog

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
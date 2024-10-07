#Changelog

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
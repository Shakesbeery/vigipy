#Changelog

## 1.3.0
- Added better error handling in the LBE function to avoid crashing when no objects meet threshold criteria
- Changed PRR argument to accept `fdr_threshold` argument which is now passed to `lbe()`
- Fixed a bug where returned signal indices were strings instead of ints

## 1.2.0
- Added the ability to run a disjoint longitudinal model that does not use cumulative reports during disproportionality analysis
- Exposed a `test_dispersion` function from vigipy.utils that determines dispersion and alpha values for any data

## 1.1.0
- Fixed bug related to bad logic flow when using `signal` as a ranking statistic
- BREAKING: Changed output column names in the signal results to better match function argument conventions

## 1.0.0
- Initial Release

- - - - -
Following [Semantic Versioning](https://semver.org/)
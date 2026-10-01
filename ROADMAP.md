# vigipy Development Roadmap & Future Opportunities

This document catalogs identified enhancements, architectural optimizations, and theoretical refinements for `vigipy` that were captured during our audits and deferred for future release cycles.

---

## 1. Statistical Methodology & Numerical Polish

- [ ] **GPS Optimizer Upgrade (L-BFGS-B with Bounds)**:
  - *Context*: Nelder-Mead currently uses soft barrier penalties for GPS Gamma-Poisson mixture priors $(\alpha_1, \beta_1, \alpha_2, \beta_2, w)$.
  - *Opportunity*: Transition from Nelder-Mead to `scipy.optimize.minimize(..., method='L-BFGS-B', bounds=...)` with analytical score equations (gradients), guaranteeing strictly positive hyperparameters without artificial penalty walls.
- [ ] **Syndromic Local False Discovery Rate (lfdr)**:
  - *Context*: Benjamini-Hochberg FDR control in SCORE-DA/DDI operates under the Positive Regression Dependency (PRDS) condition.
  - *Opportunity*: Introduce an optional local false discovery rate (`lfdr`) or Benjamini-Yekutieli (BY) mode for dense adverse event clusters exhibiting high syndromic covariance.
- [ ] **Exact Lancaster Mid-p Confidence Limits for RFET**:
  - *Context*: RFET currently computes exact mid-p p-values but relies on standard Wald or Woolf intervals for the associated odds ratio.
  - *Opportunity*: Implement exact inversion of the hypergeometric distribution to compute exact mid-p confidence intervals for small-cell odds ratios.
- [ ] **Cameron-Trivedi Overdispersion Test on Sparse Marginals**:
  - *Context*: In `expectations.py`, the overdispersion regression can encounter singular matrices when marginal counts have near-zero variance.
  - *Opportunity*: Add ridge-regularized or permutation-based dispersion testing for ultra-sparse contingency slices.

---

## 2. Code Quality, Modular Architecture & Clean Code (Theme 3)

- [ ] **PEP 8 Private Naming Normalization**:
  - *Context*: Several utility functions in `data_prep.py` and `expectations.py` use double leading underscores (e.g. `__expand_dataframe`, `__transform_dataframe`).
  - *Opportunity*: Standardize all internal module functions to single leading underscores (`_*`) per PEP 8 guidelines.
- [ ] **Dead Code Cleanup in `expectations.py`**:
  - *Context*: Legacy helper functions and unused branches from early iterations remain in `expectations.py`.
  - *Opportunity*: Remove dead code paths, simplify marginal sum calculations, and streamline the Poisson/Negative-Binomial GLM interfaces.
- [ ] **Refactor `lbe.py` (Local Bayes Estimation)**:
  - *Context*: `lbe.py` contains legacy single-letter variables (`x`, `y`, `f`, `m`) and deeply nested `if/else` control flow.
  - *Opportunity*: Rename variables to expressive clinical/statistical terms (`observed_counts`, `expected_counts`, `marginal_freq`), extract subroutines, and flatten nested blocks using early guard clauses.
- [ ] **Registry-Driven Dispatch in `analyze.py`**:
  - *Context*: `analyze.py` uses an `if/elif` cascade to route configuration dataclasses to their respective analysis functions.
  - *Opportunity*: Implement a declarative registry lookup dictionary (`CONFIG_REGISTRY: dict[type[MethodConfig], Callable]`) for $O(1)$ dispatch and effortless extension of new methods.

---

## 3. Typing & Package Infrastructure

- [ ] **Comprehensive PEP 484/561 Typing**:
  - *Context*: Core disproportionality modules have complete type signatures, but utility modules and older helpers have partial annotations.
  - *Opportunity*: Add strict type annotations across the entire codebase and ship a `py.typed` marker file so consuming applications (e.g. clinical safety pipelines, Streamlit/Dash dashboards) benefit from type checking.
- [ ] **Directory Hierarchy Modernization (Major Semver)**:
  - *Context*: Modules currently use repeated casing (e.g., `src/vigipy/PRR/PRR.py`, `src/vigipy/GPS/GPS.py`).
  - *Opportunity*: In a future major release (e.g. v4.0), consolidate algorithm files into a clean `src/vigipy/methods/` directory structure with backward-compatible re-exports in `__init__.py`.

---

## 4. Developer Experience & Error Ergonomics

- [ ] **Empathetic Contextual Error Messages (Theme 4, Item 3)**:
  - *Context*: Input validation errors (e.g. missing column names, empty time slices, dimension mismatches) currently throw standard Python `ValueError` or `KeyError`.
  - *Opportunity*: Introduce dedicated domain exceptions (`MissingColumnError`, `EmptySliceError`, `ConfoundedDataError`) with actionable suggestions (e.g. *"Did you mean 'events' instead of 'count'?", "To handle time slices with zero events, set include_gaps=True"*).
- [ ] **Unified Interactive Dashboard / Report Generator**:
  - *Context*: `vigipy` provides tabular exports (`.to_dataframe()`, `.export()`), HTML signal overlap widgets, and console previews.
  - *Opportunity*: Add a lightweight report generator (`vigipy.generate_report(result, format='html')`) to produce self-contained regulatory signal evaluation reports.

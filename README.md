# vigipy

`vigipy` is a Python library bringing modern disproportionality analyses and pharmacovigilance techniques into the Python ecosystem with a clean, intuitive, and type-safe interface. Core disproportionality methods are adapted and extended from Ismail Ahmed and Antoine Poncet's [PhViD](https://cran.r-project.org/web/packages/PhViD/index.html) package, fully vectorized with native NumPy and SciPy routines.

### Top-level Functions & Classes:

* **Unified Interface**:
  * `analyze()` - Execute any disproportionality analysis via typed configuration dataclasses
  * `analyze_all()` - Run multiple analysis methods in a single call with shared or distinct parameters
  * `PRRConfig`, `RORConfig`, `RFETConfig`, `BCPNNConfig`, `GPSConfig`, `LASSOConfig` - Type-safe configuration dataclasses
* **Disproportionality Methods**:
  * `prr()` - Proportional Reporting Ratio (frequentist log-normal approximation)
  * `ror()` - Reporting Odds Ratio (Woolf log-odds approximation)
  * `rfet()` - Reporting Fisher's Exact Test (exact hypergeometric p-values with optional mid-p correction)
  * `bcpnn()` - Bayesian Confidence Propagation Neural Network (analytical or Dirichlet Monte Carlo Information Component)
  * `gps()` - Multi-item Gamma Poisson Shrinker (Empirical Bayes bivariate mixture model)
  * `lasso()` - LASSO regression for multivariate signal detection and confounding adjustment
* **Longitudinal Modeling**:
  * `LongitudinalModel()` - Apply any analysis method over time to evaluate cumulative or disjoint signal evolution
* **Data Preparation**:
  * `convert()` - Convert adverse event and product count tables into a structured `DataContainer`
  * `convert_binary()` - Generate binary product feature matrices and event outcome matrices for LASSO
  * `convert_multi_item()` - Aggregate co-occurring product columns into multi-item interaction tables
* **Result & Data Containers**:
  * `AnalysisResult` - Structured container for `signals`, `all_signals`, `num_signals`, and model `params`, with `.export()` to Excel or CSV
  * `DataContainer` - Typed container holding contingency, event, and product matrices

---

## Getting Started

### Dependencies

`vigipy` requires Python 3.9+ and modern scientific computing libraries:

* `pandas>=2.0`
* `numpy>=1.24,<3`
* `scipy>=1.10`
* `scikit-learn>=1.3`
* `statsmodels>=0.14`

Optional dependencies:
* `openpyxl>=3.0.0` (required for exporting results directly to Excel `.xlsx` spreadsheets)

### Installation

Install `vigipy` from source or local checkout:

```bash
pip install .
```

To include Excel export capabilities:

```bash
pip install ".[excel]"
```

For development (includes test and lint tools):

```bash
pip install -e ".[dev]"
```

### Running Tests

Run the full test suite using `pytest`:

```bash
pytest test/ -v
```

---

## Usage

### Unified API (Recommended)

The unified interface provides type safety, autocompletion, and consistent result structures across all methods:

```python
import pandas as pd
from vigipy import convert, analyze, analyze_all, PRRConfig, BCPNNConfig, GPSConfig

# 1. Load data and convert to a DataContainer
df = pd.read_csv("AE_count_data.csv")
data = convert(df, product_label="name", ae_label="AE", count_label="count")

# 2. Run a single method with typed configuration
result = analyze(data, PRRConfig(min_events=3, decision_metric="fdr", fdr_threshold=0.05))

print(f"Detected {result.num_signals} signals")
print(result.signals.head())

# Export both significant signals and full dataset to Excel or CSV
result.export("prr_signals.xlsx")   # Creates 'Signals' and 'all_data' sheets
result.export("prr_signals.csv")    # Exports detected signals to CSV

# 3. Batch comparison across all methods in one call
batch_results = analyze_all(data, min_events=3, decision_metric="rank")
for method_name, res in batch_results.items():
    print(f"{method_name.upper()}: {res.num_signals} signals detected")

# 4. Iterate over custom configurations
configs = [
    PRRConfig(min_events=5, ranking_statistic="CI"),
    BCPNNConfig(min_events=5, ranking_statistic="quantile"),
    GPSConfig(min_events=5, ranking_statistic="log2"),
]
for cfg in configs:
    res = analyze(data, cfg)
    print(f"{cfg.method}: {res.num_signals} signals")
```

### Classic Function API

Direct function calls are fully supported with identical return structures:

```python
import pandas as pd
from vigipy import convert, prr, ror, rfet, bcpnn, gps

df = pd.read_csv("AE_count_data.csv")
data = convert(df)

# Frequentist: Reporting Odds Ratio with Haldane-Anscombe continuity correction
ror_res = ror(data, min_events=3, decision_metric="fdr", fdr_threshold=0.05)

# Fisher's Exact Test with Lancaster mid-p adjustment
rfet_res = rfet(data, min_events=3, mid_pval=True)

# Bayesian Confidence Propagation Neural Network
bcpnn_res = bcpnn(data, min_events=3, ranking_statistic="quantile")

# Empirical Bayes: Gamma Poisson Shrinker
gps_res = gps(data, min_events=5, decision_metric="rank", ranking_statistic="log2")

# Access results
print(gps_res.signals[["Product", "Adverse Event", "Count", "quantile", "fdr"]].head())
gps_res.export("gps_signals.xlsx")
```

---

## Decision Rules & Ranking Statistics

`vigipy` standardizes signal identification across frequentist and Bayesian methods:

### Decision Metrics (`decision_metric`)
* `"fdr"` - Controls the False Discovery Rate at `decision_thres` (default: `0.05`) using Local Bayes Estimation (LBE) or cumulative posterior null probabilities.
* `"rank"` - Retains signals where the ranking statistic meets `decision_thres`. For p-values, selects values $\le \text{threshold}$; for confidence/credible bounds, selects values $\ge \text{threshold}$.
* `"signals"` - Selects the top $N$ ranked candidates up to `decision_thres`.

### Ranking Statistics (`ranking_statistic`)
| Method | Supported Statistics | Notes |
| :--- | :--- | :--- |
| **PRR / ROR** | `"p_value"`, `"CI"` | `"CI"` ranks by the lower bound of the 95% confidence interval. |
| **RFET** | `"p_value"` | Exact hypergeometric p-value (supports `mid_pval=True`). |
| **BCPNN** | `"quantile"`, `"p_value"` | `"quantile"` ranks by $IC_{025}$ (lower 95% credible bound). |
| **GPS** | `"log2"`, `"quantile"`, `"p_value"` | `"log2"` ranks by $EB_{05}$ of $\log_2(\lambda)$, shrinked towards expected counts. |

---

## Expected Count Calculations & Dispersion Testing

Expected counts ($E$) model the baseline event frequency under the null hypothesis of no association. `vigipy` supports three expectation models:

1. `"mantel-haentzel"` (default): Standard independence assumption, $E_{ij} = \frac{n_{i\cdot} n_{\cdot j}}{N}$.
2. `"poisson"`: Generalized Linear Model using Poisson log-linear regression.
3. `"negative-binomial"`: Generalized Linear Model with negative binomial dispersion parameter `method_alpha`.

When event data exhibits overdispersion (variance significantly exceeds the mean), Poisson estimates may underestimate variance. You can test for overdispersion using Cameron and Trivedi's auxiliary regression test:

```python
import pandas as pd
from vigipy import convert, bcpnn
from vigipy.utils import test_dispersion

df = pd.read_csv("AE_count_data.csv")
data = convert(df)

# Test for overdispersion
dispersion_info = test_dispersion(data)
print(f"Dispersion ratio: {dispersion_info['dispersion']:.2f}")

# If overdispersed (> 2), use the estimated alpha in Negative Binomial regression
alpha = dispersion_info["alpha"] if dispersion_info["dispersion"] > 2 else 1.0
res = bcpnn(data, expected_method="negative-binomial", method_alpha=alpha, min_events=3)
```

---

## Longitudinal Modeling

The `LongitudinalModel` class evaluates disproportionality over time to monitor signal emergence, stability, and trajectory.

You can run models in two modes:
* **Cumulative (`run`)**: Progressively incorporates historical data up to each resampled time boundary, tracking accumulating evidence.
* **Disjoint (`run_disjoint`)**: Evaluates each time window independently without historical accumulation.

```python
import pandas as pd
from vigipy import LongitudinalModel, gps

df = pd.read_csv("AE_time_series.csv")
# Must contain: 'date', 'name', 'AE', and 'count' (or custom count_col)

# Initialize grouped by calendar year ('YE', 'QE', 'ME')
lm = LongitudinalModel(df, time_unit="YE", count_col="count")

# Run GPS cumulatively over time
lm.run(gps, include_gaps=False, decision_metric="rank", ranking_statistic="log2")

# Regroup to quarterly slices and evaluate
lm.regroup_dates("QE")
lm.run_disjoint(gps, include_gaps=False, decision_metric="rank", ranking_statistic="log2")

# Access results chronologically: (timestamp, AnalysisResult)
for timestamp, result in lm.results:
    if result is not None:
        print(f"Slice ending {timestamp.date()}: {result.num_signals} signals")
        print(result.signals.head(2))
```

---

## LASSO Signal Detection

LASSO regression models multiple products simultaneously, adjusting for co-prescriptions, confounding by indication, and polypharmacy.

### 1. Pure Binary Matrix
Each row represents an individual report or subject:

```python
import pandas as pd
from vigipy import convert_binary, lasso

df = pd.read_csv("patient_reports.csv")
bin_data = convert_binary(df, product_label="name", ae_label="AE", use_counts=False)

# Linear LASSO with Information Criterion model selection
result = lasso(bin_data, use_IC=True, IC_criterion="bic", min_events=3)
print(result.signals[["Product", "Adverse Event", "LASSO Coefficient", "CI Lower", "CI Upper"]])
result.export("lasso_signals.xlsx")
```

### 2. Count Outcomes & GLM LASSO
When aggregate event counts are used:

```python
bin_data = convert_binary(df, product_label="name", ae_label="AE", use_counts=True)

# Fit Negative Binomial GLM with L1 regularization
result = lasso(bin_data, use_glm=True, lasso_thresh=0.2, nb_alpha=1.0)
result.export("lasso_glm_signals.csv")
```

---

## Multi-Item Interaction Conversion

To analyze interactions between co-occurring drugs or devices:

```python
from vigipy.utils.data_prep import convert_multi_item

# Aggregate co-administered products
multi_data = convert_multi_item(
    df,
    product_label=["suspect_drug_1", "suspect_drug_2"],
    ae_label="AE",
    count_label="count",
    min_threshold=3,
)
```

The returned `DataContainer` is fully compatible with `prr`, `ror`, `rfet`, `bcpnn`, and `gps`.

---

## Result Inspection & Export

All analysis methods return an `AnalysisResult` object:

```python
result = analyze(data, PRRConfig(min_events=3))

# Filtered signals meeting decision criteria
signals_df = result.signals

# All evaluated candidate pairs with computed statistics
all_df = result.all_signals

# Number of identified signals
count = result.num_signals

# Input parameters and model metadata
params_dict = result.params

# Export to Excel (.xlsx) or CSV (.csv)
result.export("output.xlsx")  # Writes 'Signals' and 'all_data' sheets
result.export("output.csv")   # Writes signals DataFrame
```

---

## Authors

* **David Beery** ([@Shakesbeery](https://github.com/Shakesbeery))

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

* **Ismail Ahmed and Antoine Poncet** for the foundational design of the [PhViD](https://cran.r-project.org/web/packages/PhViD/index.html) package in R.
* **Ross Ihaka** and **Catherine Loader** for early mathematical formulations of deviance and log-gamma approximations.

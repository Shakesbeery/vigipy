# vigipy: Pharmacovigilance & Disproportionality Analysis in Python

[![PyPI version](https://img.shields.io/pypi/v/vigipy.svg)](https://pypi.org/project/vigipy/)
[![Python versions](https://img.shields.io/pypi/pyversions/vigipy.svg)](https://pypi.org/project/vigipy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!IMPORTANT]
> **What's New in `vigipy`**
> - **SCORE-DA & SCORE-DDI**: A novel syndromic outlier estimation framework combining low-rank indication absorption, patient-level Graph Laplacian regularization, non-negative FISTA optimization, and higher-order multi-drug interaction discovery (`max_order=2, 3, ...`).
> - **Cross-Method Consensus Engine (`consensus_analysis`)**: Triangulate alerts across frequentist, Bayesian, regression, and syndromic methods with agreement tiers, vote tallying, and concordance analytics (Jaccard, Cohen's Kappa, Spearman).
> - **Two-Stage Relaxed LASSO**: Unbiased adjusted reporting odds ratios ($\text{aROR}$) with sparse memory scaling and clinical confounder adjustment (e.g. Age, Sex).
> - **Production-Grade Longitudinal Pipeline**: Multi-core time-slice parallelism, closed-form analytical GPS likelihoods, and hyperprior warm-starting.

---

## The Disproportionality Analysis (DA) Lifecycle

Safety surveillance in spontaneous reporting databases (e.g., FDA FAERS, WHO VigiBase, MAUDE) follows a 4-stage lifecycle. `vigipy` provides modular, principled tools for each step:

```
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA INGESTION & PREPARATION                                 │
│  Convert raw tables into typed DataContainers (Sparse, Binary, DDI)    │
│  [ convert()  │  convert_binary()  │  convert_ddi() ]                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: STATISTICAL MODELING & SIGNAL DETECTION                      │
│  Screen for disproportionate drug-event associations via analyze()     │
│  • Frequentist:      PRR, ROR, RFET                                    │
│  • Bayesian:         GPS (Gamma Poisson), BCPNN (Information Component)│
│  • Multivariable:    Relaxed Logistic LASSO (Confounder Adjustment)    │
│  • Syndromic:        SCORE-DA (Indication SVD + Graph Regularization)  │
│  • Multi-Drug/DDI:   SCORE-DDI (Higher-Order Regimen Synergy)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐   ┌────────────────────────────────┐
│  STAGE 3: MULTI-METHOD CONSENSUS  │   │  STAGE 4: TEMPORAL MONITORING  │
│  Triangulate alerts across models │   │  Track signal emergence over   │
│  • Vote tallying & Consensus score│   │  time across quarterly/annual  │
│  • Agreement tiers & Kappa stats  │   │  reporting periods             │
│  • Signal inspection & drill-down │   │  • Cumulative vs. Disjoint     │
│  [ consensus_analysis() ]         │   │  [ LongitudinalModel ]         │
└───────────────────────────────────┘   └────────────────────────────────┘
```

---

## Installation & Setup

### Requirements
* Python 3.9+
* `numpy>=1.24`, `scipy>=1.10`, `pandas>=2.0`, `scikit-learn>=1.3`, `statsmodels>=0.14`
* Optional: `openpyxl>=3.0` (for Excel `.xlsx` multi-sheet reports)

```bash
# Standard installation
pip install vigipy

# Install with Excel export support
pip install "vigipy[excel]"
```

Run tests to verify installation:
```bash
pytest
```

---

## Stage 1: Data Preparation & Ingestion

Pharmacovigilance datasets come in different formats: pre-aggregated frequency counts, case-level binary reports, or multi-drug polypharmacy regimens. Choosing the right preprocessor ensures optimal statistical power and memory efficiency.

### 1. `convert()`: For Summary Contingency Tables
* **When to use**: Your data is already aggregated into counts of `(Product, Adverse Event, Count)`.
* **Compatible methods**: PRR, ROR, RFET, GPS, BCPNN, SCORE-DA.

```python
import pandas as pd
import vigipy as vg

df = pd.read_csv("contingency_counts.csv")
# Columns: 'product_name', 'adverse_event', 'count'

data = vg.convert(
    df,
    product_label="product_name",
    ae_label="adverse_event",
    count_label="count",
    margin_threshold=3,  # Filter out rare events with total counts < 3
)
```

### 2. `convert_binary()`: For Case-Level Reports & Confounder Adjustment
* **When to use**: You have individual patient report IDs (`report_id`). Multiple medications and symptoms can appear on the same report.
* **Why it matters**: 
  - Required for **LASSO** (evaluates all concurrent medications simultaneously).
  - Unlocks **SCORE-DA's syndromic graph**, ensuring symptom co-occurrence reflects true patient-level clinical syndromes rather than aggregate product correlation.
  - Supports **confounder adjustment** (e.g. Age, Sex).
  - Uses `sparse=True` (CSR matrix storage) to handle millions of reports with low memory usage.

```python
df_reports = pd.read_csv("faers_case_reports.csv")
# Columns: 'report_id', 'drug_name', 'reaction', 'age', 'sex'

binary_data = vg.convert_binary(
    df_reports,
    product_label="drug_name",
    ae_label="reaction",
    report_id_label="report_id",
    covariate_labels=["age", "sex"],  # Standardized continuous, dummy-encoded categorical
    sparse=True,                      # Memory-efficient sparse CSR matrices
)
```

### 3. `convert_ddi()`: For Drug-Drug & Multi-Drug Interactions
* **When to use**: You want to screen for pairwise drug-drug interactions ($k=2$) or higher-order multi-drug regimens ($k=3, \dots$).
* **How it works**: Uses sparse matrix intersection ($\mathbf{X}^\top \mathbf{X}$) to automatically find drug combinations meeting `min_co_reports`, avoiding combinatorial blowup.
* **Universal compatibility**: Produces a standard `DataContainer` containing both individual drug baselines and combination entities, allowing **any method in `vigipy`** to evaluate combinations.

```python
ddi_container = vg.convert_ddi(
    df_reports,
    product_label="drug_name",
    ae_label="reaction",
    report_id_label="report_id",
    min_co_reports=5,     # Drug combo must appear together on >= 5 reports
    max_order=3,          # Screen pairs (k=2) AND triplets (k=3)
    include_singles=True, # Retain single drugs as reference baselines
    sparse=True,
)
```

---

## Stage 2: Signal Detection Modeling

`vigipy` provides a unified interface (`analyze()`) across all modeling paradigms. Each paradigm addresses specific clinical and statistical questions.

### Summary of Signal Detection Methods

| Method | Paradigm | Primary Statistic | Best Used For |
| :--- | :--- | :--- | :--- |
| **PRR** | Frequentist | Proportional Reporting Ratio | Regulatory baseline, transparent proportional ratios |
| **ROR** | Frequentist | Reporting Odds Ratio | Regulatory compliance (Woolf log-odds CI), matching case-control logic |
| **RFET** | Frequentist | Mid-p Fisher's Exact Test | Ultra-sparse cells ($n \le 3$), eliminating normal approximation error |
| **GPS** | Empirical Bayes | EBGM ($EB_{05}$ quantile) | Large database screening; stabilizes small-count variance |
| **BCPNN** | Empirical Bayes | Information Component ($IC_{025}$) | Early warning surveillance; neural/Dirichlet credibility intervals |
| **LASSO** | Regularized GLM | Adjusted Odds Ratio ($\text{aROR}$) | Confounder adjustment; removing polypharmacy attribution noise |
| **SCORE-DA** | Syndromic / Matrix | Syndromic Excess Rate ($\text{SER}$) | Eliminating indication confounding & masking; syndromic borrowing |
| **SCORE-DDI** | Factorial / Graph | Synergy Excess Rate ($\text{SER}_{\text{int}}$) | True synergy beyond single-drug solo risks; multi-drug regimens |

---

### The Unified API (`analyze`)

Configure methods with type-safe dataclasses:

```python
from vigipy import (
    analyze,
    PRRConfig,
    RORConfig,
    GPSConfig,
    LASSOConfig,
    SCOREConfig,
    SCOREDDIConfig,
)

# 1. Classical Frequentist (PRR with False Discovery Rate control)
prr_res = analyze(data, PRRConfig(min_events=3, decision_metric="fdr", fdr_threshold=0.05))

# 2. Empirical Bayes Shrinkage (GPS ranked by EB05)
gps_res = analyze(data, GPSConfig(min_events=5, ranking_statistic="quantile"))

# 3. Multivariable Confounder-Adjusted LASSO
lasso_res = analyze(binary_data, LASSOConfig(min_events=3, relaxed=True, n_jobs=-1))

# 4. Syndromic Low-Rank Discovery (SCORE-DA)
score_res = analyze(data, SCOREConfig(latent_rank=5, syndromic_weight=0.5, fdr_threshold=0.05))

# 5. Multi-Drug Interaction Discovery (SCORE-DDI)
ddi_res = analyze(ddi_container, SCOREDDIConfig(interaction_model="multiplicative", min_events=3))
```

---

### Method Deep-Dives

#### Multivariable Relaxed LASSO
When patients take multiple drugs, single-drug methods suffer from **confounding by co-prescription** (e.g., antiemetics falsely flagged for chemotherapy toxicities).
* `vigipy.lasso()` fits a high-dimensional regularized logistic regression across all drugs and covariates simultaneously.
* By default (`relaxed=True`), it runs a **two-stage Relaxed LASSO**: Stage 1 screens active features; Stage 2 refits an unpenalized model on active features using SVD pseudo-inverse Wald standard errors to return **debiased adjusted Reporting Odds Ratios ($\text{aROR}$)**.

```python
result = vg.lasso(
    binary_data,
    min_events=3,
    relaxed=True,                   # Debiased relaxed refit
    decision_metric="lower_bound",  # Signal if 95% CI lower bound > 1.0
    n_jobs=-1,                      # Parallel across adverse events
)
print(result.signals[["Product", "Adverse Event", "Count", "aROR", "CI Lower", "CI Upper", "p_value"]])
```

#### SCORE-DA: Syndromic Cellwise Outlier & Residual Estimation
Traditional disproportionality methods assume independence across symptom columns and suffer from **blockbuster masking** and **indication confounding**.
* **Indication Absorption**: Uses Truncated SVD on standardized Pearson residuals to absorb shared drug-class and indication baselines.
* **Syndromic Borrowing**: Builds a patient-level Graph Laplacian ($\mathbf{L}_{\text{AE}}$) from co-occurring symptoms, allowing related events in a syndrome (e.g. *Urticaria + Angioedema + Hypotension*) to borrow strength without relying on external ontologies.
* **Masking Deflation**: Iteratively deflates detected signals to unmask hidden safety signals suppressed by blockbuster drugs.

```python
score_res = vg.score_da(
    data,
    latent_rank=5,           # Number of latent indication/class factors to absorb
    syndromic_weight=0.5,    # Graph Laplacian smoothness coupling
    deflate_iterations=2,    # Iterative deflation passes to eliminate masking
    fdr_threshold=0.05,      # Benjamini-Hochberg FDR cutoff
)
print(score_res.signals[["Product", "Adverse Event", "Count", "SER", "SRR", "fdr", "Syndrome_Cluster"]])
```

#### SCORE-DDI: Drug-Drug & Multi-Drug Interaction Discovery
* **Unbiased Solo Baselines**: When evaluating combinations, `score_ddi` subtracts co-prescription counts ($C_1 - C_{\text{combo}}$) so the combination's toxicity cannot artificially inflate the single-drug baseline.
* **Higher-Order Regimens**: Evaluates pairs ($k=2$), triplets ($k=3$), and custom regimens.
* **Epidemiological Archetype Classification**:
  - `EMERGENT`: Toxicity appears exclusively upon combination (neither drug active alone).
  - `POTENTIATED`: One drug has baseline activity; adding the second drug significantly magnifies risk.
  - `TWO_HIT` / `MULTI_HIT`: Multiple constituent drugs elevate risk individually; combination triggers compound injury.

```python
ddi_res = vg.score_ddi(
    ddi_container,
    interaction_model="multiplicative",  # or "additive"
    syndromic_weight=0.5,
    min_events=3,
    fdr_threshold=0.05,
)
print(ddi_res.signals[[
    "Components", "Order", "Product", "Adverse Event", "Count",
    "Expected_Null", "SER_Interaction", "DDI_Ratio", "Interaction_Archetype"
]])
```

---

## Stage 3: Triangulation & Cross-Method Consensus

Different models possess distinct biases: frequentist ratios are noisy on small counts; Bayesian shrinkage can be conservative on rare catastrophic reactions; regression models can be sensitive to collinearity.

`consensus_analysis()` synthesizes findings across multiple methods in a single unified call:
* **Vote Tallying & Consensus Scoring**: Normalizes alert votes across methods into composite scores and categorizes signals into agreement tiers (`Unanimous`, `Strong`, `Moderate`, `Weak`, `Isolated`).
* **Inter-Method Concordance Analytics**: Computes pairwise Jaccard similarity, Cohen's Kappa concordance, Spearman rank correlation, and $2 \times 2$ alert overlap matrices.
* **Signal Drill-Down**: `inspect_signal()` provides an immediate side-by-side comparison of every method's score, interval, and alert status for any candidate pair.

```python
# Run consensus across default regulatory methods (or pass custom configs)
configs = [
    vg.PRRConfig(min_events=3),
    vg.GPSConfig(min_events=3),
    vg.LASSOConfig(min_events=3),
    vg.SCOREConfig(min_events=3),
]

consensus = vg.consensus_analysis(
    binary_data,
    configs=configs,
    min_consensus=3,  # Retain signals alerted by >= 3 methods
)

# View top consensus signals
print(consensus.signals[[
    "Product", "Adverse Event", "Count", "votes", "consensus_score", "agreement_tier"
]].head())

# Inspect a specific signal across all methods
detail = consensus.inspect_signal("DRUG_A", "ACUTE_KIDNEY_INJURY")
print(detail)
# Displays Method, Alert, Metric, Score, CI Lower, CI Upper, p-value, FDR, Count

# Inter-method agreement metrics
print("Cohen's Kappa Concordance Matrix:")
print(consensus.method_agreement["kappa"].round(3))

# Export multi-sheet consensus report to Excel
consensus.export("consensus_report.xlsx")
```

---

## Stage 4: Temporal & Longitudinal Surveillance

Post-market safety surveillance requires monitoring how disproportionality metrics evolve over time. The `LongitudinalModel` applies any signal detection algorithm across resampled time slices.

* **Cumulative Mode (`run`)**: Progressively incorporates historical data up to each time boundary, tracking accumulating evidence and detecting the exact calendar date an emerging signal crosses threshold.
* **Disjoint Mode (`run_disjoint`)**: Evaluates each time window independently (e.g. quarterly or annually) without historical accumulation. Ideal for detecting transient reporting anomalies, batch contaminations, or media-driven notoriety effects.

```python
from vigipy import LongitudinalModel, gps

df_time = pd.read_csv("longitudinal_safety_data.csv")
# Requires: 'date', 'name', 'AE', 'count'

# Initialize with annual ('YE'), quarterly ('QE'), or monthly ('ME') slices
lm = LongitudinalModel(df_time, time_unit="QE", count_col="count")

# Run GPS cumulatively with hyperprior warm-starting across time slices
lm.run(gps, warm_start=True, store_all_signals=False)

# Or evaluate disjoint quarterly intervals in parallel across CPU cores
lm.run_disjoint(gps, n_jobs=-1)

# Inspect chronological signal evolution
for timestamp, result in lm.results:
    if result is not None:
        print(f"Quarter ending {timestamp.date()}: {result.num_signals} active signals")
        print(result.signals[["Product", "Adverse Event", "Count", "quantile"]].head(2))
```

---

## Decision Guide: Which Tool Should I Use?

| Scenario | Recommended Workflow | Key Parameters |
| :--- | :--- | :--- |
| **Routine regulatory submission (FDA / EMA)** | `vg.convert()` $\rightarrow$ `vg.ror()` or `vg.prr()` | `min_events=3`, `ranking_statistic="CI"` |
| **Automated screening across large database** | `vg.convert()` $\rightarrow$ `vg.gps()` or `vg.bcpnn()` | `ranking_statistic="quantile"`, `decision_metric="fdr"` |
| **High polypharmacy / co-prescription confounding** | `vg.convert_binary(sparse=True)` $\rightarrow$ `vg.lasso()` | `relaxed=True`, `covariate_labels=["age", "sex"]` |
| **Indication confounding & symptom clustering** | `vg.convert_binary()` $\rightarrow$ `vg.score_da()` | `latent_rank=5`, `syndromic_weight=0.5` |
| **Investigating drug-drug or multi-drug interactions** | `vg.convert_ddi(max_order=2)` $\rightarrow$ `vg.score_ddi()` | `min_co_reports=5`, `interaction_model="multiplicative"` |
| **Multi-method signal arbitration & triage** | `vg.consensus_analysis()` | `min_consensus=3` or `min_consensus=0.6` |
| **Monitoring signal emergence over time** | `vg.LongitudinalModel()` | `time_unit="QE"`, `warm_start=True` |

---

## Result Exporting & Reporting

All analysis methods return structured `AnalysisResult` objects with built-in export utilities:

```python
result = vg.analyze(data, vg.GPSConfig(min_events=3))

# Access primary attributes
signals_df = result.signals       # Filtered signals meeting decision criteria
all_df = result.all_signals       # Full candidate dataset with all computed metrics
num_alerts = result.num_signals   # Number of detected signals
model_meta = result.params        # Audit trail of parameters used

# Export to Excel (.xlsx) or CSV (.csv)
result.export("safety_audit.xlsx")  # Creates 'Signals' and 'all_data' tabs
result.export("signals_only.csv")   # Exports detected signals to CSV
```

---

## Authors & Citation

* **David Beery** ([@Shakesbeery](https://github.com/Shakesbeery))

### Citation
If you use `vigipy` in your research or regulatory surveillance pipelines, please cite:
```bibtex
@software{beery2026vigipy,
  author = {David Beery},
  title = {vigipy: Disproportionality Analysis and Pharmacovigilance Toolkit in Python},
  year = {2026},
  url = {https://github.com/Shakesbeery/vigipy}
}
```

### License
`vigipy` is distributed under the [MIT License](LICENSE).

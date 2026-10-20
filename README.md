# vigipy: Pharmacovigilance & Disproportionality Analysis in Python

[![PyPI version](https://img.shields.io/pypi/v/vigipy.svg)](https://pypi.org/project/vigipy/)
[![Python versions](https://img.shields.io/pypi/pyversions/vigipy.svg)](https://pypi.org/project/vigipy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!IMPORTANT]
> **What's New in `vigipy` (v3.5.0)**
> - **AEGIS (`aegis` / `AEGISConfig`)**: **A**nytime **E**mpirical **G**raph-**I**nformed **S**hrinkage — a state-of-the-art 4-pillar pharmacovigilance engine uniting Leave-One-Cell-Out (LOCO) Sinkhorn-Huber blockbuster unmasking, top-$K$ NPMI syndromic graph coherence, exposure-normalized Dual-EWMA surge velocity (immune to media notoriety waves), and anytime-valid mixture martingale $e$-values (`e_value`, `AEGIS_EBGM`, `AEGIS_LB`).
> - **Zero-Inflated Gamma-Poisson Shrinker (`zigps` / `ZIGPSConfig`)**: 3-component Spike-and-Slab Empirical Bayes mixture ($\pi_0 \delta_0(\lambda) + (1-\pi_0)[P \cdot \text{Gamma}_1 + (1-P) \cdot \text{Gamma}_2]$) isolating structural drug–event zeros from the continuous Gamma slab.
> - **Sequential Change-Point Surveillance (`maxsprt` / `cusum_da`)**: Kulldorff's Maximized Sequential Probability Ratio Test (Poisson & Binomial models) and Lucas/Page Poisson log-likelihood-ratio CUSUM with Siegmund $\text{ARL}_0$ calibration.
> - **SCORE-DA & Core Numerical Hardening**: Chance-corrected top-$K$ Jaccard graph sharpening, iterative LOCO Winsorization + Huber-clipped SVD, debiased FISTA inference, and log-space / survival-function precision across `GPS`, `RFET`, `LASSO`, and `common`.
> - **High-Performance Longitudinal Engine & Time Decay (v3.4.2)**: Continuous exponential half-life time decay (`decay_half_life`), $O(N)$ incremental accumulation via semigroup scaling, and $O(1)$ constant-memory surveillance.

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
│  • Bayesian:         GPS, ZIGPS (Zero-Inflated), BCPNN                 │
│  • Sequential:       MaxSPRT (Kulldorff LLR), CUSUM (Page/Lucas)       │
│  • Multivariable:    Relaxed Logistic LASSO (Confounder Adjustment)    │
│  • Syndromic:        SCORE-DA (LOCO + Huber SVD + Graph FISTA)         │
│  • Next-Gen Unified: AEGIS (Unmasking + NPMI Graph + Velocity + e-Val) │
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
* **Compatible methods**: PRR, ROR, RFET, GPS, ZIGPS, BCPNN, MaxSPRT, CUSUM, SCORE-DA, AEGIS.

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
  - Unlocks **AEGIS** and **SCORE-DA** patient-level syndromic graphs (`Y.T @ Y`) and **AEGIS** chronological surge velocity (`Surge_Velocity`).
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
| **ZIGPS** | Zero-Inflated EB | Spike-and-Slab EBGM ($EB_{05}$) | High-sparsity tables (>80% structural zeros); prevents prior flattening |
| **BCPNN** | Empirical Bayes | Information Component ($IC_{025}$) | Early warning surveillance; neural/Dirichlet credibility intervals |
| **MaxSPRT** | Sequential LLR | Maximized LLR vs. $CV(\alpha)$ | Continuous vaccine/drug monitoring with exact upper-spending boundaries |
| **CUSUM** | Change-Point | Page/Lucas Poisson CUSUM ($S_t$) | Detecting sudden regime shifts calibrated to target $\text{ARL}_0$ |
| **LASSO** | Regularized GLM | Adjusted Odds Ratio ($\text{aROR}$) | Confounder adjustment; removing polypharmacy attribution noise |
| **SCORE-DA** | Syndromic / Matrix | Syndromic Excess Rate ($\text{SER}$) | Ultra-fast frequentist screening with LOCO deflation & graph borrowing |
| **AEGIS** | Graph-Gated EB + Martingale | $\text{AEGIS\_EBGM}$ ($\text{AEGIS\_LB}$, $e\text{-value}$) | Early syndromic detection, blockbuster unmasking, notoriety immunity & anytime-valid sequential peeking |
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
    ZIGPSConfig,
    MaxSPRTConfig,
    CUSUMConfig,
    LASSOConfig,
    SCOREConfig,
    AEGISConfig,
    SCOREDDIConfig,
)

# 1. Next-Generation Graph-Informed Empirical Bayes & Martingale Surveillance (AEGIS)
aegis_res = analyze(binary_data, AEGISConfig(min_events=3, fdr_threshold=0.05, lb_threshold=1.05))

# 2. Zero-Inflated Gamma-Poisson Shrinker (ZIGPS)
zigps_res = analyze(data, ZIGPSConfig(min_events=3, ranking_statistic="quantile"))

# 3. Sequential Change-Point Surveillance (MaxSPRT & CUSUM)
sprt_res = analyze(data, MaxSPRTConfig(model="poisson", alpha=0.05, min_events=2))
cusum_res = analyze(data, CUSUMConfig(target_rr=2.0, arl0=500.0, min_events=2))

# 4. Multivariable Confounder-Adjusted LASSO
lasso_res = analyze(binary_data, LASSOConfig(min_events=3, relaxed=True, n_jobs=-1))

# 5. Syndromic Low-Rank Discovery (SCORE-DA)
score_res = analyze(data, SCOREConfig(latent_rank=5, syndromic_weight=0.5, fdr_threshold=0.05))

# 6. Multi-Drug Interaction Discovery (SCORE-DDI)
ddi_res = analyze(ddi_container, SCOREDDIConfig(interaction_model="multiplicative", min_events=3))
```

---

### Method Deep-Dives

#### AEGIS: Anytime Empirical Graph-Informed Shrinkage
Classical Empirical Bayes shrinkers (`GPS`, `BCPNN`) evaluate each drug–event cell in isolation—shrinking emerging signals when counts are small ($n = 2\text{–}5$) while remaining vulnerable to blockbuster masking, media-driven notoriety surges, and sequential peeking inflation. `vigipy.aegis()` addresses all four failure modes:
1. **Counterfactual LOCO Sinkhorn-Huber Baseline**: Initializes Leave-One-Cell-Out (LOCO) marginals in $O(1)$ operations to break self-cloaking fixed points, then iteratively Winsorizes extreme spikes with analytical Gaussian upper-tail bias correction ($\phi(\delta) - \delta \Phi(-\delta)$) and applies Huber-clipped SVD to absorb diffuse indication confounding (`B_Counterfactual`, `Unmasking_Ratio`).
2. **NPMI Syndromic Graph & Conjugate Poisson-Gamma Pooling**: Constructs a degree-regularized, top-$K$ pruned Normalized Pointwise Mutual Information (NPMI) symptom graph. When connected symptoms in a clinical syndrome co-activate on the same drug, a logistic context gate elevates the cell-specific signal prior $\pi_{ji}$ and pools conjugate pseudo-observations $(\Delta C_{ji}, \Delta B_{ji})$ into both the Gamma shape and rate parameters (`AEGIS_EBGM`, `AEGIS_LB`, `Syndromic_Coherence`).
3. **Exposure-Normalized Dual-EWMA Surge Velocity**: Applies a continuous exponential moving kernel over the report sequence and normalizes recent counts against the drug's recent reporting share $C_{\text{fast}, ji} / R_{\text{fast}, j}$. Drug-wide media notoriety waves inflate $C_{ji}$ and $R_j$ proportionally (yielding `Surge_Velocity ≈ 0`), whereas focal biological hazards exhibit strong positive velocity.
4. **Anytime-Valid Vovk–Wang Product Martingale $e$-Values**: Evaluates Wald-Poisson mixture martingales on true counts $(C_{ji}, B_{ji})$ and couples them with leave-one-out neighborhood $e$-values via Vovk & Wang's (2021) product martingale identity (`e_value`, `anytime_p_value`, `e_fdr`), satisfying Ville's inequality $\mathbb{P}_0(\exists t : E_{ji}(t) \ge 1/\alpha) \le \alpha$ under continuous quarterly peeking.

```python
aegis_res = vg.aegis(
    binary_data,
    min_events=3,
    fdr_threshold=0.05,        # Bayesian PEP or sequential e-BH FDR cutoff
    lb_threshold=1.05,         # 5th percentile posterior relative risk cutoff (AEGIS_LB)
    decision_metric="combined",# "combined", "fdr", "lb", or "e_value" (sequential)
    latent_rank=2,             # Low-rank SVD indication filter on Winsorized residuals
)
print(aegis_res.signals[[
    "Product", "Adverse Event", "Count", "B_Counterfactual",
    "AEGIS_EBGM", "AEGIS_LB", "Surge_Velocity", "e_value", "fdr"
]])
```

#### ZIGPS: Zero-Inflated Gamma-Poisson Shrinker
In large spontaneous reporting matrices where $>80\%$ of drug–event cells are zero (due to non-overlapping clinical populations or impossible indication pairs), fitting a standard 2-Gamma prior (`GPS`) forces the lower Gamma component toward zero, artificially inflating posterior relative risks on small counts. `vigipy.zigps()` introduces an explicit point-mass spike $\pi_0 \delta_0(\lambda)$ alongside the two-component Gamma slab.

```python
zigps_res = vg.zigps(
    data,
    min_events=3,
    decision_metric="fdr",
    ranking_statistic="quantile",
)
print(zigps_res.signals[["Product", "Adverse Event", "Count", "EBGM", "LowerBound", "Pi0_Structural", "fdr"]])
```

#### Sequential Surveillance: MaxSPRT & CUSUM
For continuous vaccine and post-market drug monitoring:
* `vigipy.maxsprt()`: Implements Kulldorff's Maximized Sequential Probability Ratio Test (`model="poisson"` or `"binomial"`) with exact Kulldorff critical boundary lookup (`CV_Boundary`) and upper relative-risk cap (`upper_rr_cap`).
* `vigipy.cusum_da()`: Implements Lucas/Page's Poisson log-likelihood-ratio CUSUM with reference drift $k = (\theta_1 - \theta_0)/\ln(\theta_1/\theta_0)$ and Siegmund's average-run-length ($\text{ARL}_0$) threshold calibration.

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
* **Indication Absorption**: Uses LOCO-initialized iterative Winsorization and Huber-clipped Truncated SVD on standardized Pearson residuals to absorb shared drug-class and indication baselines without absorbing focal signals.
* **Syndromic Borrowing**: Builds a chance-corrected, top-$K$ sharpened Graph Laplacian ($\mathbf{L}_{\text{AE}}$) from co-occurring symptoms, allowing related events in a syndrome (e.g. *Urticaria + Angioedema + Hypotension*) to borrow strength without relying on external ontologies.
* **Masking Deflation**: Iteratively deflates detected signals to unmask hidden safety signals suppressed by blockbuster drugs.

```python
score_res = vg.score_da(
    data,
    latent_rank=5,           # Number of latent indication/class factors to absorb
    syndromic_weight=0.5,    # Graph Laplacian smoothness & cooperative coupling
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
    vg.ZIGPSConfig(min_events=3),
    vg.LASSOConfig(min_events=3),
    vg.SCOREConfig(min_events=3),
    vg.AEGISConfig(min_events=3),
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
* **Continuous Exponential Time Decay (`decay_half_life`)**: Historical reports naturally attenuate over time as $2^{-\Delta t / \lambda}$ unless reinforced by recent reports, eliminating stale historical notoriety bursts. Computed in true $O(N)$ time via semigroup scaling.
* **Memory-Bounded Streaming**: Evaluates each time slice in constant $O(1)$ memory, preventing OOM crashes across hundreds of reporting periods.

```python
from vigipy import LongitudinalModel, gps

df_time = pd.read_csv("longitudinal_safety_data.csv")
# Requires: 'date', 'name', 'AE', 'count'

# Initialize with annual ('YE'), quarterly ('QE'), or monthly ('ME') slices
lm = LongitudinalModel(df_time, time_unit="QE", count_col="count")

# Run GPS cumulatively with exponential time decay (e.g. 1-year half-life) and hyperprior warm-starting
lm.run(gps, decay_half_life="365D", warm_start=True, store_all_signals=False)

# Or evaluate disjoint quarterly intervals in parallel across CPU cores
lm.run_disjoint(gps, n_jobs=-1)

# Export complete chronological panel across all slices into a tidy DataFrame
df_panel = lm.to_dataframe(which="signals")
print(df_panel.head())
```

---

## Decision Guide: Which Tool Should I Use?

| Scenario | Recommended Workflow | Key Parameters |
| :--- | :--- | :--- |
| **Routine regulatory submission (FDA / EMA)** | `vg.convert()` $\rightarrow$ `vg.ror()` or `vg.prr()` | `min_events=3`, `ranking_statistic="CI"` |
| **Automated screening across large database** | `vg.convert()` $\rightarrow$ `vg.zigps()`, `vg.gps()`, or `vg.bcpnn()` | `ranking_statistic="quantile"`, `decision_metric="fdr"` |
| **Early syndromic signals, blockbuster unmasking & media notoriety immunity** | `vg.convert_binary()` $\rightarrow$ `vg.aegis()` | `fdr_threshold=0.05`, `lb_threshold=1.05`, `decision_metric="combined"` |
| **Continuous sequential surveillance (repeated quarterly peeking)** | `vg.aegis(decision_metric="e_value")`, `vg.maxsprt()`, or `vg.cusum_da()` | `alpha=0.05` / `arl0=500.0` |
| **High polypharmacy / co-prescription confounding** | `vg.convert_binary(sparse=True)` $\rightarrow$ `vg.lasso()` | `relaxed=True`, `covariate_labels=["age", "sex"]` |
| **Ultra-fast frequentist syndromic screening** | `vg.convert_binary()` $\rightarrow$ `vg.score_da()` | `latent_rank=5`, `syndromic_weight=0.5` |
| **Investigating drug-drug or multi-drug interactions** | `vg.convert_ddi(max_order=2)` $\rightarrow$ `vg.score_ddi()` | `min_co_reports=5`, `interaction_model="multiplicative"` |
| **Multi-method signal arbitration & triage** | `vg.consensus_analysis()` | `min_consensus=3` or `min_consensus=0.6` |
| **Monitoring signal emergence over time** | `vg.LongitudinalModel()` | `time_unit="QE"`, `warm_start=True`, `decay_half_life="365D"` |

---

## Result Exporting, Serialization & Logging

### Structured Result Container (`AnalysisResult` & `ConsensusResult`)
All analysis methods return structured `AnalysisResult` (or `ConsensusResult`) objects with built-in export utilities supporting Excel, Parquet, and CSV:

```python
from pathlib import Path
import vigipy as vg

result = vg.analyze(data, vg.GPSConfig(min_events=3))

# Access primary attributes
signals_df = result.signals       # Filtered signals meeting decision criteria
all_df = result.all_signals       # Full candidate dataset with all computed metrics (includes EBGM, LowerBound, UpperBound)
num_alerts = result.num_signals   # Number of detected signals
model_meta = result.params        # Audit trail of parameters used

# Export to Excel (.xlsx), Parquet (.parquet), or CSV (.csv)
# Accepts str or pathlib.Path
result.export("safety_audit.xlsx")                   # Multi-tab workbook ('Signals' and 'all_data')
result.export(Path("signals.parquet"))               # Fast, columnar Parquet export (default: which="signals")
result.export("audit.parquet", which="both")          # Generates audit_signals.parquet & audit_all.parquet
result.export("signals_only.csv")                    # Comma-separated signals export
```

### Logging & Diagnostics
`vigipy` provides structured diagnostic logging via Python's standard `logging` library under the `"vigipy"` namespace:

```python
import logging

# Configure vigipy to display solver iterations, SVD factorizations, and GLM diagnostics
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("vigipy").setLevel(logging.DEBUG)
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

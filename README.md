# vigipy

vigipy is a project to bring modern disproportionality analyses and pharmacovigilance techniques into the Python ecosystem with a simple, intuitive interface. Many of the disproportionality functions are adapted and extended versions from Ismail Ahmed and Antoine Poncet's [amazing work](https://cran.r-project.org/web/packages/PhViD/index.html). Top-level functions:

* bcpnn() - Bayesian confidence propogation neural network
* gps() - Multi-item gamma poisson shrinker
* prr() - Proportional reporting ratio
* ror() - Reporting odds ratio
* rfet() - Reporting fisher's exact test
* LongitudinalModel() - Apply any model across time to view signal evolution
* convert() - Convert a table of AEs, product and counts into a format for analysis

## Getting Started

### Dependencies

#### For vigipy

* pandas
* numpy
* scipy
* sympy >=1.3
* statsmodels >= 0.10.0
* patsy >= 0.5.1

### Installation

If there is ever any interest, I'll create the necessary setup files.

## Usage

### Load data and apply model

```python
from vigipy import *
import pandas as pd

#This is expected to have columns: ['AE', 'name', 'count'] ('date' is optional for longitudinal models)
df = pd.read_csv('AE_count_data.csv')
vivipy_data = convert(df)

#My personal favorite model to run. With 'log2' or 'quantile' as the statistic.
results = gps(vigipy_data, min_events=5, decision_metric='rank',
              decision_thres=1, ranking_statistic='log2')
results.signals.to_excel('possible_signals.xlsx', index=False)
```

### Making any model longitudinal:
```python
from vigipy import *
import pandas as pd

df = pd.read_csv('AE_count_data.csv')
#Apply model to each calendar year, cumulative
LM = LongitudinalModel(df, 'A')
LM.run(gps, False, decision_metric='rank', ranking_statistic='quantile')

#Change model time slice to quarterly
LM.regroup_dates('Q')
LM.run(gps, False, decision_metric='rank', ranking_statistic='quantile')

#LM produces a list of timestamps and results
for timestamp, result in LM.results:
    print("Signals produced prior to {0}:".format(timestamp))
    print(result.signals.head())
```

## TODO

* Create a data set for demonstrating usage more thoroughly and to run tests
* Improve high-level documentation of the methods and their parameters
* Integrate with mmappy and spotlight (coming soon) for an end-to-end device surveillance/pharmacovigilance pipeline

## Authors

David Beery

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Acknowledgements

* Ismail Ahmed and Antoine Poncet for the original leg work in designing and implementing a good disproportionality analysis library in R
* Ross Ihaka for his implementation of the log|gamma(x)| function
* Catherine Loader for her Stirling's formula log error and deviance functions.

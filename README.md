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
* sympy
* statsmodels
* patsy

### Installation

To install, navigate to the root directory of the repository and from the command line/terminal run:
```bash
python setup.py bdist_wheel
pip install dist\vigipy-1.0.0-py3-none-any.whl
```

You should now be able to import the vigipy library in your code.

#### Unit Tests
From the root directory of the repository, run:

```bash
python -m unittest discover -s test -p "*Test.py"
```

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

### Changing Expected Reporting Calculation
The expected count calculation is an important factor in both the accuracy and stability of any DA algorithm. Vigipy
support the use of 3 expectation method: mantel-haentzel, poisson, and negative-binomial. Recent research shows
BCPNN performs very well in most DA scenarios (https://onlinelibrary.wiley.com/doi/10.1002/pds.4970) and that LASSO
is very robust againt confounding variables. To combine these approaches in vigipy, use the negative-binomial expectation
argument in BCPNN. In the event of significant dispersion in the data, you can calculate alpha using `test_dispersion`

```python
from vigipy import convert
from vigipy.utils import test_dispersion

df = pd.read_csv('AE_count_data.csv')
data_container = convert(df)

dispersion_data = test_dispersion(data_container)
alpha = dispersion_data["alpha"] if dispersion_data["dispersion"] > 2 else 1

bcpnn(data_container, expected_method='negative-binomial', method_alpha=alpha, min_events=3)
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

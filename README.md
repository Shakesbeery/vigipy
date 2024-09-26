# vigipy

vigipy is a project to bring modern disproportionality analyses and pharmacovigilance techniques into the Python ecosystem with a simple, intuitive interface. Many of the disproportionality functions are adapted and extended versions from Ismail Ahmed and Antoine Poncet's [amazing work](https://cran.r-project.org/web/packages/PhViD/index.html). Top-level functions:

* bcpnn() - Bayesian confidence propogation neural network
* gps() - Multi-item gamma poisson shrinker
* lasso() - LASSO
* prr() - Proportional reporting ratio
* ror() - Reporting odds ratio
* rfet() - Reporting fisher's exact test
* LongitudinalModel() - Apply any model across time to view signal evolution
* convert() - Convert a table of AEs, product and counts into a format for analysis

## Getting Started

### Dependencies

#### For vigipy

* pandas==2.2.2
* numpy<2
* scipy==1.13.1
* scikit-learn==1.5.1
* sympy==1.12
* statsmodels==0.14.2

### Installation

To install, navigate to the root directory of the repository and from the command line/terminal run:
```bash
python setup.py bdist_wheel
pip install dist\<WheelName>
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

results = gps(vigipy_data, min_events=5, decision_metric='rank',
              decision_thres=1, ranking_statistic='log2', minimization_method="Nelder-Mead")
results.signals.to_excel('possible_signals.xlsx', index=False)
```

### Changing Expected Reporting Calculation
The expected count calculation is an important factor in both the accuracy and stability of any DA algorithm. Vigipy
support the use of 3 expectation method: mantel-haentzel, poisson, and negative-binomial. Recent research shows
BCPNN performs very well in most DA scenarios (https://onlinelibrary.wiley.com/doi/10.1002/pds.4970) and that LASSO
is very robust againt confounding variables. It is recommended you use the negative-binomial expectation
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
LM = LongitudinalModel(df, 'YE')
LM.run(gps, include_gaps=False, decision_metric='rank', ranking_statistic='quantile')

#Change model time slice to quarterly
LM.regroup_dates('Q')
LM.run(gps, include_gaps=False, decision_metric='rank', ranking_statistic='quantile')

#LM produces a list of timestamps and results
for timestamp, result in LM.results:
    print("Signals produced prior to {0}:".format(timestamp))
    print(result.signals.head())
```

Note that there are two ways of running longitudinal models. One is continuous (via `run()`) and the other is disjoint (via `run_disjoint()`).
The major difference is that `run()` assumes you are tracking a continually accumulating signal and that the past influences the present. When
running `run_disjoint()`, the assumption is that the current time slice is the only slice relevant for signal detection.


### Creating binary matrices for LASSO
This library now supports using LASSO directly for DA, but many of the assumptions and inputs are different here than for the 
other DA methods. First, because the method measures feature importance, we must create an input structure where the drug/device
is the model feature (i.e. column) and the adverse event is the response. A new conversion method `convert_binary` has been
exposed for this. The only assumption is that the input dataframe's rows each corresponds to a _unique_ drug/device pair. 

#### Pure binary matrix
```python
from vigipy import convert_binary, lasso
data = pd.read_csv("data.csv")

#provide the column labels for the drug/device name and the column label for the adverse events
bin_data = convert_binary(data, product_label="name", ae_label="AE")
results = lasso(bin_data, use_IC=True)
results.export("lasso_results.xlsx")
```

#### Binary counts
In this particular case, we treat our product as a binary participant (feature) in the event - either it was used or not. The event itself
remains as count data. This introduces a few complications such as removing the ability to bootstrap device-events for confidence interval measurements. It also may artificially inflate reported coefficients in smaller datasets, so use with caution and consider using the `use_glm`
flag now exposed through the `lasso` function.

```python
from vigipy import convert_binary, lasso
data = pd.read_csv("data.csv")

#provide the column labels for the drug/device name and the column label for the adverse events
bin_data = convert_binary(data, product_label="name", ae_label="AE", use_counts=True)
results = lasso(bin_data, use_IC=True) # OR res = lasso(bin_data, use_glm=True, lasso_thresh=0.25)
results.export("lasso_results.xlsx")
```

It is also possible to create multi-item binary matrices by providing a list of feature columns instead of strings:
```python
#This will mark each name column as a 1 or 0 if present for that AE report
convert_binary(data, product_label=["name", "name2"])
```

The same can also be done for adverse events if they are organied in columns instead of by rows.


### Experimental features
There is currently an experimental function for creating multi-item dataframes where we want to control
for the presence of possible drug/device interactions.

```python
from vigipy.utils.data_prep import convert_multi_item

# Takes an arbitrary number of column names that correspond to co-occurring drugs/devices/etc.
convert_multi_item(data, product_cols=["name", "name2", "name3"], ae_col="AE")
```

***Note:*** The output has some assumptions:
* `count_across_brands` - This assumes that we increment the total adverse event count one time for each product associated with the AE. For example, *Infection* occurs one time, but has two products in that row: Strattice and Pelvisoft. *Infection* will therefore be counted as 2. That is to say, Strattice and Pelvisoft both have 1 report each of this AE and the count will reflect this, even though the event itself is unique.
* `product_aes` - This column of data is a straight summation of the number of times the product occurs in the provided columns.
* `events` - The event tally for each product-AE combo is calculated based on the intermediate representation of each multi-item entry in the "product_combo" column. 

The generated container object will contain all parts expected by the main DA algorithms, so code modifications should not be necessary.


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

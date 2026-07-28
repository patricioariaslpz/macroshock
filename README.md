# macroshock

**macroshock** is a Python library for the estimation and analysis of **SVAR** (Structural Vector Autoregression) models aimed at applied macroeconometrics. It allows you to estimate restricted VARs, identify the structural system using four different methodologies, compute impulse-response functions (IRF), forecast error variance decomposition (FEVD), and forecasts, all with bootstrap confidence intervals and ready-to-use plotting.

## Main Branch (GitHub)

* **macroshock_v1_0_0.pdf** is the white paper for macroshock v1.0.0.
* **macroshock_documentation.pdf** is the complete technical documentation for macroshock's latest version (v1.0.1).
* **Paper Figure Replic** contains data and code to replicate the figures found in _macroshock_v1_0_0.pdf_.
* **Replications** contains data and code to replicate the four benchmark models (macroshock_v1_0_0.pdf/4.Replications).

_Note: data for replications is taken from Cesa-Bianchi's VAR-Toolbox v4.0, visit **https://github.com/ambropo/VAR-Toolbox** for full information._

## Main Features

* **Restricted VAR estimation** (Lütkepohl-style GLS), with support for:

|Supports|Input|
|-|-|
|Constant, trend and quadratic trend|`const`|
|Exogenous variables|`data_exogenous`,`variables_exogenous`|
|Seasonal dummies (quarterly or monthly)|`DUM`|
|Linear restrictions on the coefficients|`restrictions` and more|

* **Four structural identification methods**:

|Method|Description|
|-|-|
|`short`|Short-run restrictions (Cholesky or custom restrictions via optimization)|
|`long`|Long-run restrictions (Cholesky or custom restrictions via optimization)|
|`signs`|Sign restrictions (orthogonal rotation sampling with acceptance/rejection)|
|`IV`|Instrumental variable identification (proxy-SVAR)|

* **Impulse-response functions (IRF)** with confidence bands.
* **Forecast error variance decomposition (FEVD)**.
* **Conditional forecasts** with confidence bands.
* **Bootstrap** (normal or *wild* resampling) for inference.
* **Built-in plots** for IRF, forecasts and FEVD, publication-ready.
* **Stability test** of the system (characteristic polynomial roots).

## Installation

```bash
pip install macroshock
```

### Requirements

* Python >= 3.9
* `numpy >= 1.24.0`
* `pandas >= 2.0.0`
* `scipy`
* `matplotlib >= 3.7.0`

## Quick Start

```python
import pandas as pd
from macroshock import SVAR

# df is a DataFrame with your series in columns
df = pd.read\_csv("my\_dataset.csv")

model = SVAR(
    data=df,
    variables=\["gdp", "inflation", "interest\_rate"],
    lags=4,
    const=1,          # 0: no constant, 1: constant, 2: +trend, 3: +quadratic trend
    method="short",    # 'short', 'long', 'signs' or 'IV'
    steps=24,          # horizon for IRF/FEVD
    horizon=8,         # forecast horizon
    past=12,           # past periods to show in the forecast
    alpha=32,          # confidence interval level (32 -> 68% central band)
    reps=1000, # bootstrap replications
)

# Runs the complete pipeline: Y/X construction, VAR estimation,
# structural identification and results summary
model.run(showIR=True, showFC=True, showVD=True)

# System stability check
model.stability(show=True)
```

### Step-by-step Usage

The model.run() method executes the next python method secuence in a single line:

```python
model.YX()             # builds Y, X and the restriction matrix R
model.VAR()             # estimates the reduced-form VAR and the companion matrix F
model.S()                # identifies the structural system (matrix B)
model.summary()          # prints the estimation summary

# Impulse response to shock 1, with "pair" bands (low/high percentile)
ir = model.ImpulseResponse(1, show=True, bands="pair")

# Forecast with "many" bands (multiple stacked confidence levels)
fc = model.Forecast(show=True, bands="many")

# Forecast error variance decomposition for variable 1
vd = model.VarianceDecomp(1, show=True)
```

Each of these methods returns a dictionary with the point estimate and confidence bands at different levels (`'point'`, `'low'`, `'high'`, `2.5`, `5`, `95`, `97.5`, `0.5`, `99.5`), indexed by variable name.

## GLS Restrictions

The GLS methodology can be input in four ways depending on the level of depth one needs the restrictions to be. In this section the example of restrictions will be a bivariate VAR with `lags = 2` and `variables = [ y , u ]`. The restrictions are $$a^{(1)}\_{12}=a^{(2)}\_{12}=0$$, equal to saying `y` is an $$AR(2)$$ process or that `u` does not impact `y`inside the autoregression.

### Standard [`standard`]

Standard notation with $$\beta$$ being the vector of all the regressors, matrix $$C$$ (`c_matrix`) being the coefficients of the linear combination and $$c$$ (`c_vector`) being the intercept of the system. The system of constraint is given by $$(C\beta=c)$$.

```python
c_matrix = [[0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0]]

c_vector = [0,
            0]

modelo = SVAR(data2, variables, lags,
            restrictions='standard',
            c_matrix=c_matrix,
            c_vector=c_vector,
            )

modelo.run()
```

### Custom Zero Restrictions [`zeros custom`]

This notation uses a `n_vars`$$\times$$`n_cols` matrix (`r_matrix`). If `r_matrix[i][j]=0` the *j*th regressor of the *i*th equation has a zero-constraint parameter and if `r_matrix[i][j]=1` then the same regressor has a free parameter.

```python
r_matrix = [[1,1,0,1,0],
            [1,1,1,1,1]]

modelo = SVAR(data, variables, lags,
            restrictions='zeros custom',
            r_matrix=r_matrix,
            )

modelo.run()
```

### Absolut Zero Restrictions [`zeros`]

This notation uses a `n_vars`$$\times$$`n_vars` matrix (`r_matrix`). If `r_matrix[i][j]=0` all of the *j*th variable lagged autoregressive parameters are zero inside the *i*th equation and if `r_matrix[i][j]=1` then all lagged coefficients of the *j*th variable are free for the *i*th equation.

```python
r_matrix = [[1,0],
            [1,1]]

modelo = SVAR(data, variables, lags,
            restrictions='zeros',
            r_matrix=r_matrix,
            )

modelo.run()
```

### Custom [`custom`]

This notation follows Lutkepohl (2005) p.193-197 system $$(R\gamma=r)$$, so `r_matrix` and `r_vector` follow $$R$$ and $$r$$ definitions shown in the book.

```python
modelo = SVAR(data, variables, lags,
            restrictions='custom',
            r_matrix=r_matrix,
            r_vector=r_vector,
            )

modelo.run()
```

## Structural Identification

### 1\. Short-run (Cholesky / custom restrictions)

```python
model = SVAR(data=df, variables=variables, lags=4, method="short")
```

By default it uses the Cholesky decomposition. If `matrix\_short` is passed (a binary matrix of 1s and 0s), the `B` matrix satisfying the indicated zero restrictions is solved numerically.

### 2\. Long-run (Blanchard-Quah)

```python
model = SVAR(data=df, variables=variables, lags=4, method="long")
```

By default it uses the Cholesky decomposition. If `matrix\_long` is passed (a binary matrix of 1s and 0s), the `B` matrix satisfying the indicated zero restrictions is solved numerically.

### 3\. Sign restrictions

```python
matrix\_signs = \[\[1, -1, 0],
                \[1,  1, 0],
                \[0,  0, 1]]

model = SVAR(
    data=df, variables=variables, lags=4,
    method="signs",
    matrix\_signs=matrix\_signs,
    steps\_signs=4,       # horizons over which signs are checked
    reps=1000,   # number of accepted B matrices
)
```

### 4\. Instrumental variables (Proxy-SVAR)

```python
model = SVAR(
    data=df, variables=variables, lags=4,
    method="IV",
    iv=\["instrument\_1"],  # DataFrame column(s) with the instrument(s)
)
```

## Package Structure


```
macroshock/
├── data.py # construction of Y, X and restriction matrix R
├── var.py # reduced-form VAR estimation (restricted GLS, IC, t-stats)
├── stats.py # statistical utilities (Student's t CDF and p-values)
├── identification.py # the 4 structural identification methods
├── irf.py # impulse-response functions and bands from bootstrap
├── fevd.py # forecast error variance decomposition
├── fc.py # forecast computation
├── bootstrap.py # bootstrap (normal / wild) for IR, FC and FEVD
├── plotting.py # IRF, forecast and FEVD plots
└── macroshock.py # main SVAR class (and SVEC, LP scaffolds for future development)
```

## Roadmap

* Full documentation (Sphinx / ReadTheDocs)
* SVEC model (structural Vector Error Correction)
* Local Projections (LP)
* Historical decomposition (`HistoricalDecomp`)
* Unit tests and CI

## Contributing

Issues and pull requests are welcome. If you find a bug or have an improvement proposal, open an issue in the project's repository.

## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.


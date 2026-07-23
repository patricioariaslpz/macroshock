# macroshock

**macroshock** is a Python library for the estimation and analysis of **SVAR** (Structural Vector Autoregression) models aimed at applied macroeconometrics. It allows you to estimate restricted VARs, identify the structural system using four different methodologies, compute impulse-response functions (IRF), forecast error variance decomposition (FEVD), and forecasts, all with bootstrap confidence intervals and ready-to-use plotting.

## Main Features

* - **Restricted VAR estimation** (Lütkepohl-style GLS), with support for:

  - Constant, trend and quadratic trend.
  - Exogenous variables.
  - Seasonal dummies (quarterly or monthly).
  - Linear restrictions on the coefficients (`restrictions='zeros'` or `'custom zeros'`).
* - **Four structural identification methods**:

|Method|Description|
|-|-|
|`short`|Short-run restrictions (Cholesky or custom restrictions via optimization)|
|`long`|Long-run restrictions (Cholesky or custom restrictions via optimization)|
|`signs`|Sign restrictions (orthogonal rotation sampling with acceptance/rejection)|
|`IV`|Instrumental variable identification (proxy-SVAR)|

* - **Impulse-response functions (IRF)** with confidence bands.
* - **Forecast error variance decomposition (FEVD)**.
* - **Conditional forecasts** with confidence bands.
* - **Bootstrap** (normal or *wild* resampling) for inference.
* - **Built-in plots** for IRF, forecasts and FEVD, publication-ready.
* - **Stability test** of the system (characteristic polynomial roots).

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


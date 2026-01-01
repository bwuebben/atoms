# ATOMS Module

Original implementation of **ATOMS (Adaptive Tournament Model Selection)** from:

> **"The nonstationarity-complexity tradeoff in return prediction"**
> Capponi, Huang, Sidaoui, Wang, Zou (2025)

---

## Overview

ATOMS is a method for selecting machine learning models in non-stationary environments by jointly optimizing:
- **Model complexity** (which model class to use)
- **Training window size** (how much historical data to include)

### Key Innovation

Traditional approaches fix either the model or the window size. ATOMS recognizes these choices are intertwined: complex models need more data to avoid overfitting, but longer windows may include stale data from different market regimes.

---

## Files in this Module

### Core Algorithms

**`atoms.py`** - Main implementation
- `ValidationData`: Container for non-stationary time series data
- `atoms()`: Tournament algorithm (Algorithm 1 from paper)
- `adaptive_rolling_window_comparison()`: Pairwise comparison with adaptive window (Algorithm 2)
- `fixed_validation_selection()`: Fixed-window baseline (Algorithm 3)
- `ATOMSSelector`: High-level interface for full pipeline

**`atoms_r2.py`** - R²-based variant
- Targets R² metric directly instead of MSE (Appendix B from paper)
- `atoms_r2()`: Modified tournament for R²
- `ATOMSR2Selector`: Drop-in replacement for ATOMSSelector

### Example Scripts

**`example_regime_switching.py`** - Synthetic demonstration
- Generates data with explicit regime changes
- Different coefficients per regime (e.g., recession vs expansion)
- Compares ATOMS against fixed-window baselines
- Shows how adaptive selection helps during regime shifts

**`industry_portfolios.py`** - Empirical application
- Full pipeline for industry portfolio return prediction
- Mirrors the paper's empirical setup exactly
- Loads real data from `../../data/processed/` if available
- Falls back to synthetic data for demonstration
- Model specs: Ridge, LASSO, Elastic Net, Random Forest (as in paper)
- Window sizes: 4^k months for k=0..5

---

## Quick Start

### Basic Usage

```python
from atoms import ValidationData, ATOMSSelector
from sklearn.linear_model import Ridge

# Define model specifications
model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1}
]

# Define window sizes (exponential spacing)
window_sizes = [4, 16, 64, 256]

# Create selector
selector = ATOMSSelector(
    model_specs,
    window_sizes,
    M=0.1,           # Bound on |f(x)| and |y|
    delta_prime=0.1  # Confidence level
)

# Prepare data (list of arrays, one per period)
train_data = ValidationData(train_X_list, train_y_list)
val_data = ValidationData(val_X_list, val_y_list)

# Select model for period t
best_model, info = selector.select(train_data, val_data, t=100)

print(f"Selected: {info['winner_spec']}")
print(f"Window: {info['winner_window']}")

# Make prediction
prediction = best_model.predict(X_test)
```

### Running Examples

```bash
# Synthetic regime-switching demo
python example_regime_switching.py

# Industry portfolio analysis (uses real data if available)
python industry_portfolios.py
```

---

## Key Concepts

### The Nonstationarity-Complexity Tradeoff

Prediction error decomposes as (Theorem 3.1):

```
Error ≲ Misspecification(F) + Uncertainty(F, n_k) + Nonstationarity(k)
```

- **Misspecification**: Decreases with model complexity F
- **Uncertainty**: Increases with complexity, decreases with samples n_k
- **Nonstationarity**: Increases with window size k

ATOMS jointly minimizes this tradeoff.

### Adaptive Window Selection (Goldenshluger-Lepski Method)

For each pairwise comparison, ATOMS adaptively selects the validation window:

1. For each window ℓ, compute:
   - **Bias proxy**: φ̂(t,ℓ) = max_{i<ℓ} |δ̂(t,ℓ) - δ̂(t,i)| - (ψ̂(t,ℓ) + ψ̂(t,i))
   - **Variance proxy**: ψ̂(t,ℓ) = (4M²/√n + √(v̂/n)) × log(T/δ')

2. Select ℓ* = argmin{φ̂(t,ℓ) + ψ̂(t,ℓ)}

3. Compare models using data from window [t-ℓ*, t-1]

This achieves near-oracle performance (Theorem 4.1).

### Tournament Procedure

Instead of O(Λ²) pairwise comparisons:

1. Pick random pivot model
2. Compare pivot against all others
3. Eliminate pivot or all losers
4. Repeat until one model remains

**Expected comparisons**: O(Λ) instead of O(Λ²) (Lemma 4.1)

---

## Algorithms from Paper

### Algorithm 1: ATOMS Tournament

**Location**: `atoms()` function in `atoms.py`

**Purpose**: Select best model from candidates via sequential elimination

**Complexity**: O(Λ log Λ) expected comparisons

### Algorithm 2: Adaptive Window Comparison

**Location**: `adaptive_rolling_window_comparison()` in `atoms.py`

**Purpose**: Compare two models with adaptive validation window selection

**Key steps**:
1. Compute δ̂(t,ℓ), φ̂(t,ℓ), ψ̂(t,ℓ) for all windows ℓ
2. Select optimal ℓ* = argmin{φ̂ + ψ̂}
3. Return comparison result based on window ℓ*

### Algorithm 3: Fixed Window Baseline

**Location**: `fixed_validation_selection()` in `atoms.py`

**Purpose**: Baseline method using fixed validation window

**Use**: Benchmark for comparison

---

## Hyperparameters

| Parameter | Description | Default | Paper Value |
|-----------|-------------|---------|-------------|
| `M` | Bound on \|f(x)\| and \|y\| | 1.0 | 5×10⁻⁴ (monthly returns) |
| `delta_prime` | Confidence level | 0.1 | 0.1 |
| `window_sizes` | Training windows to test | [4,16,64,256] | [4,16,64,256,1024] |

**Guidance**:
- **M**: Set based on data scale. For monthly returns ∈ [-0.05, 0.05], use M ≈ 0.0005
- **delta_prime**: Smaller = more conservative. 0.1 works well in practice
- **window_sizes**: Exponential spacing (4^k) balances exploration vs computation

---

## Model Specifications (from Paper)

The paper tests these model classes and hyperparameters:

**Ridge Regression**:
```python
alphas = [1e-3, 10**(-1.5), 1, 10**1.5, 1e3]
model_specs = [{'class': Ridge, 'alpha': a} for a in alphas]
```

**LASSO**:
```python
alphas = [1e-5, 10**(-3.5), 1e-2, 10**(-0.5), 10]
model_specs = [{'class': Lasso, 'alpha': a, 'max_iter': 10000} for a in alphas]
```

**Elastic Net**:
```python
from itertools import product
alphas = [1e-3, 1, 1e3]
l1_ratios = [0.01, 0.05, 0.1]
model_specs = [
    {'class': ElasticNet, 'alpha': a, 'l1_ratio': r, 'max_iter': 10000}
    for a, r in product(alphas, l1_ratios)
]
```

**Random Forest**:
```python
from itertools import product
n_estimators = [10, 100, 200]
max_depths = [3, 5, 10]
model_specs = [
    {'class': RandomForestRegressor, 'n_estimators': n, 'max_depth': d, 'random_state': 42}
    for n, d in product(n_estimators, max_depths)
]
```

---

## Data Structure: ValidationData

ATOMS requires data in a specific format to preserve temporal structure:

```python
# Each period is a separate array
train_X = [X_period_1, X_period_2, ..., X_period_T]
train_y = [y_period_1, y_period_2, ..., y_period_T]

# Wrap in ValidationData
train_data = ValidationData(train_X, train_y)

# Access windows
X_window, y_window = train_data.get_window(t=100, ell=64)
# Returns concatenated data from periods [36, 99] (64 periods ending before t=100)
```

**Why not concatenate all data?**
- Preserves regime boundaries
- Allows window-based selection
- Maintains non-stationary structure

---

## Performance Metrics

### Out-of-Sample R² (Zero Benchmark)

Following Gu et al. (2020), the paper uses:

```python
R² = 1 - Σ(ŷ - y)² / Σy²
```

This benchmarks against a zero forecast rather than the historical mean.

**Why zero benchmark?**
- Mean return ≈ 0 in finance
- Tests if model adds value over "do nothing"
- More realistic for financial applications

### Recession Performance

From paper Table 2, ATOMS particularly excels during recessions:

| Period | ATOMS R² | Fixed(512) | Fixed-CV |
|--------|----------|------------|----------|
| Gulf War 1990 | **0.027** | -0.031 | -0.007 |
| 2001 Recession | **0.125** | 0.117 | 0.071 |
| 2008 Financial Crisis | **0.041** | 0.039 | 0.014 |

The adaptive window selection detects regime changes faster.

---

## Common Issues

### Date Alignment Errors

```
IndexError: index 0 is out of bounds for axis 0 with size 0
```

**Cause**: Misaligned dates between datasets

**Solution**: Ensure all data uses same date convention
```python
# Use month-start dates (matches Kenneth French format)
dates = pd.date_range(start='1987-09-01', end='2016-11-01', freq='MS')
```

### Empty Validation Windows

```
ValueError: Validation window is empty
```

**Cause**: t is too small for requested window size

**Solution**: Start predictions later
```python
# If max window is 256, start at t=300 to be safe
test_periods = range(300, T)
```

### Model Fitting Failures

```
LinAlgError: SVD did not converge
```

**Cause**: More features than samples (p > n)

**Solutions**:
- Use larger training windows
- Add regularization (Ridge instead of OLS)
- Reduce feature dimensionality

---

## Citation

```bibtex
@article{capponi2025nonstationarity,
  title={The nonstationarity-complexity tradeoff in return prediction},
  author={Capponi, Agostino and Huang, Chengpiao and Sidaoui, J. Antonio and Wang, Kaizheng and Zou, Jiacheng},
  year={2025}
}
```

---

## Related

- **S-ATOMS**: See `../satoms/` for enhanced version with soft ensemble weighting, empirical proxies, and similarity-based data selection
- **Main README**: See `../../README.md` for project overview
- **Paper**: See `../../docs/ssrn-5980654.pdf`

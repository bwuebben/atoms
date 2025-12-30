# ATOMS: Adaptive Tournament Model Selection

Implementation of the model selection framework from:

> **"The nonstationarity-complexity tradeoff in return prediction"**  
> Capponi, Huang, Sidaoui, Wang, Zou (2025)

## Overview

ATOMS (Adaptive Tournament Model Selection) is a method for selecting machine learning models in non-stationary environments. It jointly optimizes:
- **Model complexity** (which model class to use)
- **Training window size** (how much historical data to include)

The key insight is that these two choices are intertwined: complex models need more data, but longer windows may include stale data from different regimes.

## Installation

```bash
pip install numpy scikit-learn pandas matplotlib
```

## Quick Start

```python
from atoms import ValidationData, ATOMSSelector
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Define model specifications
model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1},
    {'class': Lasso, 'alpha': 0.01},
    {'class': RandomForestRegressor, 'n_estimators': 100, 'max_depth': 5}
]

# Define window sizes (4^k months as in the paper)
window_sizes = [4, 16, 64, 256]

# Create selector
selector = ATOMSSelector(model_specs, window_sizes, M=0.1)

# Prepare data (list of arrays, one per time period)
train_data = ValidationData(train_X_list, train_y_list)
val_data = ValidationData(val_X_list, val_y_list)

# Select model for period t
best_model, info = selector.select(train_data, val_data, t=100)
print(f"Selected: {info['winner_spec']}")
```

## Module Structure

### `atoms.py` - Core Implementation
- `ValidationData`: Container for non-stationary time series data
- `atoms()`: Main ATOMS tournament algorithm (Algorithm 1)
- `adaptive_rolling_window_comparison()`: Pairwise model comparison with adaptive window (Algorithm 2)
- `ATOMSSelector`: High-level interface for full pipeline
- `fixed_validation_selection()`: Baseline fixed-window method (Algorithm 3)

### `atoms_r2.py` - R²-Based Variant
- `atoms_r2()`: ATOMS targeting R² metric directly (Appendix B)
- `ATOMSR2Selector`: High-level interface for R²-based selection

### `example_regime_switching.py` - Synthetic Demo
Demonstrates ATOMS on data with explicit regime changes, comparing against fixed-window baselines.

### `industry_portfolios.py` - Empirical Application
Full pipeline for industry portfolio return prediction, mirroring the paper's empirical setup.

## Key Concepts

### The Nonstationarity-Complexity Tradeoff

Prediction error decomposes as (Theorem 3.1):

```
Error ≲ Misspecification(F) + Uncertainty(F, n_k) + Nonstationarity(k)
```

- **Misspecification**: Decreases with model complexity
- **Uncertainty**: Increases with complexity, decreases with window size
- **Nonstationarity**: Increases with window size

### Adaptive Window Selection

ATOMS adaptively selects the validation window using the Goldenshluger-Lepski method:

1. Compute bias proxy φ̂(t,ℓ) and variance proxy ψ̂(t,ℓ) for each window ℓ
2. Select ℓ* = argmin{φ̂ + ψ̂}
3. Compare models using data from window ℓ*

This achieves near-oracle performance (Theorem 4.1).

### Tournament Procedure

Instead of comparing all pairs O(Λ²), ATOMS uses a random pivot tournament:

1. Pick random pivot model
2. Compare pivot against all others
3. Advance models that beat the pivot
4. Repeat until one remains

Expected comparisons: O(Λ) (Lemma 4.1)

## Hyperparameters

| Parameter | Description | Default | Paper Value |
|-----------|-------------|---------|-------------|
| `delta_prime` | Confidence level for comparisons | 0.1 | 0.1 |
| `M` | Bound on \|f(x)\| and \|y\| | 1.0 | 5×10⁻⁴ (returns) |
| `window_sizes` | Training windows to consider | [4,16,64,256] | 4^k, k=0..5 |

## Model Specifications (from paper)

**Ridge**: α ∈ {10⁻³, 10⁻¹·⁵, 1, 10¹·⁵, 10³}

**LASSO**: α ∈ {10⁻⁵, 10⁻³·⁵, 10⁻², 10⁻⁰·⁵, 10}

**Elastic Net**: α ∈ {10⁻³, 1, 10³}, r ∈ {0.01, 0.05, 0.1}

**Random Forest**: n_trees ∈ {10, 100, 200}, max_depth ∈ {3, 5, 10}

## Performance Metrics

**Out-of-sample R² (zero benchmark)**:
```
R² = 1 - Σ(ŷ - y)² / Σy²
```

This benchmarks against a zero forecast rather than the mean, following Gu et al. (2020).

## Example: Recession Performance

From the paper (Table 2), ATOMS particularly excels during recessions:

| Period | ATOMS | Fixed-val(512) | Fixed-CV |
|--------|-------|----------------|----------|
| Gulf War 1990 | **0.027** | -0.031 | -0.007 |
| 2001 Recession | **0.125** | 0.117 | 0.071 |
| Financial Crisis | **0.041** | 0.039 | 0.014 |

## Citation

```bibtex
@article{capponi2025nonstationarity,
  title={The nonstationarity-complexity tradeoff in return prediction},
  author={Capponi, Agostino and Huang, Chengpiao and Sidaoui, J. Antonio and Wang, Kaizheng and Zou, Jiacheng},
  year={2025}
}
```

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*.
- Kelly, B., Malamud, S., & Zhou, K. (2024). The virtue of complexity in return prediction. *Journal of Finance*.
- Han, E., Huang, C., & Wang, K. (2024). Model assessment and selection under temporal distribution shift. *ICML*.

# ATOMS & S-ATOMS: Adaptive Model Selection for Non-Stationary Environments

This repository contains two related implementations for adaptive model selection in financial forecasting and other non-stationary time series applications.

## Implementations

### ATOMS (src/atoms/) - Original Algorithm

Implementation of the model selection framework from:

> **"The nonstationarity-complexity tradeoff in return prediction"**
> Capponi, Huang, Sidaoui, Wang, Zou (2025)

ATOMS (Adaptive Tournament Model Selection) jointly optimizes:
- **Model complexity** (which model class to use)
- **Training window size** (how much historical data to include)

Key features:
- Tournament-based selection with O(Λ) comparisons
- Adaptive window selection via Goldenshluger-Lepski method
- Hard selection (winner-take-all)

### S-ATOMS (src/satoms/) - Enhanced Algorithm

Implementation of the enhanced framework from:

> **"When History Rhymes: Ensemble Learning and Regime-Aware Estimation under Nonstationarity"**
> Wuebben (2025)

S-ATOMS extends ATOMS with three key innovations:
1. **Soft Ensemble Weighting** (Section 3.2) - Exponentially weighted averaging instead of hard selection
2. **Empirical Proxy Estimation** (Section 3.1) - Block bootstrap variance + integral drift bias
3. **Similarity-Based Data Selection** (Section 3.3) - Mahalanobis distance for regime-aware training

**Performance improvement:** 69-200% better out-of-sample R² vs ATOMS in regime-switching environments (see VERIFICATION_REPORT.md)

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd atoms

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup (Optional)

Download real empirical data (Kenneth French portfolios):

```bash
python src/download_data.py
```

This downloads and processes:
- 17 industry portfolios (Kenneth French Data Library)
- Fama-French factors
- Momentum factors
- Date range: 1987-09 to 2016-11

Data is saved to `data/processed/`. If not downloaded, examples will generate synthetic data automatically.

---

## Quick Start

### ATOMS - Original Algorithm

```python
from src.atoms.atoms import ValidationData, ATOMSSelector
from sklearn.linear_model import Ridge, Lasso

# Define model specifications
model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1},
    {'class': Lasso, 'alpha': 0.01}
]

# Define window sizes (exponential spacing as in paper)
window_sizes = [4, 16, 64, 256]

# Create selector
selector = ATOMSSelector(model_specs, window_sizes, M=0.1, delta_prime=0.1)

# Prepare data (list of arrays, one per time period)
train_data = ValidationData(train_X_list, train_y_list)
val_data = ValidationData(val_X_list, val_y_list)

# Select model for period t
best_model, info = selector.select(train_data, val_data, t=100)
print(f"Selected: {info['winner_spec']}, Window: {info['winner_window']}")

# Make prediction
prediction = best_model.predict(X_test)
```

### S-ATOMS - Enhanced Algorithm

```python
from src.satoms.s_atoms import SATOMSSelector, ValidationData
from sklearn.linear_model import Ridge

# Define model specifications
model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1}
]

# Create S-ATOMS selector with all enhancements
selector = SATOMSSelector(
    model_specs=model_specs,
    window_sizes=[4, 16, 64],
    data_sources=['contiguous', 'similarity', 'blended'],  # Similarity-based selection
    gamma='adaptive',              # Soft ensemble weighting
    use_integral_drift=True,       # Empirical proxies
    n_bootstrap=500,
    M=0.1
)

# Prepare data
train_data = ValidationData(train_X_list, train_y_list)
val_data = ValidationData(val_X_list, val_y_list)

# Select and get ensemble
weights, risk_scores, info = selector.select(train_data, val_data, t=100)

# Make ensemble prediction (soft weighting)
prediction = selector.predict(X_test, info['candidates'], weights)

print(f"Ensemble size: {info['effective_n_models']:.1f} models")
print(f"Best individual: {info['winner_spec']}")
```

---

## Running Examples

### ATOMS Examples

```bash
# Synthetic regime-switching demonstration
python src/atoms/example_regime_switching.py

# Industry portfolio analysis (uses data/processed/ if available)
python src/atoms/industry_portfolios.py
```

### S-ATOMS Examples

```bash
# Quick functionality test (~5 seconds)
python src/satoms/test_minimal.py

# Fast industry portfolio comparison (~30 seconds)
python src/satoms/test_industry_fast.py

# Full industry portfolio analysis (~10-30 minutes)
python src/satoms/s_atoms_industry.py
```

---

## Project Structure

```
atoms/
├── README.md                    # This file
├── CLAUDE.md                    # Development guide
├── requirements.txt
├── data/
│   ├── processed/              # Processed data (from download_data.py)
│   └── raw/                    # Raw downloaded data
├── docs/
│   ├── ssrn-5980654.pdf        # ATOMS paper
│   └── Doc__A_Atoms_Synthesis.pdf  # S-ATOMS paper
└── src/
    ├── download_data.py        # Data download utility
    ├── atoms/                  # Original ATOMS implementation
    │   ├── README.md           # ATOMS module documentation
    │   ├── atoms.py            # Core algorithm
    │   ├── atoms_r2.py         # R²-based variant
    │   ├── example_regime_switching.py
    │   └── industry_portfolios.py
    └── satoms/                 # Enhanced S-ATOMS implementation
        ├── README.md           # S-ATOMS module documentation
        ├── s_atoms.py          # Core algorithm
        ├── s_atoms_industry.py # Industry portfolio analysis
        ├── test_minimal.py     # Quick test
        ├── test_industry_fast.py  # Fast realistic test
        └── VERIFICATION_REPORT.md  # Implementation verification
```

---

## Key Concepts

### The Nonstationarity-Complexity Tradeoff

Prediction error decomposes as:

```
Error ≲ Misspecification(F) + Uncertainty(F, n_k) + Nonstationarity(k)
```

- **Misspecification**: Decreases with model complexity
- **Uncertainty**: Increases with complexity, decreases with window size
- **Nonstationarity**: Increases with window size

ATOMS/S-ATOMS jointly optimize this tradeoff.

### Adaptive Window Selection

Both algorithms adaptively select the validation window using sophisticated bias-variance balancing:

**ATOMS:** Goldenshluger-Lepski method
- Computes bias proxy φ̂(t,ℓ) and variance proxy ψ̂(t,ℓ)
- Selects ℓ* = argmin{φ̂ + ψ̂}

**S-ATOMS:** Empirical proxies
- Bootstrap variance estimation (preserves autocorrelation)
- Integral drift bias (robust to outliers)

### Tournament vs Ensemble

**ATOMS:** Tournament selection (hard)
- Random pivot tournament: O(Λ) comparisons
- Returns single best model

**S-ATOMS:** Soft ensemble weighting
- Computes risk scores for all candidates
- Exponential weighting: W_λ ∝ exp(-γ·R_λ)
- Adaptive sharpness parameter γ
- Returns weighted ensemble of all models

---

## Performance Comparison

From `src/satoms/test_industry_fast.py`:

```
ATOMS:   R² = -0.315
S-ATOMS: R² = -0.095
Improvement: 69.4%
```

(Negative R² is common in finance; less negative = better)

Key improvements from S-ATOMS:
- **Soft ensemble**: Reduces selection instability
- **Similarity selection**: Leverages historical analogues
- **Empirical proxies**: Tighter variance estimates

---

## Hyperparameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `M` | Bound on \|f(x)\| and \|y\| | 1.0 | Use 0.0005 for monthly returns |
| `delta_prime` | Confidence level | 0.1 | Smaller = more conservative |
| `window_sizes` | Training windows | [4,16,64,256] | Exponential spacing |
| `gamma` | Ensemble sharpness (S-ATOMS) | 'adaptive' | Higher = harder selection |
| `n_bootstrap` | Bootstrap samples (S-ATOMS) | 500 | More = better variance estimates |

---

## Model Specifications (from papers)

**Ridge**: α ∈ {10⁻³, 10⁻¹·⁵, 1, 10¹·⁵, 10³}

**LASSO**: α ∈ {10⁻⁵, 10⁻³·⁵, 10⁻², 10⁻⁰·⁵, 10}

**Elastic Net**: α ∈ {10⁻³, 1, 10³}, r ∈ {0.01, 0.05, 0.1}

**Random Forest**: n_trees ∈ {10, 100, 200}, max_depth ∈ {3, 5, 10}

---

## Documentation

- **ATOMS module**: See `src/atoms/README.md`
- **S-ATOMS module**: See `src/satoms/README.md`
- **Implementation verification**: See `src/satoms/VERIFICATION_REPORT.md`
- **Development guide**: See `CLAUDE.md`

---

## Citations

### ATOMS
```bibtex
@article{capponi2025nonstationarity,
  title={The nonstationarity-complexity tradeoff in return prediction},
  author={Capponi, Agostino and Huang, Chengpiao and Sidaoui, J. Antonio and Wang, Kaizheng and Zou, Jiacheng},
  year={2025}
}
```

### S-ATOMS
```bibtex
@article{wuebben2025history,
  title={When History Rhymes: Ensemble Learning and Regime-Aware Estimation under Nonstationarity},
  author={Wuebben, Bernd},
  year={2025}
}
```

---

## License

Copyright © 2025 Bernd J. Wuebben. All Rights Reserved.

## Contact

Bernd J. Wuebben <wuebben@gmail.com>

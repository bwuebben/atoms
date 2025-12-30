# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ATOMS (Adaptive Tournament Model Selection) is a research implementation of model selection for non-stationary environments, based on the paper "The nonstationarity-complexity tradeoff in return prediction" (Capponi, Huang, Sidaoui, Wang, Zou, 2025).

The key innovation is jointly optimizing:
- **Model complexity** (which model class to use)
- **Training window size** (how much historical data to include)

This tradeoff is critical because complex models need more data, but longer windows may include stale data from different regimes.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (if not already done)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Download
```bash
# Download empirical data (Kenneth French portfolios)
python download_data.py

# Force re-download (overwrite existing files)
python download_data.py --overwrite

# Use only synthetic data (no downloads)
python download_data.py --synthetic
```

The script:
- Downloads French data library files (17 industry portfolios, FF factors, momentum, 5-factor)
- Generates synthetic approximations for GKX characteristics and CPZ SDF factors
- Aligns all data to common date range (1987-09 to 2016-11)
- Saves processed data to `data/processed/`

**Important**: All dates use month-start convention (e.g., "1987-09-01") to match Kenneth French data format.

### Running Examples
```bash
# Synthetic regime-switching demonstration
python example_regime_switching.py

# Empirical industry portfolio analysis
# Automatically uses data from data/processed/ if available
python industry_portfolios.py
```

**Note**: After running `python download_data.py`, both example scripts will automatically load data from `./data/processed/` instead of generating synthetic data. You can still pass custom file paths if needed.

### Testing Individual Components
```python
# Test core ATOMS algorithm on custom data
from atoms import ValidationData, ATOMSSelector
from sklearn.linear_model import Ridge

model_specs = [{'class': Ridge, 'alpha': 1.0}, {'class': Ridge, 'alpha': 0.1}]
window_sizes = [4, 16, 64]
selector = ATOMSSelector(model_specs, window_sizes, M=0.1)
# ... prepare train_data and val_data as ValidationData objects
best_model, info = selector.select(train_data, val_data, t=100)
```

## Data Loading Behavior

The example scripts (`industry_portfolios.py`) are designed to work flexibly:

1. **First priority**: Load from `./data/processed/` directory (created by `download_data.py`)
   - `industry_returns.csv` - 17 industry portfolios from Kenneth French
   - `features.csv` - 133 predictors (FF factors, characteristics, SDF factors, lagged returns)

2. **Fallback**: Generate synthetic data for demonstration if processed data not found

3. **Custom paths**: Pass `filepath` parameter to load from custom locations

**Example**:
```python
from industry_portfolios import load_french_industry_portfolios, create_factor_data

# Automatic - uses data/processed/ if available, else synthetic
returns = load_french_industry_portfolios()
features = create_factor_data(returns)

# Custom path
returns = load_french_industry_portfolios(filepath='path/to/custom/returns.csv')
features = create_factor_data(returns, filepath='path/to/custom/features.csv')
```

## Code Architecture

### Core Algorithms (atoms.py)

The implementation follows the paper's algorithms directly:

1. **Algorithm 1: ATOMS Tournament** (`atoms()` function)
   - Sequential elimination tournament over model-window combinations
   - Uses random pivot selection for efficiency: O(Λ) comparisons instead of O(Λ²)
   - Returns winner, winner index, and comparison metadata

2. **Algorithm 2: Adaptive Window Selection** (`adaptive_rolling_window_comparison()`)
   - Goldenshluger-Lepski method for bias-variance balancing
   - Computes bias proxy φ̂(t,ℓ) and variance proxy ψ̂(t,ℓ) for each window ℓ
   - Selects ℓ* = argmin{φ̂ + ψ̂}
   - Used within pairwise model comparisons

3. **Algorithm 3: Fixed Window Baseline** (`fixed_validation_selection()`)
   - Simple baseline using fixed validation window
   - Useful for comparison benchmarks

### Key Data Structures

**ValidationData**: Core container for non-stationary time series
- Stores data as `List[np.ndarray]` (one array per time period)
- Maintains temporal structure explicitly (unlike concatenating all data)
- Methods: `get_window(t, ell)` retrieves data from periods [t-ell, t-1]
- Important: This structure preserves regime boundaries

**BaseModel/ModelWrapper**: Abstraction layer
- `BaseModel`: Abstract class defining `predict()` and `mse()` interface
- `ModelWrapper`: Adapts sklearn models to BaseModel interface
- All comparisons happen through this unified interface

**ATOMSSelector**: High-level pipeline manager
- Manages (model_spec, window_size) combinations
- Handles training on appropriate windows: `train_data.get_window(t, window_size)`
- Returns both the selected model and selection metadata

### Control Flow

For prediction at time t:
1. `ATOMSSelector.select()` creates all (model, window) candidates
2. For each combination: train model on `train_data[t-window:t-1]`
3. Call `atoms()` with trained candidates and `val_data`
4. `atoms()` runs tournament using `adaptive_rolling_window_comparison()`
5. Each comparison adaptively selects validation window from `val_data[0:t-1]`
6. Winner is returned and can predict on test data at time t

**Critical distinction**: Training data and validation data are separate. Training windows vary by candidate (part of the search space). Validation window is adaptive (selected by GL method).

### R²-Based Variant (atoms_r2.py)

Implements Appendix B from the paper:
- Targets R² metric directly instead of MSE
- Uses variance-normalized comparison statistics (Eq B.2)
- Different variance proxy computation (Eq B.3)
- Same tournament structure, different pairwise comparison logic
- Use `ATOMSR2Selector` as drop-in replacement for `ATOMSSelector`

### Example Scripts

**example_regime_switching.py**: Synthetic demonstration
- Generates data with explicit regime changes at specified periods
- Different coefficients per regime (simulating recessions/recoveries)
- Compares ATOMS against fixed-window baselines
- Good for understanding how adaptive selection helps during regime shifts

**industry_portfolios.py**: Empirical application
- Mirrors the paper's setup exactly
- Model specifications from paper: Ridge (5 alphas), LASSO (5 alphas), Elastic Net (9 configs), Random Forest (9 configs)
- Window sizes: 4^k months for k=0..5 → [4, 16, 64, 256, 1024]
- Computes out-of-sample R² benchmarked against zero forecast (following Gu et al. 2020)
- Can run with synthetic or real Kenneth French data

**download_data.py**: Data fetching utility
- Downloads from Kenneth French Data Library
- Processes zip files and extracts relevant date ranges
- Default range: 1987-09 to 2016-11 (matching paper)
- Creates `data/raw/` and `data/processed/` directories

## Hyperparameter Guidance

**M (bound parameter)**:
- Theory requires |f(x)| ≤ M and |y| ≤ M
- For monthly returns: paper uses M = 5×10⁻⁴
- For synthetic examples: typically M = 1.0
- Affects variance proxy scaling: ψ̂ = (4M²/√n + √(v̂/n)) × log(T/δ')

**delta_prime (confidence level)**:
- Controls comparison confidence intervals
- Default: 0.1 (paper's choice)
- Smaller = more conservative comparisons
- Appears in log factor: log(2T(T+1)/δ')

**window_sizes**:
- Paper uses exponential spacing: 4^k for k=0..5
- Rationale: geometric growth balances exploration
- Too few windows: may miss optimal bias-variance tradeoff
- Too many windows: computational cost increases linearly

**v_lower (R² variant only)**:
- Lower bound on E[y²] (Assumption B.1)
- Prevents division by zero in variance normalization
- Default: 0.01
- Set based on minimum expected target variance

## Performance Notes

**Computational cost**:
- O(Λ log Λ) expected comparisons for Λ candidate models
- Each comparison iterates over O(T) validation windows
- Dominant cost: model training (happens once per candidate)
- Validation window selection is typically fast (closed-form statistics)

**Memory considerations**:
- ValidationData stores all historical periods in memory
- For T periods with n samples each: O(nT × d) memory for features
- Large empirical studies may need chunked processing

**Numerical stability**:
- Variance estimates use sample std with ddof=1
- When n_ell=1, falls back to conservative bound: 8M²
- GL method uses max(0, ·) to handle negative bias estimates

## Common Issues

**Date alignment errors** (IndexError: index 0 is out of bounds):
- Ensure all datasets use month-start dates (`freq='MS'` in pd.date_range)
- Kenneth French data parses as month-start (YYYYMM → YYYY-MM-01)
- Synthetic data generation must match this convention

**Empty validation windows**:
- Check that `t` is large enough for the requested window size
- ValidationData.get_window(t, ell) requires at least t > ell
- Early periods may not have sufficient history for large windows

**Model fitting failures**:
- Verify training data has sufficient samples for model complexity
- Ridge/LASSO may fail with singular matrices if features > samples
- RandomForest requires n_samples >= min_samples_split

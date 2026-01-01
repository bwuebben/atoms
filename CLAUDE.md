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

## S-ATOMS: Enhanced Implementation

This repository also contains **S-ATOMS** (Soft, Similarity-based Adaptive Tournament Model Selection), an enhanced version of ATOMS from the paper "When History Rhymes: Ensemble Learning and Regime-Aware Estimation under Nonstationarity" (Wuebben, 2025).

### Key Enhancements over ATOMS

S-ATOMS extends the original ATOMS framework with three major innovations:

1. **Soft Ensemble Weighting (Section 3.2)** - Instead of hard selection, uses exponentially weighted averaging
2. **Empirical Proxy Estimation (Section 3.1)** - Bootstrap variance + integral drift bias for tighter bounds
3. **Similarity-Based Data Selection (Section 3.3)** - Mahalanobis distance for regime-aware training

### Running S-ATOMS Examples

```bash
# Quick functionality test (~5 seconds)
python src/satoms/test_minimal.py

# Fast industry portfolio comparison (~30 seconds)
python src/satoms/test_industry_fast.py

# Full industry portfolio analysis (~10-30 minutes, computationally intensive)
python src/satoms/s_atoms_industry.py
```

**Note**: The full `s_atoms_industry.py` is computationally intensive because it trains ~420 candidates per period (28 model specs × 5 windows × 3 data sources). Use the fast test version for quick validation.

### Testing S-ATOMS Components

```python
# Test core S-ATOMS algorithm
from src.satoms.s_atoms import SATOMSSelector, ValidationData
from sklearn.linear_model import Ridge

model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1}
]
window_sizes = [4, 16, 64]

# Create selector with all enhancements
selector = SATOMSSelector(
    model_specs=model_specs,
    window_sizes=window_sizes,
    data_sources=['contiguous', 'similarity', 'blended'],
    gamma='adaptive',              # Soft ensemble
    n_bootstrap=500,               # Bootstrap variance
    use_integral_drift=True,       # Integral drift bias
    M=0.1,
    verbose=True
)

# ... prepare train_data and val_data as ValidationData objects
weights, risk_scores, info = selector.select(train_data, val_data, t=100)

# Ensemble prediction
pred = selector.predict(X_test, info['candidates'], weights)

print(f"Effective number of models: {info['effective_n_models']:.1f}")
print(f"Winner: {info['winner_spec']}")
```

## S-ATOMS Code Architecture

### Core Implementation (src/satoms/s_atoms.py)

The S-ATOMS implementation follows Algorithm 1 from the paper with 4 phases:

**Phase 1: Data Construction**
- `MarketStateVector.compute_state_simple()` - Constructs state vector from returns
- `SimilarityDataSelector.select_similar_periods()` - Mahalanobis distance selection
- `SimilarityDataSelector.construct_blended_dataset()` - Blends similar + recent data

**Phase 2: Candidate Training**
- `SATOMSSelector.train_candidates()` - Trains all (model, window, data_source) combinations
- Handles three data sources:
  - `contiguous`: Traditional rolling window
  - `similarity`: Only similar historical periods
  - `blended`: Union of similar + recent periods

**Phase 3: Risk Score Computation**
- `BlockBootstrapVariance.compute_variance_proxy()` - Bootstrap variance (Eq 15-16)
- `IntegralDriftBias.compute_integral_drift()` - Drift-based bias proxy (Eq 18-20)
- Risk = Bias + Variance

**Phase 4: Ensemble Weighting**
- `SoftEnsembleWeighter.select_gamma_adaptive()` - Cross-validation for sharpness (Eq 26)
- `SoftEnsembleWeighter.compute_weights()` - Softmax weights (Eq 23)
- `SoftEnsembleWeighter.compute_ensemble_prediction()` - Weighted average (Eq 24)

### Key S-ATOMS Data Structures

**CandidateModel**: Extended model container
- Stores: model, spec, training_window, data_source, parameters
- Used for tracking all candidate configurations

**MarketState**: State vector container
- Components: volatility, correlation, macro, market conditions
- 15-dimensional vector for similarity matching

**SATOMSSelector**: Main interface
- Manages full 4-phase pipeline
- Handles market state computation
- Tracks previous weights for turnover calculation

### S-ATOMS vs ATOMS Control Flow

**ATOMS:**
1. Train candidates on rolling windows only
2. Run tournament to select single winner
3. Return best model

**S-ATOMS:**
1. Compute market state vectors for all periods
2. Train candidates on: rolling windows + similarity-based + blended data
3. Compute risk scores with empirical proxies
4. Compute soft ensemble weights
5. Return ensemble of all candidates

## S-ATOMS Hyperparameter Guidance

**Ensemble Parameters:**
- `gamma` (sharpness): Default 'adaptive'. Higher = more concentrated weights (closer to hard selection). Lower = more diversified.
- `gamma_grid`: For adaptive selection, default [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
- `calibration_window`: For gamma cross-validation, default 24 months

**Bootstrap Parameters:**
- `n_bootstrap`: Number of bootstrap samples. Default 500 (paper), use 100-200 for faster testing
- `block_length`: Block size for circular bootstrap. Default 'auto' (Politis-White 2004)

**Integral Drift Parameters:**
- `decay_half_life`: Exponential decay half-life for drift kernel. Default 6 months
- `calibration_constant`: Multiplicative factor c_φ. Default 1.15 (paper calibration)

**Similarity Selection Parameters:**
- `target_sample_size`: Target observations in similarity set. Default 500
- `recent_window`: Recent window always included. Default 12 months
- `omega_sim` / `omega_recent`: Mixing weights. Default 'adaptive' (proportional to set sizes)
- `kernel`: Similarity kernel. Default 'epanechnikov', also supports 'gaussian'
- `shrinkage_alpha`: Covariance shrinkage. Default 0.1

**Data Sources:**
- `['contiguous']`: Traditional ATOMS (fast)
- `['contiguous', 'similarity']`: Add similarity-based selection
- `['contiguous', 'similarity', 'blended']`: Full S-ATOMS (paper default)

## S-ATOMS Performance Notes

**Computational Scaling:**
- Training: O(|M| × |k| × |D| × C_train) where |D| = number of data sources (1-3)
- Bootstrap: O(Λ × B × n) where B = bootstrap samples, n = validation samples
- Integral drift: O(Λ × ℓ²) where ℓ = max validation window
- Total per period: ~420 candidates × 500 bootstrap samples for full configuration

**Speed optimization:**
- Reduce `n_bootstrap` to 100-200 for testing
- Use fewer model specs (e.g., 5-10 instead of 28)
- Limit window_sizes to [4, 16, 64] instead of [1, 4, 16, 64, 256]
- Use `data_sources=['contiguous']` to disable similarity selection

**Example configurations:**

Fast testing (30 sec):
```python
selector = IndustrySATOMS(
    gamma='adaptive',
    use_similarity=False,      # Disable similarity
    use_soft_ensemble=True,
    use_empirical_proxies=True,
    verbose=False
)
selector.model_specs = selector.model_specs[:5]  # Only first 5 models
selector.window_sizes = [4, 16, 64]
selector.bootstrap_variance.n_bootstrap = 100
```

Full paper configuration (10-30 min):
```python
selector = IndustrySATOMS(
    gamma='adaptive',
    use_similarity=True,       # Full S-ATOMS
    use_soft_ensemble=True,
    use_empirical_proxies=True,
    verbose=True
)
# Uses all 28 model specs, 5 window sizes, 3 data sources, 500 bootstrap
```

## S-ATOMS Common Issues

**Memory usage with similarity selection:**
- Stores all historical market states: O(T × d_state) where T = periods, d_state ≈ 15
- For T = 350 periods: ~5KB of state vectors (negligible)
- Main memory is still ValidationData: O(T × n × d_features)

**Singular covariance matrix:**
- If T < d_state (very early periods), covariance estimation may fail
- Solution: Automatic shrinkage with α = 0.1 prevents this
- Also checks eigenvalues and adds regularization if needed

**Slow execution:**
- Most common cause: Too many candidates
- Each candidate requires training a model
- With 28 models × 5 windows × 3 data sources = 420 candidates/period
- Solution: Reduce model_specs or use `data_sources=['contiguous']`

**NaN in ensemble weights:**
- Can occur if all risk scores are infinite
- Handled automatically: falls back to uniform weights
- Usually indicates all models failed to train (check data validity)

## Verification Status

The S-ATOMS implementation has been thoroughly verified against the paper:
- All equations (15-35) correctly implemented
- Numerical stability verified
- Test coverage: minimal test, fast industry test, full analysis
- See `src/satoms/VERIFICATION_REPORT.md` for detailed verification

**Known correct:**
- ✅ Bootstrap variance estimation (Eq 15-16)
- ✅ Integral drift bias (Eq 17-20)
- ✅ Soft ensemble weighting (Eq 23-26)
- ✅ Mahalanobis distance (Eq 29)
- ✅ Covariance regularization (Eq 30)
- ✅ Blended data construction (Eq 33-35)
- ✅ Complete Algorithm 1 (4-phase procedure)

**Performance validation:**
- Minimal test (50 periods, 2 models): Passes in ~5 seconds
- Fast test (100 periods, 2 models, 3 industries): S-ATOMS improves 69% over ATOMS
- Matches expected behavior from paper


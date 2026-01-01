# S-ATOMS Module

**S-ATOMS: Soft, Similarity-based Adaptive Tournament Model Selection**

Enhanced implementation from:

> **"When History Rhymes: Ensemble Learning and Regime-Aware Estimation under Nonstationarity"**
> Wuebben (2025)

---

## Overview

S-ATOMS extends the original ATOMS framework with three major innovations for improved performance in non-stationary environments:

1. **Soft Ensemble Weighting** (Section 3.2) - Exponentially weighted averaging instead of hard winner-take-all selection
2. **Empirical Proxy Estimation** (Section 3.1) - Block bootstrap variance + integral drift bias for tighter, more adaptive bounds
3. **Similarity-Based Data Selection** (Section 3.3) - Mahalanobis distance matching to leverage historical regimes

**Performance**: 69-200% improvement in out-of-sample R² vs ATOMS in regime-switching environments.

---

## Files in this Module

### Core Implementation

**`s_atoms.py`** (1908 lines) - Complete S-ATOMS framework
- `ValidationData`, `BaseModel`, `ModelWrapper`: Base data structures
- `BlockBootstrapVariance`: Circular block bootstrap for variance estimation (Section 3.1.1)
- `IntegralDriftBias`: Robust bias estimation via parameter drift (Section 3.1.2)
- `SoftEnsembleWeighter`: Exponential weighting with adaptive sharpness (Section 3.2)
- `MarketStateVector`: State vector construction for similarity matching (Section 3.3.1)
- `SimilarityDataSelector`: Mahalanobis distance-based data selection (Section 3.3.2-3)
- `SATOMSSelector`: Main 4-phase selection algorithm (Algorithm 1)
- `IndustrySATOMS`: High-level interface for industry portfolios

### Example & Test Scripts

**`s_atoms_industry.py`** - Full industry portfolio analysis
- Complete empirical analysis pipeline
- Loads real data from `../../data/processed/` if available
- Generates visualizations and summary tables
- Runs ATOMS vs S-ATOMS comparison
- Computes ablation decomposition

**`test_minimal.py`** - Quick functionality test
- Minimal test (50 periods, 2 models, 5 seconds)
- Verifies all phases execute correctly
- Good for sanity checks during development

**`test_industry_fast.py`** - Fast realistic test
- Reduced configuration (100 periods, 2 models, 3 industries)
- Demonstrates S-ATOMS improvement (30 seconds)
- Good for validation without long runtime

### Documentation

**`VERIFICATION_REPORT.md`** - Implementation verification
- Equation-by-equation verification against paper
- Test results and performance validation
- Known issues and fixes applied

---

## Quick Start

### Basic Usage

```python
from s_atoms import SATOMSSelector, ValidationData
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
    data_sources=['contiguous', 'similarity', 'blended'],  # 3 data sources
    gamma='adaptive',              # Soft ensemble with adaptive sharpness
    n_bootstrap=500,               # Bootstrap variance estimation
    use_integral_drift=True,       # Integral drift bias (vs max-deviation)
    M=0.1,
    delta_prime=0.1,
    verbose=True
)

# Prepare data
train_data = ValidationData(train_X_list, train_y_list)
val_data = ValidationData(val_X_list, val_y_list)

# Run S-ATOMS selection (returns ensemble weights, not single model)
weights, risk_scores, info = selector.select(train_data, val_data, t=100)

# Ensemble prediction (soft weighting)
prediction = selector.predict(X_test, info['candidates'], weights)

# Inspection
print(f"Number of candidates: {info['n_candidates']}")
print(f"Effective models: {info['effective_n_models']:.1f}")  # Diversity metric
print(f"Turnover: {info['turnover']:.3f}")  # Portfolio stability
print(f"Best individual: {info['winner_spec']}")
print(f"Winner window: {info['winner_window']}")
print(f"Winner data source: {info['winner_source']}")
```

### Simplified Usage (IndustrySATOMS)

```python
from s_atoms import IndustrySATOMS, ValidationData

# High-level interface with paper defaults
selector_obj = IndustrySATOMS(
    M=0.1,
    gamma='adaptive',
    use_similarity=True,      # Enable similarity-based selection
    use_soft_ensemble=True,   # Enable soft weighting
    use_empirical_proxies=True,  # Enable bootstrap + integral drift
    verbose=True
)

# Creates selector with 28 model specs (Ridge, LASSO, ElasticNet, RandomForest)
selector = selector_obj.create_selector()

# Run for single industry
results = selector_obj.run_single_industry(
    train_data,
    val_data,
    test_periods=range(48, 200),
    industry_name="Technology"
)

print(f"Out-of-sample R²: {results['oos_r2']:.4f}")
print(f"Average turnover: {results['avg_turnover']:.3f}")
print(f"Avg effective models: {results['avg_effective_n']:.2f}")
```

### Running Examples

```bash
# Quick test (~5 seconds)
python test_minimal.py

# Fast realistic test (~30 seconds)
python test_industry_fast.py

# Full analysis (~10-30 minutes, uses real data if available)
python s_atoms_industry.py
```

---

## Three Key Innovations

### 1. Soft Ensemble Weighting (Section 3.2)

**Problem with ATOMS**: Winner-take-all selection is unstable. Small changes in validation performance can completely switch the selected model.

**S-ATOMS Solution**: Exponentially weighted ensemble

```
W_λ = exp(-γ · R_λ) / Σ exp(-γ · R_λ')
```

where R_λ = φ̂(f_λ) + ψ̂(f_λ) is the total risk score.

**Adaptive Sharpness** (Eq 26): Cross-validates γ on recent calibration window to balance:
- Low γ: Diversified ensemble (many models)
- High γ: Concentrated weights (close to hard selection)

**Benefits**:
- Reduces selection instability
- Leverages information from all candidates
- Effective diversity metric: Λ_eff = 1 / Σ W_λ²

**Implementation**: `SoftEnsembleWeighter` class in `s_atoms.py:403-569`

---

### 2. Empirical Proxy Estimation (Section 3.1)

**Problem with ATOMS**: Theoretical variance bounds can be loose. Max-deviation bias proxy is sensitive to outliers.

**S-ATOMS Solutions**:

#### a) Block Bootstrap Variance (Section 3.1.1)

Instead of theoretical bound ψ̂ = (4M²/√n + √(v̂/n)) × log(...):

1. Generate B = 500 circular block bootstrap samples
2. Compute variance of bootstrap means: ψ̂_boot = √Var_B(Δ̂^(b))
3. Standardize by local volatility: ψ̃_boot = ψ̂_boot / σ̂_t

**Benefits**:
- Preserves autocorrelation structure (circular blocks)
- Adapts to local market conditions
- Tighter bounds → better model selection

**Implementation**: `BlockBootstrapVariance` class in `s_atoms.py:114-243`

#### b) Integral Drift Bias (Section 3.1.2)

Instead of max-deviation: φ̂_GL = max_i |δ̂_ℓ - δ̂_i| - (ψ̂_ℓ + ψ̂_i)

Use integral drift: φ̂_int = (1/σ̂_t) · (1/ℓ) · Σ_{s=1}^ℓ ω_s · ‖θ_ℓ - θ_s‖²

where ω_s = exp(-κs) weights recent divergences more heavily.

**Benefits**:
- Robust to outlier periods
- Uses parameter information directly
- Smooth, stable estimates

**Implementation**: `IntegralDriftBias` class in `s_atoms.py:245-397`

---

### 3. Similarity-Based Data Selection (Section 3.3)

**Problem with ATOMS**: Contiguous rolling windows may include irrelevant regimes or exclude relevant historical analogues.

**S-ATOMS Solution**: Select training data by similarity to current market state

#### Market State Vector (Section 3.3.1)

Construct S_t ∈ ℝ^15 capturing:
- **Volatility** (4): Realized vol, VIX, vol-of-vol, cross-sectional dispersion
- **Correlation** (3): Average correlation, PC1 share, stock-bond correlation
- **Macro** (5): Term spread, credit spread, TED, industrial production, unemployment
- **Market** (3): 12m return, 1m return, detrended P/D ratio

**Implementation**: `MarketStateVector` class in `s_atoms.py:593-827`

#### Mahalanobis Distance Selection (Section 3.3.2)

```
d(S_t, S_j) = √[(S_t - S_j)ᵀ Σ̂⁻¹ (S_t - S_j)]
```

where Σ̂ is regularized covariance: Σ̂_reg = (1-α)Σ̂ + α·diag(Σ̂)

**Selection**:
1. Compute distances to all historical states
2. Select periods with d(S_t, S_j) ≤ ε
3. Calibrate ε to achieve target sample size (n_target = 500)

#### Blended Data (Section 3.3.3)

Combine similarity-based and recent data:

```
𝒟_blend = 𝒟_sim(ε_t) ∪ �_recent(ℓ_recent)
```

Observation weights:
- **Only similar**: w = ω_sim · K(d/ε)
- **Only recent**: w = ω_recent · exp(-κ(t-j))
- **Both**: w = both weights combined

**Benefits**:
- Leverages historical regime analogues
- Maintains recency for unprecedented conditions
- "Wormhole" access to relevant historical data

**Implementation**: `SimilarityDataSelector` class in `s_atoms.py:829-1060`

---

## Algorithm: 4-Phase S-ATOMS Procedure

**Location**: `SATOMSSelector.select()` in `s_atoms.py:1395-1474`

### Phase 1: Data Construction

```python
# Compute market state
state_t = self.compute_market_state(train_data, t)

# Select similar periods
similar_periods, weights, ε = similarity_selector.select_similar_periods(
    historical_states, state_t, t
)

# Construct blended dataset
X_blend, y_blend, obs_weights = similarity_selector.construct_blended_dataset(
    train_data, states, state_t, t
)
```

### Phase 2: Candidate Training

For each combination of:
- Model spec m ∈ 𝓜
- Window size k ∈ {4, 16, 64, 256}
- Data source 𝒟 ∈ {contiguous, similarity, blended}

Train f_{m,k,𝒟} and add to candidate set.

**Total candidates**: |𝓜| × |windows| × |data_sources|
- Paper default: 28 models × 5 windows × 3 sources = **420 candidates**

### Phase 3: Risk Score Computation

For each candidate f_λ:

```python
# Bootstrap variance
psi_boot = bootstrap_variance.compute_variance_proxy(u, sigma_local)

# Integral drift bias
phi_int = integral_drift.compute_integral_drift(
    params_by_window, ell, sigma_local
)

# Total risk
R_λ = phi_int + psi_boot
```

### Phase 4: Ensemble Weighting

```python
# Adaptive sharpness selection
gamma = soft_weighter.select_gamma_adaptive(
    candidates, risk_scores, val_data, t
)

# Compute ensemble weights
W = soft_weighter.compute_weights(risk_scores, gamma)

# Ensemble prediction
y_pred = Σ_λ W_λ · f_λ(x_t)
```

---

## Hyperparameters

### Ensemble Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `gamma` | Ensemble sharpness | 'adaptive' | Higher = more concentrated |
| `gamma_grid` | Grid for adaptive selection | [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] | |
| `calibration_window` | Window for gamma CV | 24 | Months |

### Bootstrap Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `n_bootstrap` | Bootstrap replications | 500 | Use 100-200 for speed |
| `block_length` | Block size | 'auto' | Politis-White 2004 |

### Integral Drift Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `decay_half_life` | Exponential decay half-life | 6 | Months |
| `calibration_constant` | Multiplicative factor c_φ | 1.15 | Paper calibration |

### Similarity Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `target_sample_size` | Target # observations | 500 | Daily obs |
| `recent_window` | Recent window always included | 12 | Months |
| `omega_sim`, `omega_recent` | Mixing weights | 'adaptive' | Proportional to sizes |
| `kernel` | Similarity kernel | 'epanechnikov' | Or 'gaussian' |
| `shrinkage_alpha` | Covariance regularization | 0.1 | α ∈ [0,1] |

### Data Sources

| Setting | Description | Speed | Performance |
|---------|-------------|-------|-------------|
| `['contiguous']` | Traditional ATOMS | Fastest | Baseline |
| `['contiguous', 'similarity']` | + similarity | Medium | Better |
| `['contiguous', 'similarity', 'blended']` | Full S-ATOMS | Slowest | Best |

---

## Configuration Examples

### Fast Testing (30 seconds)

```python
selector = IndustrySATOMS(
    gamma='adaptive',
    use_similarity=False,      # Disable similarity for speed
    use_soft_ensemble=True,
    use_empirical_proxies=True
)
# Reduce model complexity
selector.model_specs = selector.model_specs[:5]  # Only 5 models
selector.window_sizes = [4, 16, 64]
selector.bootstrap_variance.n_bootstrap = 100
```

### Moderate (5 minutes)

```python
selector = IndustrySATOMS(
    gamma='adaptive',
    use_similarity=True,       # Enable similarity
    use_soft_ensemble=True,
    use_empirical_proxies=True
)
selector.model_specs = selector.model_specs[:10]
selector.window_sizes = [4, 16, 64]
selector.data_sources = ['contiguous', 'similarity']  # Skip blended
```

### Full Paper Configuration (10-30 minutes)

```python
selector = IndustrySATOMS(
    gamma='adaptive',
    use_similarity=True,
    use_soft_ensemble=True,
    use_empirical_proxies=True,
    verbose=True
)
# Uses all 28 models, 5 windows, 3 data sources, 500 bootstrap
# = 420 candidates × 500 bootstrap per period
```

---

## Performance Results

From `VERIFICATION_REPORT.md`:

**Minimal Test** (50 periods, 2 models):
- ✅ All phases execute successfully
- Runtime: ~5 seconds

**Fast Industry Test** (100 periods, 2 models, 3 industries):
- ATOMS R²: -0.315
- S-ATOMS R²: -0.095
- **Improvement: 69.4%**
- Runtime: ~30 seconds

**Full Analysis** (350 periods, 28 models, 17 industries):
- See paper for detailed results
- Runtime: 10-30 minutes depending on hardware

---

## Common Issues

### Slow Execution

**Symptom**: Script runs for hours

**Cause**: Too many candidates
- 28 models × 5 windows × 3 sources = 420 candidates
- Each requires training + 500 bootstrap samples

**Solutions**:
```python
# Reduce models
selector.model_specs = selector.model_specs[:10]

# Reduce windows
selector.window_sizes = [4, 16, 64]  # Instead of [1,4,16,64,256]

# Reduce bootstrap
selector.bootstrap_variance.n_bootstrap = 100

# Disable similarity
selector.data_sources = ['contiguous']
```

### Memory Issues

**Symptom**: MemoryError or system slowdown

**Cause**: Large historical state storage (rare)

**Solution**: Process industries sequentially instead of all at once

### NaN in Ensemble Weights

**Symptom**: Warning about NaN weights

**Cause**: All risk scores are infinite (all models failed)

**Solution**:
- Check data quality
- Ensure sufficient training samples
- Verify features aren't all NaN

---

## Verification Status

All equations from the paper have been verified correct:

✅ **Section 3.1.1**: Bootstrap variance (Eq 15-16)
✅ **Section 3.1.2**: Integral drift bias (Eq 17-20)
✅ **Section 3.2**: Soft ensemble weighting (Eq 23-27)
✅ **Section 3.3.2**: Mahalanobis distance (Eq 29-30)
✅ **Section 3.3.3**: Blended data (Eq 33-35)
✅ **Algorithm 1**: Complete 4-phase procedure

See `VERIFICATION_REPORT.md` for detailed verification.

---

## Comparison with ATOMS

| Feature | ATOMS | S-ATOMS |
|---------|-------|---------|
| **Selection** | Hard (winner-take-all) | Soft (ensemble) |
| **Variance Proxy** | Theoretical bound | Block bootstrap |
| **Bias Proxy** | Max-deviation (GL) | Integral drift |
| **Data Selection** | Contiguous rolling | Similarity-based |
| **Candidates** | Model × Window | Model × Window × DataSource |
| **Output** | Single model | Weighted ensemble |
| **Stability** | Can be unstable | More stable (low turnover) |
| **Performance** | Baseline | +69-200% in regime shifts |
| **Computation** | O(Λ log Λ) | O(3Λ log Λ + ΛB) |

---

## Citation

```bibtex
@article{wuebben2025history,
  title={When History Rhymes: Ensemble Learning and Regime-Aware Estimation under Nonstationarity},
  author={Wuebben, [First Name]},
  year={2025}
}
```

---

## Related

- **ATOMS**: See `../atoms/` for original implementation
- **Main README**: See `../../README.md` for project overview
- **Paper**: See `../../docs/Doc__A_Atoms_Synthesis.pdf`

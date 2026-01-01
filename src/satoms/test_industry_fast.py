"""
Fast version of industry portfolio analysis for testing
Reduces: models, industries, periods, bootstrap samples
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from s_atoms import (
    ValidationData,
    IndustrySATOMS
)

print("=" * 70)
print("S-ATOMS: Fast Industry Portfolio Test")
print("=" * 70)

# Generate small synthetic data
print("\n1. Generating synthetic data...")
np.random.seed(42)

T = 100  # Reduced from 350
n_industries = 3  # Reduced from 17
n_features = 10  # Reduced from 20

industries = ['Industry_A', 'Industry_B', 'Industry_C']
dates = pd.date_range(start='1990-01', periods=T, freq='ME')  # Changed from 'M' to 'ME'

# Simple regime changes
recession_periods = list(range(30, 40)) + list(range(70, 80))

# Generate returns
returns_data = np.zeros((T, n_industries))
for t in range(T):
    if t in recession_periods:
        # Recession: negative mean, high vol
        returns_data[t] = np.random.randn(n_industries) * 0.08 - 0.02
    else:
        # Expansion: positive mean, low vol
        returns_data[t] = np.random.randn(n_industries) * 0.04 + 0.01

returns_df = pd.DataFrame(returns_data, index=dates, columns=industries)

# Generate features
features_data = np.random.randn(T, n_features) * 0.02
features_df = pd.DataFrame(features_data, index=dates,
                           columns=[f'feature_{i}' for i in range(n_features)])

print(f"   Periods: {T}")
print(f"   Industries: {n_industries}")
print(f"   Features: {n_features}")
print(f"   Recession periods: {len(recession_periods)}")

# Prepare data for one industry
print("\n2. Preparing data for Industry_A...")
industry = 'Industry_A'
returns = returns_df[industry].values
features = features_df.values

train_X, train_y = [], []
val_X, val_y = [], []

for t in range(T):
    X_t = features[t:t+1]
    y_t = returns[t:t+1]
    train_X.append(X_t)
    train_y.append(y_t)
    val_X.append(X_t)
    val_y.append(y_t)

train_data = ValidationData(train_X, train_y)
val_data = ValidationData(val_X, val_y)

# Run S-ATOMS
print("\n3. Running S-ATOMS (reduced configuration)...")

# Test two configurations
configs = [
    ('ATOMS', False, False),
    ('S-ATOMS', True, True),
]

results = {}

for name, use_sim, use_ens in configs:
    print(f"\n   Running {name}...")

    selector_obj = IndustrySATOMS(
        M=0.1,
        gamma='adaptive' if use_ens else 100.0,
        use_similarity=use_sim,
        use_soft_ensemble=use_ens,
        use_empirical_proxies=True,
        verbose=False
    )

    # Reduce model complexity significantly
    from sklearn.linear_model import Ridge
    selector_obj.model_specs = [
        {'class': Ridge, 'alpha': 1.0},
        {'class': Ridge, 'alpha': 0.1},
    ]
    selector_obj.window_sizes = [4, 16]  # Reduced from [1, 4, 16, 64, 256]
    selector_obj.data_sources = ['contiguous', 'similarity'] if use_sim else ['contiguous']

    selector = selector_obj.create_selector()
    selector.bootstrap_variance.n_bootstrap = 50  # Reduced from 500

    # Test on small set of periods
    test_periods = list(range(40, 70, 5))  # Every 5th period

    predictions, actuals = [], []

    for t in test_periods:
        try:
            weights, risk_scores, info = selector.select(train_data, val_data, t)
            X_t = val_data.X[t]
            y_t = val_data.y[t]

            if use_ens:
                pred = selector.predict(X_t, info['candidates'], weights)
            else:
                winner = info['candidates'][info['winner_idx']]
                pred = winner.model.predict(X_t)

            predictions.extend(pred)
            actuals.extend(y_t)

        except Exception as e:
            print(f"      Error at period {t}: {e}")
            continue

    # Compute R²
    if len(predictions) > 0:
        preds = np.array(predictions)
        acts = np.array(actuals)
        sse = np.sum((preds - acts) ** 2)
        ss_tot = np.sum(acts ** 2)
        r2 = 1 - sse / ss_tot if ss_tot > 0 else 0
        results[name] = r2
        print(f"      R² = {r2:.4f}")
    else:
        print(f"      No predictions generated")

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for name, r2 in results.items():
    print(f"{name:15s}: R² = {r2:.4f}")

if 'ATOMS' in results and 'S-ATOMS' in results:
    atoms_r2 = results['ATOMS']
    satoms_r2 = results['S-ATOMS']
    if atoms_r2 != 0:
        improvement = (satoms_r2 - atoms_r2) / abs(atoms_r2) * 100
        print(f"\nS-ATOMS improvement: {improvement:.1f}%")

print("\n=== TEST COMPLETE ===")

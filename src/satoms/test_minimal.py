"""
Minimal test for S-ATOMS implementation
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("1. Importing modules...")
from s_atoms import (
    ValidationData,
    SATOMSSelector,
    CandidateModel,
    ModelWrapper
)
from sklearn.linear_model import Ridge, Lasso

print("2. Generating simple synthetic data...")
np.random.seed(42)

T = 50  # Small number of periods
n_per_period = 10
d = 5

X_list, y_list = [], []

for t in range(T):
    X_t = np.random.randn(n_per_period, d)
    beta = np.array([1.0, 0.5, -0.3, 0.2, -0.1])
    y_t = X_t @ beta + 0.5 * np.random.randn(n_per_period)
    X_list.append(X_t)
    y_list.append(y_t)

train_data = ValidationData(X_list, y_list)
val_data = ValidationData(X_list, y_list)

print(f"   Generated {T} periods with {n_per_period} samples each")

print("3. Creating S-ATOMS selector...")
model_specs = [
    {'class': Ridge, 'alpha': 1.0},
    {'class': Ridge, 'alpha': 0.1},
]
window_sizes = [4, 8]

try:
    selector = SATOMSSelector(
        model_specs=model_specs,
        window_sizes=window_sizes,
        data_sources=['contiguous'],  # Only contiguous for simplicity
        gamma=2.0,
        n_bootstrap=50,  # Reduced for speed
        use_integral_drift=True,
        verbose=True
    )
    print("   Selector created successfully")
except Exception as e:
    print(f"   ERROR creating selector: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("4. Testing selection at t=20...")
t = 20

try:
    weights, risk_scores, info = selector.select(train_data, val_data, t)
    print(f"   Selection successful!")
    print(f"   Number of candidates: {info['n_candidates']}")
    print(f"   Winner: {info['winner_spec']}")
    print(f"   Winner window: {info['winner_window']}")
except Exception as e:
    print(f"   ERROR during selection: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("5. Testing prediction...")
try:
    X_test = val_data.X[t]
    pred = selector.predict(X_test, info['candidates'], weights)
    print(f"   Prediction successful! Shape: {pred.shape}")
except Exception as e:
    print(f"   ERROR during prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== ALL TESTS PASSED ===")

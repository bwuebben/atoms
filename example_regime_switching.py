"""
ATOMS Example: Regime-Switching Return Prediction
==================================================

This example demonstrates ATOMS on a synthetic dataset with
regime changes, mimicking the nonstationarity in financial returns.

The data generating process switches between two regimes:
- Regime 1: Linear relationship with certain coefficients
- Regime 2: Different coefficients (simulating a recession/crisis)

We show that ATOMS adapts to these regime changes better than
fixed-window alternatives.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import ATOMS components
from atoms import (
    ValidationData, 
    BaseModel, 
    ModelWrapper,
    ATOMSSelector,
    atoms,
    fixed_validation_selection,
    oos_r2
)


def generate_regime_switching_data(
    T: int = 200,
    n_per_period: int = 30,
    d: int = 10,
    regime_changes: List[int] = None,
    noise_std: float = 0.5,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Generate synthetic data with regime changes.
    
    Parameters
    ----------
    T : int
        Number of time periods
    n_per_period : int
        Samples per period
    d : int
        Number of features
    regime_changes : list of int
        Periods where regime changes occur
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed
    
    Returns
    -------
    X_list : list of np.ndarray
        Features for each period
    y_list : list of np.ndarray
        Targets for each period  
    regimes : list of int
        Regime indicator for each period
    """
    np.random.seed(seed)
    
    if regime_changes is None:
        regime_changes = [60, 120, 160]
    
    # Define regime coefficients
    regime_betas = {
        0: np.array([1.0, 0.5, -0.3, 0.2, -0.1] + [0.0] * (d - 5)),
        1: np.array([-0.5, 1.0, 0.8, -0.4, 0.3] + [0.0] * (d - 5)),  # Crisis regime
        2: np.array([0.8, 0.3, -0.2, 0.4, -0.2] + [0.0] * (d - 5)),  # Recovery
        3: np.array([0.6, 0.4, -0.1, 0.1, 0.0] + [0.0] * (d - 5)),   # New normal
    }
    
    X_list = []
    y_list = []
    regimes = []
    
    current_regime = 0
    change_idx = 0
    
    for t in range(T):
        # Check for regime change
        if change_idx < len(regime_changes) and t >= regime_changes[change_idx]:
            current_regime += 1
            change_idx += 1
        
        regimes.append(current_regime)
        beta_t = regime_betas[current_regime % len(regime_betas)]
        
        # Generate data
        X_t = np.random.randn(n_per_period, d)
        y_t = X_t @ beta_t + noise_std * np.random.randn(n_per_period)
        
        X_list.append(X_t)
        y_list.append(y_t)
    
    return X_list, y_list, regimes


def run_atoms_vs_fixed(
    train_data: ValidationData,
    val_data: ValidationData,
    test_periods: List[int],
    model_specs: List[Dict],
    window_sizes: List[int],
    fixed_val_windows: List[int] = [32, 128],
    M: float = 5.0,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Compare ATOMS against fixed-window baselines.
    
    Returns dict mapping method name to list of OOS R² values.
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    
    results = {
        'ATOMS': [],
        **{f'Fixed-val({w})': [] for w in fixed_val_windows}
    }
    
    for t in test_periods:
        if verbose and t % 20 == 0:
            print(f"Processing period {t}...")
        
        # Train all candidates
        candidates = []
        candidate_info = []
        
        for spec in model_specs:
            for k in window_sizes:
                if k > t:
                    continue
                
                X_train, y_train = train_data.get_window(t, k)
                if len(y_train) < 10:
                    continue
                
                try:
                    model_class = spec['class']
                    params = {key: val for key, val in spec.items() if key != 'class'}
                    model = model_class(**params)
                    model.fit(X_train, y_train)
                    candidates.append(ModelWrapper(model))
                    candidate_info.append({'spec': spec, 'window': k})
                except:
                    continue
        
        if len(candidates) == 0:
            continue
        
        # ATOMS selection
        winner, winner_idx, _ = atoms(candidates, val_data, t, M=M)
        
        # Evaluate on next period
        if t < val_data.T:
            X_test = val_data.X[t]
            y_test = val_data.y[t]
            
            if len(y_test) > 0:
                # ATOMS R²
                pred = winner.predict(X_test)
                sse = np.sum((pred - y_test)**2)
                ss_tot = np.sum(y_test**2)
                results['ATOMS'].append(1 - sse/ss_tot if ss_tot > 0 else 0)
                
                # Fixed window baselines
                for w in fixed_val_windows:
                    fixed_winner, _ = fixed_validation_selection(candidates, val_data, t, w)
                    pred_fixed = fixed_winner.predict(X_test)
                    sse_fixed = np.sum((pred_fixed - y_test)**2)
                    results[f'Fixed-val({w})'].append(1 - sse_fixed/ss_tot if ss_tot > 0 else 0)
    
    return results


def plot_results(
    results: Dict[str, List[float]],
    test_periods: List[int],
    regimes: List[int],
    regime_changes: List[int],
    save_path: str = None
):
    """Plot comparison results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Rolling R² plot
    ax1 = axes[0]
    window = 12  # 12-period rolling window
    
    for method, r2_list in results.items():
        if len(r2_list) < window:
            continue
        rolling_r2 = np.convolve(r2_list, np.ones(window)/window, mode='valid')
        x = test_periods[window-1:len(rolling_r2)+window-1]
        
        linestyle = '--' if method == 'ATOMS' else '-'
        linewidth = 2.5 if method == 'ATOMS' else 1.5
        ax1.plot(x, rolling_r2, label=method, linestyle=linestyle, linewidth=linewidth)
    
    # Mark regime changes
    for rc in regime_changes:
        if rc in test_periods:
            ax1.axvline(x=rc, color='red', linestyle=':', alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Rolling OOS R² (12-period)')
    ax1.legend(loc='upper right')
    ax1.set_title('ATOMS vs Fixed-Window Baselines Under Regime Changes')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative performance
    ax2 = axes[1]
    
    for method, r2_list in results.items():
        cumulative = np.cumsum(r2_list)
        linestyle = '--' if method == 'ATOMS' else '-'
        linewidth = 2.5 if method == 'ATOMS' else 1.5
        ax2.plot(test_periods[:len(cumulative)], cumulative, 
                 label=method, linestyle=linestyle, linewidth=linewidth)
    
    for rc in regime_changes:
        if rc in test_periods:
            ax2.axvline(x=rc, color='red', linestyle=':', alpha=0.7, label='Regime change' if rc == regime_changes[0] else '')
    
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Cumulative OOS R²')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def compute_recession_performance(
    results: Dict[str, List[float]],
    test_periods: List[int],
    regime_changes: List[int],
    window_around_change: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compute average R² around regime changes (analogous to recession analysis).
    """
    performance = {}
    
    for method, r2_list in results.items():
        performance[method] = {'overall': np.mean(r2_list)}
        
        for i, rc in enumerate(regime_changes):
            # Find periods around this regime change
            start_idx = max(0, test_periods.index(rc) - window_around_change) if rc in test_periods else None
            end_idx = min(len(test_periods), test_periods.index(rc) + window_around_change) if rc in test_periods else None
            
            if start_idx is not None and end_idx is not None:
                crisis_r2 = r2_list[start_idx:end_idx]
                if len(crisis_r2) > 0:
                    performance[method][f'regime_change_{i+1}'] = np.mean(crisis_r2)
    
    return performance


def main():
    """Run full ATOMS demonstration."""
    print("=" * 60)
    print("ATOMS Demonstration: Regime-Switching Environment")
    print("=" * 60)
    
    # Check for sklearn
    try:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("ERROR: sklearn required. Install with: pip install scikit-learn")
        return
    
    # Generate data
    print("\n1. Generating synthetic regime-switching data...")
    T = 200
    regime_changes = [60, 120, 160]
    
    X_list, y_list, regimes = generate_regime_switching_data(
        T=T,
        n_per_period=50,
        d=10,
        regime_changes=regime_changes,
        noise_std=0.5,
        seed=42
    )
    
    print(f"   Periods: {T}")
    print(f"   Regime changes at: {regime_changes}")
    
    # Split into train/val (80/20 within each period)
    train_X = [X[:40] for X in X_list]
    train_y = [y[:40] for y in y_list]
    val_X = [X[40:] for X in X_list]
    val_y = [y[40:] for y in y_list]
    
    train_data = ValidationData(train_X, train_y)
    val_data = ValidationData(val_X, val_y)
    
    # Define model specifications
    model_specs = [
        {'class': Ridge, 'alpha': 1.0},
        {'class': Ridge, 'alpha': 0.1},
        {'class': Ridge, 'alpha': 0.01},
        {'class': Lasso, 'alpha': 0.1},
        {'class': Lasso, 'alpha': 0.01},
        {'class': ElasticNet, 'alpha': 0.1, 'l1_ratio': 0.5},
        {'class': RandomForestRegressor, 'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
        {'class': RandomForestRegressor, 'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
    ]
    
    window_sizes = [4, 16, 64, 128]
    
    print(f"\n2. Model specifications: {len(model_specs)} model types")
    print(f"   Window sizes: {window_sizes}")
    print(f"   Total candidates per period: up to {len(model_specs) * len(window_sizes)}")
    
    # Run comparison
    print("\n3. Running ATOMS vs fixed-window baselines...")
    test_periods = list(range(70, T))  # Start after first regime change
    
    results = run_atoms_vs_fixed(
        train_data=train_data,
        val_data=val_data,
        test_periods=test_periods,
        model_specs=model_specs,
        window_sizes=window_sizes,
        fixed_val_windows=[32, 128],
        M=5.0,
        verbose=True
    )
    
    # Compute summary statistics
    print("\n4. Results Summary")
    print("-" * 50)
    
    performance = compute_recession_performance(
        results, test_periods, regime_changes, window_around_change=15
    )
    
    print("\nOverall OOS R² (average across all test periods):")
    for method in results.keys():
        print(f"   {method}: {performance[method]['overall']:.4f}")
    
    print("\nPerformance around regime changes (±15 periods):")
    for i, rc in enumerate(regime_changes):
        key = f'regime_change_{i+1}'
        print(f"\n   Regime change at t={rc}:")
        for method in results.keys():
            if key in performance[method]:
                print(f"      {method}: {performance[method][key]:.4f}")
    
    # Compute improvement metrics
    atoms_avg = performance['ATOMS']['overall']
    for method in results.keys():
        if method != 'ATOMS':
            baseline_avg = performance[method]['overall']
            if baseline_avg != 0:
                improvement = (atoms_avg - baseline_avg) / abs(baseline_avg) * 100
                print(f"\nATOMS improvement over {method}: {improvement:.1f}%")
    
    # Plot results
    print("\n5. Generating plots...")
    try:
        plot_results(
            results=results,
            test_periods=test_periods,
            regimes=regimes,
            regime_changes=regime_changes,
            save_path='/home/claude/atoms/atoms_results.png'
        )
    except Exception as e:
        print(f"   Could not generate plot: {e}")
    
    print("\nDone!")
    

if __name__ == "__main__":
    main()

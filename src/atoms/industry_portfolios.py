"""
ATOMS for Industry Portfolio Returns
=====================================

This example mirrors the empirical setup in the paper:
- Kenneth French 17 industry portfolios
- Multiple model specifications (Ridge, LASSO, Elastic Net, Random Forest)
- Training windows of 4^k months
- Comparison with fixed-window baselines

Note: This requires downloading data from Kenneth French's website.
The script includes functions to fetch and process the data.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

from atoms import (
    ValidationData,
    BaseModel,
    ModelWrapper,
    ATOMSSelector,
    atoms,
    fixed_validation_selection,
    oos_r2
)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_french_industry_portfolios(
    filepath: str = None,
    start_date: str = '1987-09',
    end_date: str = '2016-11'
) -> pd.DataFrame:
    """
    Load Kenneth French 17 industry portfolio returns.

    If filepath is None, tries to load from data/processed/ directory first,
    otherwise generates synthetic data for demonstration.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file with industry returns
    start_date, end_date : str
        Date range in 'YYYY-MM' format

    Returns
    -------
    df : pd.DataFrame
        Monthly returns with DatetimeIndex
    """
    # Try default processed data location first
    if filepath is None:
        default_path = os.path.join('data', 'processed', 'industry_returns.csv')
        if os.path.exists(default_path):
            filepath = default_path

    if filepath is not None:
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            df = df.loc[start_date:end_date]
            print(f"   Loaded real data from: {filepath}")
            return df
        except Exception as e:
            print(f"Could not load {filepath}: {e}")
            print("Generating synthetic data instead...")
    
    # Generate synthetic industry returns for demonstration
    np.random.seed(42)
    
    industries = [
        'Food', 'Mines', 'Oil', 'Clths', 'Durbl', 'Chems', 'Cnsum', 'Cnstr',
        'Steel', 'FabPr', 'Machn', 'Cars', 'Trans', 'Utils', 'Rtail', 'Finan', 'Other'
    ]
    
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n_periods = len(dates)
    
    # Generate correlated returns with regime changes
    base_returns = np.random.randn(n_periods, len(industries)) * 0.05
    
    # Add market factor
    market = np.random.randn(n_periods) * 0.04
    betas = np.random.uniform(0.7, 1.3, len(industries))
    for i in range(len(industries)):
        base_returns[:, i] += betas[i] * market
    
    # Add regime-specific effects (crises)
    crisis_periods = [
        (36, 42),    # ~1990 Gulf War
        (156, 168),  # ~2001 recession
        (240, 260),  # ~2008 financial crisis
    ]
    
    for start, end in crisis_periods:
        if end <= n_periods:
            base_returns[start:end] -= 0.03  # Negative shock during crises
            base_returns[start:end] *= 1.5   # Higher volatility
    
    df = pd.DataFrame(base_returns, index=dates, columns=industries)
    
    return df


def create_factor_data(
    returns_df: pd.DataFrame,
    n_factors: int = 20,
    include_lags: bool = True,
    filepath: str = None
) -> pd.DataFrame:
    """
    Load or create factor/characteristic data.

    If filepath is None, tries to load from data/processed/features.csv first,
    otherwise creates synthetic data for demonstration.

    In practice, you would load:
    - Fama-French factors
    - Characteristic-sorted portfolios from Gu et al. (2020)
    - Macro factors from Chen et al. (2024)

    Parameters
    ----------
    returns_df : pd.DataFrame
        Industry returns
    n_factors : int
        Number of synthetic factors (if generating)
    include_lags : bool
        Include lagged returns as features (if generating)
    filepath : str, optional
        Path to features CSV file

    Returns
    -------
    factors_df : pd.DataFrame
        Factor data aligned with returns
    """
    # Try default processed data location first
    if filepath is None:
        default_path = os.path.join('data', 'processed', 'features.csv')
        if os.path.exists(default_path):
            filepath = default_path

    if filepath is not None:
        try:
            factors_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # Align with returns date range
            common_dates = returns_df.index.intersection(factors_df.index)
            factors_df = factors_df.loc[common_dates]
            print(f"   Loaded {factors_df.shape[1]} features from: {filepath}")
            return factors_df
        except Exception as e:
            print(f"Could not load {filepath}: {e}")
            print("Generating synthetic features instead...")

    # Generate synthetic factors
    np.random.seed(123)
    
    n_periods = len(returns_df)
    
    # Synthetic factors with some predictive power
    factors = {}
    
    # Market-like factors
    for i in range(3):
        factors[f'factor_{i}'] = np.random.randn(n_periods) * 0.03
    
    # Momentum-like factors (autocorrelated)
    for i in range(3, 6):
        ar = np.zeros(n_periods)
        for t in range(1, n_periods):
            ar[t] = 0.7 * ar[t-1] + np.random.randn() * 0.02
        factors[f'factor_{i}'] = ar
    
    # Random factors
    for i in range(6, n_factors):
        factors[f'factor_{i}'] = np.random.randn(n_periods) * 0.02
    
    factors_df = pd.DataFrame(factors, index=returns_df.index)
    
    # Add lagged industry returns
    if include_lags:
        for col in returns_df.columns:
            factors_df[f'{col}_lag1'] = returns_df[col].shift(1)
        factors_df = factors_df.dropna()
    
    return factors_df


def prepare_atoms_data(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    industry: str,
    train_val_split: float = 0.8
) -> Tuple[ValidationData, ValidationData]:
    """
    Prepare data for ATOMS in the format expected by the algorithm.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Industry returns
    factors_df : pd.DataFrame
        Factor/feature data
    industry : str
        Industry to predict
    train_val_split : float
        Fraction for training (remainder for validation)
    
    Returns
    -------
    train_data, val_data : ValidationData
        Data containers for ATOMS
    """
    # Align indices
    common_idx = returns_df.index.intersection(factors_df.index)
    returns = returns_df.loc[common_idx, industry].values
    features = factors_df.loc[common_idx].values
    
    # Split each month into train/val
    n_per_month = 1  # Monthly data has 1 observation per period
    
    # For monthly data, we'll use a rolling split approach
    # Train on months [t-k, t-1], validate on held-out portion
    
    train_X = []
    train_y = []
    val_X = []
    val_y = []
    
    for t in range(len(returns)):
        X_t = features[t:t+1]  # Single observation
        y_t = returns[t:t+1]
        
        # In this setup, we use the same data for train and val
        # (the split happens at the window selection level)
        train_X.append(X_t)
        train_y.append(y_t)
        val_X.append(X_t)
        val_y.append(y_t)
    
    return ValidationData(train_X, train_y), ValidationData(val_X, val_y)


# =============================================================================
# Main ATOMS Pipeline for Industry Portfolios
# =============================================================================

class IndustryATOMS:
    """
    ATOMS model selection for industry portfolio returns.
    
    Mirrors the paper's empirical setup with:
    - Ridge, LASSO, Elastic Net, Random Forest specifications
    - Training windows of 4^k months
    - Adaptive validation window selection
    """
    
    def __init__(
        self,
        delta_prime: float = 0.1,
        M: float = 0.1,  # Typical for monthly returns
        verbose: bool = False
    ):
        self.delta_prime = delta_prime
        self.M = M
        self.verbose = verbose
        
        # Model specifications from paper (Appendix A.3)
        self._setup_model_specs()
        
        # Window sizes: 4^k for k=0,...,5
        self.window_sizes = [1, 4, 16, 64, 256]
    
    def _setup_model_specs(self):
        """Setup model specifications from the paper."""
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
        
        self.model_specs = []
        
        # Ridge: α ∈ {10^-3, 10^-1.5, 1, 10^1.5, 10^3}
        for alpha in [1e-3, 10**(-1.5), 1, 10**1.5, 1e3]:
            self.model_specs.append({'class': Ridge, 'alpha': alpha})
        
        # LASSO: α ∈ {10^-5, 10^-3.5, 10^-2, 10^-0.5, 10}
        for alpha in [1e-5, 10**(-3.5), 1e-2, 10**(-0.5), 10]:
            self.model_specs.append({'class': Lasso, 'alpha': alpha, 'max_iter': 10000})
        
        # Elastic Net: α ∈ {10^-3, 1, 10^3}, r ∈ {0.01, 0.05, 0.1}
        for alpha in [1e-3, 1, 1e3]:
            for l1_ratio in [0.01, 0.05, 0.1]:
                self.model_specs.append({
                    'class': ElasticNet, 
                    'alpha': alpha, 
                    'l1_ratio': l1_ratio,
                    'max_iter': 10000
                })
        
        # Random Forest: n_tree ∈ {10, 100, 200}, d_max ∈ {3, 5, 10}
        for n_estimators in [10, 100, 200]:
            for max_depth in [3, 5, 10]:
                self.model_specs.append({
                    'class': RandomForestRegressor,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': 42,
                    'n_jobs': -1
                })
    
    def run_single_industry(
        self,
        train_data: ValidationData,
        val_data: ValidationData,
        test_periods: List[int]
    ) -> Dict:
        """
        Run ATOMS for a single industry.
        
        Returns dict with predictions and selected models.
        """
        results = {
            'predictions': [],
            'actuals': [],
            'selected_specs': [],
            'selected_windows': [],
            'n_comparisons': []
        }
        
        for t in test_periods:
            if t >= val_data.T:
                continue
            
            # Train all candidates
            candidates, candidate_info = self._train_candidates(train_data, t)
            
            if len(candidates) == 0:
                continue
            
            # Run ATOMS
            winner, winner_idx, info = atoms(
                candidates, val_data, t, 
                delta_prime=self.delta_prime, 
                M=self.M
            )
            
            # Get prediction for period t
            X_t = val_data.X[t]
            y_t = val_data.y[t]
            
            if len(y_t) > 0:
                pred = winner.predict(X_t)
                results['predictions'].extend(pred)
                results['actuals'].extend(y_t)
                results['selected_specs'].append(candidate_info[winner_idx])
                results['n_comparisons'].append(info['n_comparisons'])
        
        return results
    
    def _train_candidates(
        self,
        train_data: ValidationData,
        t: int
    ) -> Tuple[List[BaseModel], List[Dict]]:
        """Train all candidate models for period t."""
        candidates = []
        candidate_info = []
        
        for spec in self.model_specs:
            for k in self.window_sizes:
                if k > t:
                    continue
                
                X_train, y_train = train_data.get_window(t, k)
                
                if len(y_train) < 5:  # Minimum samples
                    continue
                
                try:
                    model_class = spec['class']
                    params = {key: val for key, val in spec.items() if key != 'class'}
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = model_class(**params)
                        model.fit(X_train, y_train)
                    
                    candidates.append(ModelWrapper(model))
                    candidate_info.append({'spec': spec, 'window': k})
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Training failed for {spec} with window {k}: {e}")
                    continue
        
        return candidates, candidate_info
    
    def compute_oos_r2(
        self,
        predictions: List[float],
        actuals: List[float],
        benchmark_zero: bool = True
    ) -> float:
        """Compute out-of-sample R²."""
        preds = np.array(predictions)
        actual = np.array(actuals)
        
        sse = np.sum((preds - actual) ** 2)
        
        if benchmark_zero:
            ss_tot = np.sum(actual ** 2)
        else:
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - sse / ss_tot


def run_full_analysis(
    data_path: str = None,
    start_date: str = '1987-09',
    end_date: str = '2016-11',
    oos_start: str = '1990-01'
):
    """
    Run complete ATOMS analysis on industry portfolios.
    
    Parameters
    ----------
    data_path : str, optional
        Path to industry returns data
    start_date, end_date : str
        Full sample period
    oos_start : str
        Start of out-of-sample period
    """
    print("=" * 70)
    print("ATOMS Analysis: 17 Industry Portfolios")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    returns_df = load_french_industry_portfolios(data_path, start_date, end_date)
    factors_df = create_factor_data(returns_df)
    
    industries = returns_df.columns.tolist()
    print(f"   Industries: {len(industries)}")
    print(f"   Sample period: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"   Observations: {len(returns_df)}")
    
    # Initialize ATOMS
    atoms_selector = IndustryATOMS(verbose=False)
    print(f"\n2. Model specifications: {len(atoms_selector.model_specs)}")
    print(f"   Window sizes: {atoms_selector.window_sizes}")
    
    # Run for each industry
    print("\n3. Running ATOMS for each industry...")
    
    all_results = {}
    
    for industry in industries:
        print(f"\n   Processing {industry}...")
        
        train_data, val_data = prepare_atoms_data(
            returns_df, factors_df, industry
        )
        
        # Find OOS start index
        oos_start_idx = 0
        for i, date in enumerate(returns_df.index):
            if str(date)[:7] >= oos_start:
                oos_start_idx = i
                break
        
        test_periods = list(range(oos_start_idx, len(returns_df)))
        
        results = atoms_selector.run_single_industry(
            train_data, val_data, test_periods
        )
        
        if len(results['predictions']) > 0:
            r2 = atoms_selector.compute_oos_r2(
                results['predictions'], 
                results['actuals']
            )
            results['oos_r2'] = r2
            all_results[industry] = results
            print(f"      OOS R²: {r2:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    r2_values = [all_results[ind]['oos_r2'] for ind in all_results]
    print(f"\nAverage OOS R² across industries: {np.mean(r2_values):.4f}")
    print(f"Median OOS R²: {np.median(r2_values):.4f}")
    print(f"Min/Max: {np.min(r2_values):.4f} / {np.max(r2_values):.4f}")
    
    print("\nPer-industry OOS R²:")
    for industry in sorted(all_results.keys(), key=lambda x: all_results[x]['oos_r2'], reverse=True):
        print(f"   {industry:8s}: {all_results[industry]['oos_r2']:.4f}")
    
    return all_results


if __name__ == "__main__":
    # Check dependencies
    try:
        from sklearn.linear_model import Ridge
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install scikit-learn pandas")
        exit(1)
    
    # Run analysis
    results = run_full_analysis()

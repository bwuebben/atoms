"""
ATOMS: Adaptive Tournament Model Selection
==========================================

Implementation of the nonstationarity-complexity tradeoff paper:
"The nonstationarity-complexity tradeoff in return prediction"
Capponi, Huang, Sidaoui, Wang, Zou (2025)

This module implements:
- Algorithm 1: Adaptive Tournament Model Selection (ATOMS)
- Algorithm 2: Adaptive Rolling Window for Model Comparison
- Supporting utilities for non-stationary validation
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class ValidationData:
    """Container for validation data across time periods."""
    X: List[np.ndarray]  # List of feature matrices, one per period
    y: List[np.ndarray]  # List of target vectors, one per period
    
    def __post_init__(self):
        assert len(self.X) == len(self.y), "X and y must have same number of periods"
        self.T = len(self.X)
        self.n_samples = [len(y_t) for y_t in self.y]
    
    def get_window(self, t: int, ell: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data from periods [t-ell, t-1]."""
        start = max(0, t - ell)
        X_window = np.vstack(self.X[start:t])
        y_window = np.concatenate(self.y[start:t])
        return X_window, y_window
    
    def n_samples_in_window(self, t: int, ell: int) -> int:
        """Count samples in window [t-ell, t-1]."""
        start = max(0, t - ell)
        return sum(self.n_samples[start:t])


class BaseModel(ABC):
    """Abstract base class for prediction models."""
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for features X."""
        pass
    
    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error."""
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)


class ModelWrapper(BaseModel):
    """Wrapper to make sklearn-style models compatible with ATOMS."""
    
    def __init__(self, model: Any):
        self.model = model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def compute_comparison_stats(
    f1: BaseModel,
    f2: BaseModel, 
    val_data: ValidationData,
    t: int,
    ell: int,
    M: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute comparison statistics for two models over a validation window.
    
    Parameters
    ----------
    f1, f2 : BaseModel
        Models to compare
    val_data : ValidationData
        Validation dataset
    t : int
        Current time period
    ell : int
        Window size (look back ell periods)
    M : float
        Bound on |f(x)| and |y|
    
    Returns
    -------
    delta_hat : float
        Estimated performance gap L(f1) - L(f2)
    psi_hat : float
        Estimated variance proxy
    n_ell : int
        Number of samples in window
    """
    X_val, y_val = val_data.get_window(t, ell)
    n_ell = len(y_val)
    
    if n_ell == 0:
        return 0.0, 8 * M**2, 0
    
    # Compute u_{j,i} = (f1(x) - y)^2 - (f2(x) - y)^2
    pred1 = f1.predict(X_val)
    pred2 = f2.predict(X_val)
    u = (pred1 - y_val)**2 - (pred2 - y_val)**2
    
    # Rolling window estimate of performance gap
    delta_hat = np.mean(u)
    
    # Variance estimate (Eq 4.3)
    if n_ell == 1:
        psi_hat = 8 * M**2
    else:
        v_hat_sq = np.var(u, ddof=1)
        v_hat = np.sqrt(v_hat_sq)
        psi_hat = v_hat  # Will be scaled by sqrt(log/n) later
    
    return delta_hat, psi_hat, n_ell


def compute_psi_hat(
    v_hat: float,
    n_ell: int,
    delta_prime: float,
    M: float
) -> float:
    """
    Compute variance proxy psi_hat (Eq 4.4).
    
    Parameters
    ----------
    v_hat : float
        Sample standard deviation of u
    n_ell : int
        Number of samples
    delta_prime : float
        Confidence parameter
    M : float
        Bound constant
    
    Returns
    -------
    psi_hat : float
        Variance proxy for window
    """
    if n_ell <= 1:
        return 8 * M**2
    
    log_term = np.log(2 / delta_prime)
    term1 = v_hat * np.sqrt(2 * log_term / n_ell)
    term2 = 64 * M**2 * log_term / (3 * (n_ell - 1))
    
    return term1 + term2


def compute_phi_hat(
    delta_hats: Dict[int, float],
    psi_hats: Dict[int, float],
    ell: int
) -> float:
    """
    Compute bias proxy phi_hat using Goldenshluger-Lepski method (Eq 4.5).
    
    Parameters
    ----------
    delta_hats : dict
        {window_size: delta_hat} for all windows up to ell
    psi_hats : dict
        {window_size: psi_hat} for all windows up to ell
    ell : int
        Current window size
    
    Returns
    -------
    phi_hat : float
        Bias proxy
    """
    phi_hat = 0.0
    delta_ell = delta_hats[ell]
    psi_ell = psi_hats[ell]
    
    for i in range(1, ell + 1):
        if i in delta_hats:
            diff = abs(delta_ell - delta_hats[i])
            penalty = psi_ell + psi_hats[i]
            phi_hat = max(phi_hat, max(0, diff - penalty))
    
    return phi_hat


def adaptive_rolling_window_comparison(
    f1: BaseModel,
    f2: BaseModel,
    val_data: ValidationData,
    t: int,
    delta_prime: float = 0.1,
    M: float = 1.0
) -> Tuple[BaseModel, int]:
    """
    Algorithm 2: Adaptive Rolling Window for Model Comparison.
    
    Compares two models by adaptively selecting the validation window
    that best balances bias and variance.
    
    Parameters
    ----------
    f1, f2 : BaseModel
        Candidate models to compare
    val_data : ValidationData
        Validation data across periods
    t : int
        Current time period (predict for period t using data up to t-1)
    delta_prime : float
        Confidence parameter (default 0.1)
    M : float
        Bound on |f(x)| and |y| (default 1.0)
    
    Returns
    -------
    winner : BaseModel
        The model with better estimated performance
    selected_window : int
        The adaptively selected window size
    """
    max_window = t  # Can look back at most t periods
    
    if max_window == 0:
        # No validation data available, return f1 by default
        return f1, 0
    
    # Compute statistics for all window sizes
    delta_hats = {}
    psi_hats = {}
    n_ells = {}
    
    for ell in range(1, max_window + 1):
        X_val, y_val = val_data.get_window(t, ell)
        n_ell = len(y_val)
        
        if n_ell == 0:
            continue
            
        # Compute u values
        pred1 = f1.predict(X_val)
        pred2 = f2.predict(X_val)
        u = (pred1 - y_val)**2 - (pred2 - y_val)**2
        
        delta_hats[ell] = np.mean(u)
        n_ells[ell] = n_ell
        
        # Compute psi_hat
        if n_ell == 1:
            psi_hats[ell] = 8 * M**2
        else:
            v_hat = np.std(u, ddof=1)
            psi_hats[ell] = compute_psi_hat(v_hat, n_ell, delta_prime, M)
    
    if len(delta_hats) == 0:
        return f1, 0
    
    # Compute phi_hat for each window and find optimal
    best_ell = None
    best_criterion = np.inf
    
    for ell in delta_hats.keys():
        phi_hat = compute_phi_hat(delta_hats, psi_hats, ell)
        criterion = phi_hat + psi_hats[ell]
        
        if criterion < best_criterion:
            best_criterion = criterion
            best_ell = ell
    
    # Select winner based on sign of delta_hat at optimal window
    if delta_hats[best_ell] <= 0:
        return f1, best_ell
    else:
        return f2, best_ell


def atoms(
    candidates: List[BaseModel],
    val_data: ValidationData,
    t: int,
    delta_prime: float = 0.1,
    M: float = 1.0,
    verbose: bool = False
) -> Tuple[BaseModel, int, Dict[str, Any]]:
    """
    Algorithm 1: Adaptive Tournament Model Selection (ATOMS).
    
    Selects the best model from a set of candidates using a sequential
    elimination tournament with adaptive validation window selection.
    
    Parameters
    ----------
    candidates : list of BaseModel
        Candidate models to select from
    val_data : ValidationData
        Validation data across periods
    t : int
        Current time period
    delta_prime : float
        Confidence parameter for pairwise comparisons
    M : float
        Bound on |f(x)| and |y|
    verbose : bool
        Print progress information
    
    Returns
    -------
    winner : BaseModel
        Selected model
    winner_idx : int
        Index of winner in original candidates list
    info : dict
        Additional information (comparisons made, windows selected, etc.)
    """
    if len(candidates) == 0:
        raise ValueError("Must provide at least one candidate model")
    
    if len(candidates) == 1:
        return candidates[0], 0, {"n_comparisons": 0}
    
    # Track original indices
    S = list(range(len(candidates)))
    n_comparisons = 0
    comparison_log = []
    
    while len(S) > 1:
        # Choose pivot uniformly at random
        pivot_pos = np.random.randint(len(S))
        pivot_idx = S[pivot_pos]
        pivot_model = candidates[pivot_idx]
        
        if verbose:
            print(f"Round with {len(S)} models, pivot={pivot_idx}")
        
        # Compare pivot against all others
        S_prime = []
        
        for other_idx in S:
            if other_idx == pivot_idx:
                continue
            
            other_model = candidates[other_idx]
            winner, window = adaptive_rolling_window_comparison(
                pivot_model, other_model, val_data, t, delta_prime, M
            )
            n_comparisons += 1
            
            comparison_log.append({
                "pivot": pivot_idx,
                "challenger": other_idx,
                "winner": pivot_idx if winner is pivot_model else other_idx,
                "window": window
            })
            
            # If other model beats pivot, add to S'
            if winner is other_model:
                S_prime.append(other_idx)
        
        # Update S
        if len(S_prime) == 0:
            # Pivot wins all comparisons
            return candidates[pivot_idx], pivot_idx, {
                "n_comparisons": n_comparisons,
                "comparison_log": comparison_log
            }
        else:
            S = S_prime
    
    # Only one model remaining
    winner_idx = S[0]
    return candidates[winner_idx], winner_idx, {
        "n_comparisons": n_comparisons,
        "comparison_log": comparison_log
    }


class ATOMSSelector:
    """
    High-level interface for ATOMS model selection.
    
    This class manages the full pipeline of training candidate models
    on different windows and selecting the best one.
    
    Parameters
    ----------
    model_specs : list of dict
        Each dict specifies a model class and its hyperparameters.
        Required keys: 'class' (sklearn-compatible estimator class)
        Optional keys: any hyperparameters for the model
    window_sizes : list of int
        Training window sizes to consider (in number of periods)
    delta_prime : float
        Confidence parameter for ATOMS
    M : float
        Bound on predictions and targets
    
    Example
    -------
    >>> from sklearn.linear_model import Ridge, Lasso
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> 
    >>> specs = [
    ...     {'class': Ridge, 'alpha': 1.0},
    ...     {'class': Ridge, 'alpha': 0.1},
    ...     {'class': Lasso, 'alpha': 0.01},
    ...     {'class': RandomForestRegressor, 'n_estimators': 100, 'max_depth': 5}
    ... ]
    >>> windows = [4, 16, 64, 256]
    >>> 
    >>> selector = ATOMSSelector(specs, windows)
    >>> best_model = selector.select(train_data, val_data, t=100)
    """
    
    def __init__(
        self,
        model_specs: List[Dict[str, Any]],
        window_sizes: List[int],
        delta_prime: float = 0.1,
        M: float = 1.0
    ):
        self.model_specs = model_specs
        self.window_sizes = sorted(window_sizes)
        self.delta_prime = delta_prime
        self.M = M
    
    def _train_candidate(
        self,
        spec: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> BaseModel:
        """Train a single candidate model."""
        model_class = spec['class']
        params = {k: v for k, v in spec.items() if k != 'class'}
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        return ModelWrapper(model)
    
    def select(
        self,
        train_data: ValidationData,
        val_data: ValidationData,
        t: int,
        verbose: bool = False
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Select best model using ATOMS.
        
        Parameters
        ----------
        train_data : ValidationData
            Training data across periods
        val_data : ValidationData
            Validation data across periods
        t : int
            Current time period
        verbose : bool
            Print progress
        
        Returns
        -------
        best_model : BaseModel
            Selected model
        info : dict
            Selection metadata
        """
        candidates = []
        candidate_info = []
        
        # Train all combinations of model specs and window sizes
        for spec in self.model_specs:
            for k in self.window_sizes:
                if k > t:
                    continue  # Not enough history
                
                # Get training data for window
                X_train, y_train = train_data.get_window(t, k)
                
                if len(y_train) == 0:
                    continue
                
                try:
                    model = self._train_candidate(spec, X_train, y_train)
                    candidates.append(model)
                    candidate_info.append({
                        'spec': spec,
                        'window': k,
                        'n_train': len(y_train)
                    })
                except Exception as e:
                    if verbose:
                        print(f"Failed to train {spec} with window {k}: {e}")
                    continue
        
        if len(candidates) == 0:
            raise ValueError("No valid candidate models could be trained")
        
        if verbose:
            print(f"Trained {len(candidates)} candidate models")
        
        # Run ATOMS
        winner, winner_idx, atoms_info = atoms(
            candidates, val_data, t, self.delta_prime, self.M, verbose
        )
        
        return winner, {
            **atoms_info,
            'winner_spec': candidate_info[winner_idx],
            'n_candidates': len(candidates),
            'candidate_info': candidate_info
        }


# =============================================================================
# Fixed-window baselines for comparison
# =============================================================================

def fixed_validation_selection(
    candidates: List[BaseModel],
    val_data: ValidationData,
    t: int,
    ell: int
) -> Tuple[BaseModel, int]:
    """
    Algorithm 3: Fixed Validation Window Selection.
    
    Baseline method that uses a fixed validation window.
    
    Parameters
    ----------
    candidates : list of BaseModel
        Candidate models
    val_data : ValidationData
        Validation data
    t : int
        Current period
    ell : int
        Fixed validation window size
    
    Returns
    -------
    winner : BaseModel
        Model with lowest MSE on validation window
    winner_idx : int
        Index of winner
    """
    X_val, y_val = val_data.get_window(t, ell)
    
    if len(y_val) == 0:
        return candidates[0], 0
    
    best_mse = np.inf
    best_idx = 0
    
    for idx, model in enumerate(candidates):
        mse = model.mse(X_val, y_val)
        if mse < best_mse:
            best_mse = mse
            best_idx = idx
    
    return candidates[best_idx], best_idx


# =============================================================================
# R² metrics
# =============================================================================

def oos_r2(
    model: BaseModel,
    test_data: ValidationData,
    t_start: int,
    t_end: int,
    benchmark_zero: bool = True
) -> float:
    """
    Compute out-of-sample R² over evaluation period.
    
    Parameters
    ----------
    model : BaseModel
        Fitted model
    test_data : ValidationData
        Test data
    t_start, t_end : int
        Evaluation period [t_start, t_end]
    benchmark_zero : bool
        If True, benchmark against zero forecast (Eq 2.2)
        If False, benchmark against mean (standard R², Eq 2.3)
    
    Returns
    -------
    r2 : float
        Out-of-sample R²
    """
    sse = 0.0
    ss_total = 0.0
    all_y = []
    
    for t in range(t_start, t_end + 1):
        if t >= test_data.T:
            break
        X_t = test_data.X[t]
        y_t = test_data.y[t]
        
        if len(y_t) == 0:
            continue
        
        preds = model.predict(X_t)
        sse += np.sum((preds - y_t) ** 2)
        
        if benchmark_zero:
            ss_total += np.sum(y_t ** 2)
        else:
            all_y.extend(y_t)
    
    if not benchmark_zero:
        y_mean = np.mean(all_y) if len(all_y) > 0 else 0
        ss_total = sum(
            np.sum((test_data.y[t] - y_mean) ** 2)
            for t in range(t_start, min(t_end + 1, test_data.T))
            if len(test_data.y[t]) > 0
        )
    
    if ss_total == 0:
        return 0.0
    
    return 1 - sse / ss_total


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    
    # Generate synthetic non-stationary data
    T = 100
    n_per_period = 50
    d = 5
    
    X_list = []
    y_list = []
    
    for t in range(T):
        X_t = np.random.randn(n_per_period, d)
        # Time-varying coefficients (drift)
        beta_t = np.array([1.0, 0.5, -0.3, 0.2, -0.1]) + 0.01 * t * np.array([1, -1, 0.5, -0.5, 0])
        y_t = X_t @ beta_t + 0.5 * np.random.randn(n_per_period)
        X_list.append(X_t)
        y_list.append(y_t)
    
    # Split into train/val
    train_X = [X[:n_per_period//2] for X in X_list]
    train_y = [y[:n_per_period//2] for y in y_list]
    val_X = [X[n_per_period//2:] for X in X_list]
    val_y = [y[n_per_period//2:] for y in y_list]
    
    train_data = ValidationData(train_X, train_y)
    val_data = ValidationData(val_X, val_y)
    
    print("ATOMS Test Run")
    print("=" * 50)
    print(f"Periods: {T}, Samples per period: {n_per_period}, Features: {d}")
    
    # Test with sklearn models
    try:
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        
        specs = [
            {'class': Ridge, 'alpha': 1.0},
            {'class': Ridge, 'alpha': 0.01},
            {'class': Lasso, 'alpha': 0.01},
        ]
        windows = [4, 16, 64]
        
        selector = ATOMSSelector(specs, windows, M=10.0)
        
        t_test = 80
        winner, info = selector.select(train_data, val_data, t_test, verbose=True)
        
        print(f"\nSelected model: {info['winner_spec']}")
        print(f"Comparisons made: {info['n_comparisons']}")
        print(f"Total candidates: {info['n_candidates']}")
        
    except ImportError:
        print("sklearn not available, skipping full test")

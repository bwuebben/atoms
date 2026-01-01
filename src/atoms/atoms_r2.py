"""
ATOMS-R2: R²-Based Model Selection
===================================

Implementation of the R²-based variant from Appendix B of the paper.

This variant directly targets the R² metric rather than MSE,
which may be more appropriate for asset pricing applications.

Key differences from standard ATOMS:
- Comparison statistics normalized by variance (Eq B.2)
- Different variance proxy (Eq B.3)
- Guarantees stated in terms of R² (Theorem B.1)
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from atoms import (
    ValidationData,
    BaseModel,
    ModelWrapper
)


def compute_r2_comparison_stats(
    f1: BaseModel,
    f2: BaseModel,
    val_data: ValidationData,
    t: int,
    ell: int,
    M: float = 1.0,
    v_lower: float = 0.01
) -> Tuple[float, float, float, int]:
    """
    Compute R²-based comparison statistics.
    
    Parameters
    ----------
    f1, f2 : BaseModel
        Models to compare
    val_data : ValidationData
        Validation data
    t : int
        Current period
    ell : int
        Window size
    M : float
        Bound on |f(x)| and |y|
    v_lower : float
        Lower bound on E[y²] (Assumption B.1)
    
    Returns
    -------
    delta_R_hat : float
        R²-normalized performance gap estimate
    v_R_hat : float
        Normalized variance estimate
    V_t_ell : float
        Average second moment
    n_ell : int
        Sample count
    """
    X_val, y_val = val_data.get_window(t, ell)
    n_ell = len(y_val)
    
    if n_ell == 0:
        return 0.0, 8 * M**2 / v_lower, v_lower, 0
    
    # Compute V_{t,ell} = average of E[y²] over window
    # In practice, approximate with sample second moments
    V_t_ell = 0.0
    start = max(0, t - ell)
    total_n = 0
    
    for j in range(start, t):
        if j < len(val_data.y):
            y_j = val_data.y[j]
            n_j = len(y_j)
            V_j = np.mean(y_j ** 2) if n_j > 0 else v_lower
            V_t_ell += n_j * V_j
            total_n += n_j
    
    V_t_ell = V_t_ell / total_n if total_n > 0 else v_lower
    V_t_ell = max(V_t_ell, v_lower)  # Ensure lower bound
    
    # Compute u values
    pred1 = f1.predict(X_val)
    pred2 = f2.predict(X_val)
    u = (pred1 - y_val)**2 - (pred2 - y_val)**2
    
    # Delta estimates (Eq B.2)
    delta_hat = np.mean(u)
    delta_R_hat = delta_hat / V_t_ell
    
    # Variance estimate
    if n_ell == 1:
        v_R_hat = 8 * M**2 / v_lower
    else:
        v_hat = np.std(u, ddof=1)
        v_R_hat = v_hat / V_t_ell
    
    return delta_R_hat, v_R_hat, V_t_ell, n_ell


def compute_psi_R_hat(
    v_R_hat: float,
    n_ell: int,
    delta_prime: float,
    M: float,
    v_lower: float
) -> float:
    """
    Compute R²-based variance proxy (Eq B.3).
    """
    if n_ell <= 1:
        return 8 * M**2 / v_lower
    
    log_term = np.log(2 / delta_prime)
    term1 = v_R_hat * np.sqrt(2 * log_term / n_ell)
    term2 = 64 * (M**2 / v_lower) * log_term / (3 * (n_ell - 1))
    
    return term1 + term2


def compute_phi_R_hat(
    delta_R_hats: Dict[int, float],
    psi_R_hats: Dict[int, float],
    ell: int
) -> float:
    """
    Compute R²-based bias proxy (Eq B.5).
    """
    phi_R_hat = 0.0
    delta_R_ell = delta_R_hats[ell]
    psi_R_ell = psi_R_hats[ell]
    
    for i in range(1, ell + 1):
        if i in delta_R_hats:
            diff = abs(delta_R_ell - delta_R_hats[i])
            penalty = psi_R_ell + psi_R_hats[i]
            phi_R_hat = max(phi_R_hat, max(0, diff - penalty))
    
    return phi_R_hat


def adaptive_rolling_window_comparison_r2(
    f1: BaseModel,
    f2: BaseModel,
    val_data: ValidationData,
    t: int,
    delta_prime: float = 0.1,
    M: float = 1.0,
    v_lower: float = 0.01
) -> Tuple[BaseModel, int]:
    """
    Algorithm 4: Adaptive Rolling Window for Model Comparison (R² Metric).
    
    R²-based variant of Algorithm 2.
    
    Parameters
    ----------
    f1, f2 : BaseModel
        Candidate models
    val_data : ValidationData
        Validation data
    t : int
        Current period
    delta_prime : float
        Confidence parameter
    M : float
        Bound constant
    v_lower : float
        Lower bound on E[y²]
    
    Returns
    -------
    winner : BaseModel
        Selected model
    selected_window : int
        Adaptively chosen window size
    """
    max_window = t
    
    if max_window == 0:
        return f1, 0
    
    # Compute statistics for all windows
    delta_R_hats = {}
    psi_R_hats = {}
    
    for ell in range(1, max_window + 1):
        delta_R_hat, v_R_hat, V_t_ell, n_ell = compute_r2_comparison_stats(
            f1, f2, val_data, t, ell, M, v_lower
        )
        
        if n_ell == 0:
            continue
        
        delta_R_hats[ell] = delta_R_hat
        psi_R_hats[ell] = compute_psi_R_hat(v_R_hat, n_ell, delta_prime, M, v_lower)
    
    if len(delta_R_hats) == 0:
        return f1, 0
    
    # Find optimal window
    best_ell = None
    best_criterion = np.inf
    
    for ell in delta_R_hats.keys():
        phi_R_hat = compute_phi_R_hat(delta_R_hats, psi_R_hats, ell)
        criterion = phi_R_hat + psi_R_hats[ell]
        
        if criterion < best_criterion:
            best_criterion = criterion
            best_ell = ell
    
    # Select winner
    if delta_R_hats[best_ell] <= 0:
        return f1, best_ell
    else:
        return f2, best_ell


def atoms_r2(
    candidates: List[BaseModel],
    val_data: ValidationData,
    t: int,
    delta_prime: float = 0.1,
    M: float = 1.0,
    v_lower: float = 0.01,
    verbose: bool = False
) -> Tuple[BaseModel, int, Dict[str, Any]]:
    """
    ATOMS-R2: Adaptive Tournament Model Selection targeting R² metric.
    
    Parameters
    ----------
    candidates : list of BaseModel
        Candidate models
    val_data : ValidationData
        Validation data
    t : int
        Current period
    delta_prime : float
        Confidence parameter
    M : float
        Bound constant
    v_lower : float
        Lower bound on E[y²]
    verbose : bool
        Print progress
    
    Returns
    -------
    winner : BaseModel
        Selected model
    winner_idx : int
        Index in candidates
    info : dict
        Metadata
    """
    if len(candidates) == 0:
        raise ValueError("Must provide at least one candidate")
    
    if len(candidates) == 1:
        return candidates[0], 0, {"n_comparisons": 0}
    
    S = list(range(len(candidates)))
    n_comparisons = 0
    comparison_log = []
    
    while len(S) > 1:
        # Random pivot
        pivot_pos = np.random.randint(len(S))
        pivot_idx = S[pivot_pos]
        pivot_model = candidates[pivot_idx]
        
        if verbose:
            print(f"R² Round: {len(S)} models, pivot={pivot_idx}")
        
        S_prime = []
        
        for other_idx in S:
            if other_idx == pivot_idx:
                continue
            
            other_model = candidates[other_idx]
            winner, window = adaptive_rolling_window_comparison_r2(
                pivot_model, other_model, val_data, t, 
                delta_prime, M, v_lower
            )
            n_comparisons += 1
            
            comparison_log.append({
                "pivot": pivot_idx,
                "challenger": other_idx,
                "winner": pivot_idx if winner is pivot_model else other_idx,
                "window": window
            })
            
            if winner is other_model:
                S_prime.append(other_idx)
        
        if len(S_prime) == 0:
            return candidates[pivot_idx], pivot_idx, {
                "n_comparisons": n_comparisons,
                "comparison_log": comparison_log
            }
        else:
            S = S_prime
    
    winner_idx = S[0]
    return candidates[winner_idx], winner_idx, {
        "n_comparisons": n_comparisons,
        "comparison_log": comparison_log
    }


class ATOMSR2Selector:
    """
    High-level interface for ATOMS-R2 model selection.
    
    Similar to ATOMSSelector but uses R²-based comparison.
    
    Parameters
    ----------
    model_specs : list of dict
        Model specifications
    window_sizes : list of int
        Training window sizes
    delta_prime : float
        Confidence parameter
    M : float
        Bound constant
    v_lower : float
        Lower bound on E[y²]
    """
    
    def __init__(
        self,
        model_specs: List[Dict[str, Any]],
        window_sizes: List[int],
        delta_prime: float = 0.1,
        M: float = 1.0,
        v_lower: float = 0.01
    ):
        self.model_specs = model_specs
        self.window_sizes = sorted(window_sizes)
        self.delta_prime = delta_prime
        self.M = M
        self.v_lower = v_lower
    
    def select(
        self,
        train_data: ValidationData,
        val_data: ValidationData,
        t: int,
        verbose: bool = False
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Select best model using ATOMS-R2.
        """
        candidates = []
        candidate_info = []
        
        for spec in self.model_specs:
            for k in self.window_sizes:
                if k > t:
                    continue
                
                X_train, y_train = train_data.get_window(t, k)
                
                if len(y_train) < 5:
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
            raise ValueError("No valid candidates")
        
        winner, winner_idx, atoms_info = atoms_r2(
            candidates, val_data, t,
            self.delta_prime, self.M, self.v_lower, verbose
        )
        
        return winner, {
            **atoms_info,
            'winner_spec': candidate_info[winner_idx],
            'n_candidates': len(candidates)
        }


def compare_atoms_variants(
    train_data: ValidationData,
    val_data: ValidationData,
    test_periods: List[int],
    model_specs: List[Dict],
    window_sizes: List[int],
    M: float = 1.0
) -> Dict[str, List[float]]:
    """
    Compare standard ATOMS vs ATOMS-R2.
    
    Returns per-period R² for each method.
    """
    from atoms import atoms
    
    results = {
        'ATOMS': [],
        'ATOMS-R2': []
    }
    
    for t in test_periods:
        if t >= val_data.T:
            continue
        
        # Train candidates
        candidates = []
        
        for spec in model_specs:
            for k in window_sizes:
                if k > t:
                    continue
                
                X_train, y_train = train_data.get_window(t, k)
                if len(y_train) < 5:
                    continue
                
                try:
                    model_class = spec['class']
                    params = {key: val for key, val in spec.items() if key != 'class'}
                    model = model_class(**params)
                    model.fit(X_train, y_train)
                    candidates.append(ModelWrapper(model))
                except:
                    continue
        
        if len(candidates) == 0:
            continue
        
        # Standard ATOMS
        winner_std, _, _ = atoms(candidates, val_data, t, M=M)
        
        # ATOMS-R2
        winner_r2, _, _ = atoms_r2(candidates, val_data, t, M=M)
        
        # Evaluate
        if t < val_data.T:
            X_test = val_data.X[t]
            y_test = val_data.y[t]
            
            if len(y_test) > 0:
                ss_tot = np.sum(y_test ** 2)
                
                if ss_tot > 0:
                    pred_std = winner_std.predict(X_test)
                    r2_std = 1 - np.sum((pred_std - y_test)**2) / ss_tot
                    results['ATOMS'].append(r2_std)
                    
                    pred_r2 = winner_r2.predict(X_test)
                    r2_r2 = 1 - np.sum((pred_r2 - y_test)**2) / ss_tot
                    results['ATOMS-R2'].append(r2_r2)
    
    return results


if __name__ == "__main__":
    # Test ATOMS-R2
    np.random.seed(42)
    
    print("ATOMS-R2 Test")
    print("=" * 50)
    
    # Generate test data
    T = 100
    n_per_period = 30
    d = 5
    
    X_list = []
    y_list = []
    
    for t in range(T):
        X_t = np.random.randn(n_per_period, d)
        beta = np.array([1.0, 0.5, -0.3, 0.2, -0.1]) + 0.02 * t * np.array([1, -1, 0, 0, 0])
        y_t = X_t @ beta + 0.5 * np.random.randn(n_per_period)
        X_list.append(X_t)
        y_list.append(y_t)
    
    # Split
    train_X = [X[:n_per_period//2] for X in X_list]
    train_y = [y[:n_per_period//2] for y in y_list]
    val_X = [X[n_per_period//2:] for X in X_list]
    val_y = [y[n_per_period//2:] for y in y_list]
    
    train_data = ValidationData(train_X, train_y)
    val_data = ValidationData(val_X, val_y)
    
    try:
        from sklearn.linear_model import Ridge, Lasso
        
        specs = [
            {'class': Ridge, 'alpha': 1.0},
            {'class': Ridge, 'alpha': 0.1},
            {'class': Lasso, 'alpha': 0.01},
        ]
        windows = [4, 16, 64]
        
        print("\nComparing ATOMS vs ATOMS-R2...")
        
        results = compare_atoms_variants(
            train_data, val_data,
            test_periods=list(range(70, 100)),
            model_specs=specs,
            window_sizes=windows
        )
        
        print(f"\nATOMS average R²: {np.mean(results['ATOMS']):.4f}")
        print(f"ATOMS-R2 average R²: {np.mean(results['ATOMS-R2']):.4f}")
        
    except ImportError:
        print("sklearn not available")

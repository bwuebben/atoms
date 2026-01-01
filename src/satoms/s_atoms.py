"""
S-ATOMS: Soft, Similarity-based Adaptive Tournament Model Selection
====================================================================

Implementation of the S-ATOMS framework from:
"When History Rhymes: Ensemble Learning and Regime-Aware Estimation 
under Nonstationarity" - Wuebben (2025)

This module extends the ATOMS framework with three key innovations:
1. Soft Ensemble Weighting (Section 3.2)
2. Empirical Proxy Estimation via Block Bootstrap (Section 3.1)
3. Similarity-Based Data Selection (Section 3.3)

Key Classes:
- SATOMSSelector: Main interface for S-ATOMS model selection
- BlockBootstrapVariance: Circular block bootstrap for variance estimation
- IntegralDriftBias: Robust bias estimation via integral drift
- SimilarityDataSelector: Mahalanobis distance-based data selection
- MarketStateVector: Constructs regime-characterizing state vectors
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.spatial.distance import mahalanobis
from collections import defaultdict
import copy


# =============================================================================
# Import base classes from ATOMS
# =============================================================================

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
        if start >= t:
            return np.array([]).reshape(0, self.X[0].shape[1] if len(self.X) > 0 else 0), np.array([])
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
    
    def get_params(self) -> Optional[np.ndarray]:
        """Get model parameters (for drift computation). Override in subclasses."""
        return None


class ModelWrapper(BaseModel):
    """Wrapper to make sklearn-style models compatible with S-ATOMS."""
    
    def __init__(self, model: Any, name: str = ""):
        self.model = model
        self.name = name or type(model).__name__
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def get_params(self) -> Optional[np.ndarray]:
        """Extract parameters for drift computation."""
        if hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if hasattr(self.model, 'intercept_'):
                intercept = self.model.intercept_
                if np.isscalar(intercept):
                    return np.append(coef.flatten(), intercept)
                else:
                    return np.append(coef.flatten(), intercept.flatten())
            return coef.flatten()
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models - use feature importances as proxy
            return self.model.feature_importances_
        return None


# =============================================================================
# Section 3.1: Empirical Proxy Estimation
# =============================================================================

class BlockBootstrapVariance:
    """
    Block Bootstrap Variance Estimation (Section 3.1.1).
    
    Implements circular block bootstrap for variance estimation that
    preserves the autocorrelation and heteroskedasticity structure
    of financial returns.
    
    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap replications (B in paper, default 500)
    block_length : int or str
        Block length for bootstrap. If 'auto', uses Politis-White (2004)
        automatic selection.
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_bootstrap: int = 500,
        block_length: Union[int, str] = 'auto',
        random_state: Optional[int] = None
    ):
        self.n_bootstrap = n_bootstrap
        self.block_length = block_length
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    def _select_block_length(self, data: np.ndarray) -> int:
        """
        Automatic block length selection via Politis-White (2004).
        
        Uses the 'optimal' block length for variance estimation:
        b = 2 * (2 * m_hat)^(1/3) * n^(1/3)
        where m_hat is estimated from the autocorrelation structure.
        """
        n = len(data)
        if n < 10:
            return max(1, n // 2)
        
        # Estimate autocorrelation decay
        max_lag = min(n // 3, 100)
        autocorrs = np.array([
            np.corrcoef(data[:-lag], data[lag:])[0, 1] if lag > 0 else 1.0
            for lag in range(max_lag)
        ])
        
        # Find first insignificant lag (simple heuristic)
        threshold = 2 / np.sqrt(n)
        m_hat = np.argmax(np.abs(autocorrs) < threshold)
        if m_hat == 0:
            m_hat = max_lag // 2
        
        # Politis-White formula
        block_length = int(np.ceil(2 * (2 * m_hat) ** (1/3) * n ** (1/3)))
        
        return max(1, min(block_length, n // 2))
    
    def compute_variance_proxy(
        self,
        u: np.ndarray,
        sigma_local: float = 1.0
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute bootstrap variance proxy (Eq 15 in paper).
        
        Parameters
        ----------
        u : np.ndarray
            Loss differences u_{j,i} = (f1(x) - y)^2 - (f2(x) - y)^2
        sigma_local : float
            Local volatility for standardization (Eq 16)
        
        Returns
        -------
        psi_boot : float
            Bootstrap variance proxy
        psi_boot_std : float
            Standardized variance proxy (by local volatility)
        bootstrap_means : np.ndarray
            Bootstrap distribution of mean estimates
        """
        n = len(u)
        if n < 2:
            return 1.0, 1.0, np.array([np.mean(u)])
        
        # Select block length
        if self.block_length == 'auto':
            b = self._select_block_length(u)
        else:
            b = self.block_length
        b = max(1, min(b, n))
        
        # Generate bootstrap samples
        bootstrap_means = np.zeros(self.n_bootstrap)
        
        for r in range(self.n_bootstrap):
            # Circular block bootstrap
            boot_sample = self._circular_block_bootstrap(u, b)
            bootstrap_means[r] = np.mean(boot_sample)
        
        # Compute variance proxy (Eq 15)
        mean_of_means = np.mean(bootstrap_means)
        psi_boot = np.sqrt(np.var(bootstrap_means, ddof=1))
        
        # Standardize by local volatility (Eq 16)
        psi_boot_std = psi_boot / sigma_local if sigma_local > 0 else psi_boot
        
        return psi_boot, psi_boot_std, bootstrap_means
    
    def _circular_block_bootstrap(self, data: np.ndarray, block_length: int) -> np.ndarray:
        """
        Generate one circular block bootstrap sample.
        """
        n = len(data)
        n_blocks = int(np.ceil(n / block_length))
        
        boot_indices = []
        for _ in range(n_blocks):
            # Random starting point (circular)
            start = self._rng.randint(0, n)
            for j in range(block_length):
                boot_indices.append((start + j) % n)
        
        # Truncate to original length
        boot_indices = boot_indices[:n]
        return data[boot_indices]


class IntegralDriftBias:
    """
    Integral Drift Bias Estimation (Section 3.1.2).
    
    Replaces the max-deviation Goldenshluger-Lepski criterion with
    a robust integral drift metric that averages parameter divergence.
    
    Parameters
    ----------
    decay_half_life : int
        Half-life for exponential decay kernel (ℓ_half in paper, default 6)
    calibration_constant : float
        Multiplicative calibration factor (c_φ in paper, default 1.15)
    """
    
    def __init__(
        self,
        decay_half_life: int = 6,
        calibration_constant: float = 1.15
    ):
        self.decay_half_life = decay_half_life
        self.calibration_constant = calibration_constant
        self._kappa = np.log(2) / decay_half_life
    
    def compute_decay_weights(self, ell: int) -> np.ndarray:
        """
        Compute exponential decay weights (Eq 19).
        
        ω_s = exp(-κs) / Σ exp(-κs')
        """
        if ell <= 0:
            return np.array([1.0])
        
        s_values = np.arange(1, ell + 1)
        weights = np.exp(-self._kappa * s_values)
        weights = weights / np.sum(weights)
        return weights
    
    def compute_parameter_divergence(
        self,
        theta_ell: np.ndarray,
        theta_s: np.ndarray
    ) -> float:
        """
        Compute parameter divergence D(t, ℓ, s) = ||θ_{t,ℓ} - θ_{t,s}||²₂ (Eq 17).
        """
        if theta_ell is None or theta_s is None:
            return 0.0
        
        return np.sum((theta_ell - theta_s) ** 2)
    
    def compute_integral_drift(
        self,
        params_by_window: Dict[int, np.ndarray],
        ell: int,
        sigma_local: float = 1.0
    ) -> float:
        """
        Compute integral drift bias proxy (Eq 18).
        
        φ̂_int(t, ℓ) = (1/σ̂_t) · (1/ℓ) · Σ_{s=1}^ℓ ω_s · D(t, ℓ, s)
        
        Parameters
        ----------
        params_by_window : dict
            {window_size: parameter_vector} for fitted models
        ell : int
            Current window size
        sigma_local : float
            Local volatility for standardization
        
        Returns
        -------
        phi_int : float
            Integral drift bias proxy
        """
        if ell not in params_by_window or params_by_window[ell] is None:
            return 0.0
        
        theta_ell = params_by_window[ell]
        weights = self.compute_decay_weights(ell)
        
        integral_drift = 0.0
        valid_count = 0
        
        for s in range(1, ell + 1):
            if s in params_by_window and params_by_window[s] is not None:
                theta_s = params_by_window[s]
                divergence = self.compute_parameter_divergence(theta_ell, theta_s)
                if s - 1 < len(weights):
                    integral_drift += weights[s - 1] * divergence
                    valid_count += 1
        
        if valid_count == 0:
            return 0.0
        
        # Normalize and standardize
        phi_int = (integral_drift / ell) / sigma_local if sigma_local > 0 else integral_drift / ell
        
        # Apply calibration constant (Eq 20)
        return self.calibration_constant * phi_int
    
    def compute_bias_proxy(
        self,
        delta_hats: Dict[int, float],
        psi_hats: Dict[int, float],
        params_by_window: Dict[int, np.ndarray],
        ell: int,
        sigma_local: float = 1.0,
        use_integral: bool = True
    ) -> float:
        """
        Compute bias proxy - either integral drift or traditional max-deviation.
        
        Parameters
        ----------
        delta_hats : dict
            {window_size: delta_hat} estimates
        psi_hats : dict
            {window_size: psi_hat} variance proxies
        params_by_window : dict
            {window_size: parameters} for drift computation
        ell : int
            Current window size
        sigma_local : float
            Local volatility
        use_integral : bool
            If True, use integral drift; otherwise use max-deviation
        
        Returns
        -------
        phi_hat : float
            Bias proxy
        """
        if use_integral and len(params_by_window) > 0:
            return self.compute_integral_drift(params_by_window, ell, sigma_local)
        
        # Fallback to max-deviation (Goldenshluger-Lepski)
        phi_hat = 0.0
        if ell not in delta_hats:
            return phi_hat
        
        delta_ell = delta_hats[ell]
        psi_ell = psi_hats.get(ell, 0.0)
        
        for i in range(1, ell + 1):
            if i in delta_hats:
                diff = abs(delta_ell - delta_hats[i])
                penalty = psi_ell + psi_hats.get(i, 0.0)
                phi_hat = max(phi_hat, max(0, diff - penalty))
        
        return phi_hat


# =============================================================================
# Section 3.2: Soft Ensemble Weighting
# =============================================================================

class SoftEnsembleWeighter:
    """
    Soft Ensemble Weighting (Section 3.2).
    
    Replaces hard winner-take-all selection with exponentially-weighted
    averaging across all candidates.
    
    Parameters
    ----------
    gamma : float or str
        Sharpness parameter. If 'adaptive', selects via cross-validation.
    gamma_grid : list
        Grid of gamma values for adaptive selection
    calibration_window : int
        Window length for cross-validation (ℓ_cal in paper)
    """
    
    def __init__(
        self,
        gamma: Union[float, str] = 'adaptive',
        gamma_grid: List[float] = None,
        calibration_window: int = 24
    ):
        self.gamma = gamma
        self.gamma_grid = gamma_grid or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        self.calibration_window = calibration_window
    
    def compute_weights(
        self,
        risk_scores: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Compute softmax ensemble weights (Eq 23).
        
        W_λ = exp(-γ · R_λ) / Σ exp(-γ · R_λ')
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Risk scores R_λ for each candidate
        gamma : float
            Sharpness parameter
        
        Returns
        -------
        weights : np.ndarray
            Normalized ensemble weights
        """
        # Use log-sum-exp trick for numerical stability
        log_weights = -gamma * risk_scores
        log_weights = log_weights - np.max(log_weights)  # Shift for stability
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        return weights
    
    def select_gamma_adaptive(
        self,
        candidates: List[BaseModel],
        risk_scores: np.ndarray,
        val_data: ValidationData,
        t: int
    ) -> float:
        """
        Adaptive sharpness selection via cross-validation (Eq 26).
        
        γ̂_t = argmin_γ Σ_{s=t-ℓ_cal}^{t-1} (y_s - ŷ_s(γ))²
        """
        if self.gamma != 'adaptive' or t <= self.calibration_window:
            return self.gamma if isinstance(self.gamma, (int, float)) else 2.0
        
        best_gamma = 2.0
        best_mse = np.inf
        
        for gamma in self.gamma_grid:
            mse = self._evaluate_gamma(
                candidates, risk_scores, val_data, t, gamma
            )
            if mse < best_mse:
                best_mse = mse
                best_gamma = gamma
        
        return best_gamma
    
    def _evaluate_gamma(
        self,
        candidates: List[BaseModel],
        risk_scores: np.ndarray,
        val_data: ValidationData,
        t: int,
        gamma: float
    ) -> float:
        """Evaluate a gamma value on calibration window."""
        weights = self.compute_weights(risk_scores, gamma)
        
        total_se = 0.0
        n_samples = 0
        
        start = max(0, t - self.calibration_window)
        
        for s in range(start, t):
            if s >= val_data.T:
                continue
            
            X_s = val_data.X[s]
            y_s = val_data.y[s]
            
            if len(y_s) == 0:
                continue
            
            # Ensemble prediction
            pred = np.zeros(len(y_s))
            for w, model in zip(weights, candidates):
                pred += w * model.predict(X_s)
            
            total_se += np.sum((pred - y_s) ** 2)
            n_samples += len(y_s)
        
        return total_se / n_samples if n_samples > 0 else np.inf
    
    def compute_ensemble_prediction(
        self,
        candidates: List[BaseModel],
        weights: np.ndarray,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted ensemble prediction (Eq 24).
        
        ŷ_t = Σ_λ W_λ · f_λ(x_t)
        """
        pred = np.zeros(X.shape[0])
        for w, model in zip(weights, candidates):
            pred += w * model.predict(X)
        return pred
    
    def compute_turnover(
        self,
        weights_prev: np.ndarray,
        weights_curr: np.ndarray
    ) -> float:
        """
        Compute portfolio turnover (Eq 27).
        
        Turnover_t = (1/2) · Σ_λ |W_{λ,t+1} - W_{λ,t}|
        """
        if weights_prev is None or len(weights_prev) != len(weights_curr):
            return 0.0
        return 0.5 * np.sum(np.abs(weights_curr - weights_prev))
    
    @staticmethod
    def effective_n_models(weights: np.ndarray) -> float:
        """
        Compute effective number of models: Λ_eff = 1 / Σ W_λ²
        """
        return 1.0 / np.sum(weights ** 2)
    
    @staticmethod
    def variance_reduction_factor(weights: np.ndarray, rho_bar: float = 0.5) -> float:
        """
        Compute variance reduction factor from ensembling (Proposition 3).
        
        Factor = ρ̄ + (1 - ρ̄) · Σ W_λ²
        """
        return rho_bar + (1 - rho_bar) * np.sum(weights ** 2)


# =============================================================================
# Section 3.3: Similarity-Based Data Selection
# =============================================================================

@dataclass
class MarketState:
    """Container for market state at a given time."""
    volatility_measures: np.ndarray  # RV, VIX, VoV, CSD
    correlation_measures: np.ndarray  # AvgCorr, PC1 share, stock-bond corr
    macro_measures: np.ndarray  # Term spread, credit spread, TED, IP, unemployment
    market_conditions: np.ndarray  # 12m return, 1m return, detrended P/D
    
    def to_vector(self) -> np.ndarray:
        """Concatenate all measures into single state vector."""
        return np.concatenate([
            self.volatility_measures,
            self.correlation_measures,
            self.macro_measures,
            self.market_conditions
        ])


class MarketStateVector:
    """
    Market State Vector Construction (Section 3.3.1).
    
    Constructs the state vector S_t capturing volatility, correlation structure,
    and macroeconomic conditions for similarity-based data selection.
    
    Parameters
    ----------
    volatility_window : int
        Window for realized volatility calculation (default 21 days)
    correlation_window : int
        Window for correlation calculations (default 63 days)
    shrinkage_alpha : float
        Covariance matrix shrinkage parameter (default 0.1)
    """
    
    def __init__(
        self,
        volatility_window: int = 21,
        correlation_window: int = 63,
        shrinkage_alpha: float = 0.1
    ):
        self.volatility_window = volatility_window
        self.correlation_window = correlation_window
        self.shrinkage_alpha = shrinkage_alpha
        
        # Storage for computed states
        self.states: Dict[int, np.ndarray] = {}
        self.covariance_matrix: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
    
    def compute_state_from_data(
        self,
        returns: pd.DataFrame,  # Industry returns
        t: int,
        vix: Optional[pd.Series] = None,
        macro_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Compute market state vector at time t.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns with DatetimeIndex
        t : int
            Time index
        vix : pd.Series, optional
            VIX index values
        macro_data : pd.DataFrame, optional
            Macroeconomic variables
        
        Returns
        -------
        state : np.ndarray
            Market state vector S_t
        """
        state_components = []
        
        # 1. Volatility measures (4 variables)
        if t >= self.volatility_window:
            window_returns = returns.iloc[t - self.volatility_window:t]
            
            # Realized volatility (annualized)
            market_returns = window_returns.mean(axis=1)
            rv = np.std(market_returns) * np.sqrt(252)
            
            # VIX (if available)
            if vix is not None and t < len(vix):
                vix_val = vix.iloc[t]
            else:
                vix_val = rv * 100  # Proxy
            
            # Volatility of volatility
            if t >= 2 * self.volatility_window:
                daily_vols = window_returns.std(axis=1)
                vov = np.std(daily_vols)
            else:
                vov = 0.01
            
            # Cross-sectional dispersion
            csd = np.std(returns.iloc[t]) if t < len(returns) else 0.02
            
            state_components.extend([rv, vix_val / 100, vov, csd])
        else:
            state_components.extend([0.15, 0.15, 0.01, 0.02])  # Defaults
        
        # 2. Correlation measures (3 variables)
        if t >= self.correlation_window:
            window_returns = returns.iloc[t - self.correlation_window:t]
            
            # Average pairwise correlation
            corr_matrix = window_returns.corr()
            avg_corr = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix) ** 2 - len(corr_matrix))
            
            # PC1 share of variance
            try:
                cov_matrix = window_returns.cov()
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                pc1_share = np.max(eigenvalues) / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0.5
            except:
                pc1_share = 0.5
            
            # Stock-bond correlation (proxy with negative of market vol)
            stock_bond_corr = -0.1  # Default; would need bond data
            
            state_components.extend([avg_corr, pc1_share, stock_bond_corr])
        else:
            state_components.extend([0.5, 0.5, -0.1])
        
        # 3. Macroeconomic indicators (5 variables)
        if macro_data is not None and t < len(macro_data):
            macro_row = macro_data.iloc[t]
            term_spread = macro_row.get('term_spread', 0.02)
            credit_spread = macro_row.get('credit_spread', 0.01)
            ted_spread = macro_row.get('ted_spread', 0.005)
            ip_change = macro_row.get('ip_change', 0.0)
            unemp_change = macro_row.get('unemp_change', 0.0)
            state_components.extend([term_spread, credit_spread, ted_spread, ip_change, unemp_change])
        else:
            state_components.extend([0.02, 0.01, 0.005, 0.0, 0.0])
        
        # 4. Market conditions (3 variables)
        if t >= 252:  # Need 1 year of data
            ret_12m = returns.iloc[t - 252:t].mean(axis=1).sum()
        else:
            ret_12m = 0.0
        
        if t >= 21:
            ret_1m = returns.iloc[t - 21:t].mean(axis=1).sum()
        else:
            ret_1m = 0.0
        
        pd_ratio = 0.0  # Would need price/dividend data
        
        state_components.extend([ret_12m, ret_1m, pd_ratio])
        
        return np.array(state_components)
    
    def compute_state_simple(
        self,
        returns: np.ndarray,
        t: int,
        window: int = 21
    ) -> np.ndarray:
        """
        Simplified state computation from returns matrix only.
        
        Uses observable features that can be computed from returns:
        - Realized volatility
        - Cross-sectional dispersion
        - Average correlation (rolling)
        - Momentum (recent returns)
        """
        n_assets = returns.shape[1] if returns.ndim > 1 else 1
        state = []
        
        # Volatility
        if t >= window:
            window_data = returns[t - window:t]
            if returns.ndim > 1:
                vol = np.std(window_data.mean(axis=1)) * np.sqrt(252)
                csd = np.mean(np.std(window_data, axis=0))
            else:
                vol = np.std(window_data) * np.sqrt(252)
                csd = 0.0
        else:
            vol = 0.15
            csd = 0.02
        state.extend([vol, csd])
        
        # Correlation structure (simplified)
        if t >= window and returns.ndim > 1 and n_assets > 1:
            window_data = returns[t - window:t]
            corr_matrix = np.corrcoef(window_data.T)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.mean(corr_matrix[mask])
        else:
            avg_corr = 0.5
        state.append(avg_corr)
        
        # Momentum
        if t >= 252:
            mom_12m = np.mean(returns[t - 252:t])
        else:
            mom_12m = 0.0
        
        if t >= 21:
            mom_1m = np.mean(returns[t - 21:t])
        else:
            mom_1m = 0.0
        state.extend([mom_12m, mom_1m])
        
        return np.array(state)
    
    def fit_covariance(self, states: np.ndarray) -> None:
        """
        Fit covariance matrix for Mahalanobis distance with shrinkage (Eq 30).
        
        Σ̂_reg = (1 - α)Σ̂ + α · diag(Σ̂)
        """
        if len(states) < 2:
            self.covariance_matrix = np.eye(states.shape[1] if states.ndim > 1 else 1)
            return
        
        # Standardize
        self._mean = np.mean(states, axis=0)
        self._std = np.std(states, axis=0)
        self._std[self._std < 1e-8] = 1.0  # Avoid division by zero
        
        states_normalized = (states - self._mean) / self._std
        
        # Sample covariance
        cov = np.cov(states_normalized.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        
        # Shrinkage toward diagonal
        cov_reg = (1 - self.shrinkage_alpha) * cov + self.shrinkage_alpha * np.diag(np.diag(cov))
        
        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvalsh(cov_reg))
        if min_eig < 1e-8:
            cov_reg += (1e-8 - min_eig) * np.eye(cov_reg.shape[0])
        
        self.covariance_matrix = cov_reg
    
    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        """Standardize state using fitted mean/std."""
        if self._mean is None:
            return state
        return (state - self._mean) / self._std


class SimilarityDataSelector:
    """
    Similarity-Based Data Selection (Section 3.3).
    
    Selects training data based on structural similarity to current market state
    using Mahalanobis distance, enabling "wormhole" access to historical analogs.
    
    Parameters
    ----------
    target_sample_size : int
        Target number of similar observations (n_target in paper, default 500)
    recent_window : int
        Length of recent window to always include (ℓ_recent in paper, default 12)
    omega_sim : float or 'adaptive'
        Weight for similarity-based observations
    omega_recent : float or 'adaptive'
        Weight for recent observations
    kernel : str
        Kernel function for similarity weighting ('epanechnikov' or 'gaussian')
    kappa_recent : float
        Temporal decay parameter for recent window
    """
    
    def __init__(
        self,
        target_sample_size: int = 500,
        recent_window: int = 12,
        omega_sim: Union[float, str] = 'adaptive',
        omega_recent: Union[float, str] = 'adaptive',
        kernel: str = 'epanechnikov',
        kappa_recent: float = 0.05
    ):
        self.target_sample_size = target_sample_size
        self.recent_window = recent_window
        self.omega_sim = omega_sim
        self.omega_recent = omega_recent
        self.kernel = kernel
        self.kappa_recent = kappa_recent
        
        self.state_constructor = MarketStateVector()
    
    def compute_mahalanobis_distance(
        self,
        state_t: np.ndarray,
        state_j: np.ndarray,
        cov_inv: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance (Eq 29).
        
        d(S_t, S_j) = √((S_t - S_j)ᵀ Σ̂⁻¹ (S_t - S_j))
        """
        diff = state_t - state_j
        return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
    
    def kernel_weight(self, distance: float, threshold: float) -> float:
        """
        Compute kernel weight K(d/ε).
        """
        u = distance / threshold if threshold > 0 else 0
        
        if self.kernel == 'epanechnikov':
            # K(u) = (3/4)(1 - u²)₊
            if u <= 1:
                return 0.75 * (1 - u ** 2)
            return 0.0
        else:  # Gaussian
            return np.exp(-0.5 * u ** 2)
    
    def select_similar_periods(
        self,
        states: Dict[int, np.ndarray],
        current_state: np.ndarray,
        t: int
    ) -> Tuple[List[int], Dict[int, float], float]:
        """
        Select periods with similar market states (Eq 31-32).
        
        Parameters
        ----------
        states : dict
            {period: state_vector} for all historical periods
        current_state : np.ndarray
            State vector at time t
        t : int
            Current time period
        
        Returns
        -------
        similar_periods : list
            Indices of similar periods
        weights : dict
            {period: observation_weight}
        threshold : float
            Selected similarity threshold ε
        """
        if len(states) == 0:
            return [], {}, 0.0
        
        # Stack states and fit covariance
        periods = sorted([p for p in states.keys() if p < t])
        if len(periods) == 0:
            return [], {}, 0.0
        
        state_matrix = np.array([states[p] for p in periods])
        self.state_constructor.fit_covariance(state_matrix)
        
        # Standardize
        current_std = self.state_constructor.standardize_state(current_state)
        
        # Compute distances
        cov_inv = np.linalg.inv(self.state_constructor.covariance_matrix)
        distances = {}
        
        for p in periods:
            state_std = self.state_constructor.standardize_state(states[p])
            distances[p] = self.compute_mahalanobis_distance(current_std, state_std, cov_inv)
        
        # Sort by distance and find threshold
        sorted_periods = sorted(periods, key=lambda p: distances[p])
        
        # Select threshold to achieve target sample size
        n_target = min(self.target_sample_size, len(sorted_periods))
        if n_target > 0:
            threshold = distances[sorted_periods[n_target - 1]] * 1.01  # Slight margin
        else:
            threshold = 1.0
        
        # Select similar periods
        similar_periods = [p for p in periods if distances[p] <= threshold]
        
        # Compute weights
        weights = {}
        for p in similar_periods:
            weights[p] = self.kernel_weight(distances[p], threshold)
        
        return similar_periods, weights, threshold
    
    def construct_blended_dataset(
        self,
        train_data: ValidationData,
        states: Dict[int, np.ndarray],
        current_state: np.ndarray,
        t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct blended dataset combining similar and recent data (Eq 33-35).
        
        Returns
        -------
        X_blend : np.ndarray
            Blended features
        y_blend : np.ndarray
            Blended targets
        sample_weights : np.ndarray
            Observation weights
        """
        # Get similar periods
        similar_periods, sim_weights, threshold = self.select_similar_periods(
            states, current_state, t
        )
        
        # Get recent periods
        recent_start = max(0, t - self.recent_window)
        recent_periods = list(range(recent_start, t))
        
        # Combine (union)
        all_periods = list(set(similar_periods) | set(recent_periods))
        all_periods = [p for p in all_periods if p < train_data.T]
        
        if len(all_periods) == 0:
            # Fallback to most recent available data
            X, y = train_data.get_window(t, min(t, self.recent_window))
            return X, y, np.ones(len(y))
        
        # Compute adaptive mixing weights (Eq 35)
        n_sim = len(similar_periods)
        n_recent = len(recent_periods)
        
        if self.omega_sim == 'adaptive':
            omega_sim = n_sim / (n_sim + n_recent) if (n_sim + n_recent) > 0 else 0.5
            omega_recent = 1 - omega_sim
        else:
            omega_sim = self.omega_sim
            omega_recent = self.omega_recent
        
        # Build dataset with weights (Eq 34)
        X_list, y_list, w_list = [], [], []
        
        for p in all_periods:
            X_p = train_data.X[p]
            y_p = train_data.y[p]
            
            if len(y_p) == 0:
                continue
            
            in_sim = p in similar_periods
            in_recent = p in recent_periods
            
            # Compute weight for each observation
            for i in range(len(y_p)):
                weight = 0.0
                
                if in_sim and not in_recent:
                    weight = omega_sim * sim_weights.get(p, 0.0)
                elif in_recent and not in_sim:
                    decay = np.exp(-self.kappa_recent * (t - p))
                    weight = omega_recent * decay
                else:  # Both
                    sim_weight = omega_sim * sim_weights.get(p, 0.0)
                    decay = np.exp(-self.kappa_recent * (t - p))
                    recent_weight = omega_recent * decay
                    weight = sim_weight + recent_weight
                
                if weight > 0:
                    X_list.append(X_p[i])
                    y_list.append(y_p[i])
                    w_list.append(weight)
        
        if len(y_list) == 0:
            X, y = train_data.get_window(t, min(t, self.recent_window))
            return X, y, np.ones(len(y))
        
        X_blend = np.array(X_list)
        y_blend = np.array(y_list)
        sample_weights = np.array(w_list)
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        return X_blend, y_blend, sample_weights


# =============================================================================
# Section 3.4: Complete S-ATOMS Algorithm
# =============================================================================

@dataclass
class CandidateModel:
    """Container for a candidate model with metadata."""
    model: BaseModel
    spec: Dict[str, Any]
    training_window: int
    data_source: str  # 'contiguous', 'similarity', 'blended'
    n_train: int
    risk_score: float = np.inf
    params: Optional[np.ndarray] = None


class SATOMSSelector:
    """
    S-ATOMS: Soft, Similarity-based Adaptive Tournament Model Selection.
    
    Main interface implementing Algorithm 1 from the paper.
    
    Parameters
    ----------
    model_specs : list of dict
        Model specifications (class + hyperparameters)
    window_sizes : list of int
        Training window sizes to consider
    data_sources : list of str
        Data selection strategies ('contiguous', 'similarity', 'blended')
    gamma : float or 'adaptive'
        Ensemble sharpness parameter
    n_bootstrap : int
        Bootstrap replications for variance estimation
    decay_half_life : int
        Half-life for integral drift kernel
    target_sample_size : int
        Target size for similarity-based selection
    recent_window : int
        Recent window length for blended data
    delta_prime : float
        Confidence parameter
    M : float
        Bound on predictions/targets
    use_integral_drift : bool
        Use integral drift (True) or max-deviation (False) for bias
    """
    
    def __init__(
        self,
        model_specs: List[Dict[str, Any]],
        window_sizes: List[int] = None,
        data_sources: List[str] = None,
        gamma: Union[float, str] = 'adaptive',
        n_bootstrap: int = 500,
        decay_half_life: int = 6,
        calibration_constant: float = 1.15,
        target_sample_size: int = 500,
        recent_window: int = 12,
        delta_prime: float = 0.1,
        M: float = 1.0,
        use_integral_drift: bool = True,
        verbose: bool = False
    ):
        self.model_specs = model_specs
        self.window_sizes = window_sizes or [1, 4, 16, 64, 256]
        self.data_sources = data_sources or ['contiguous', 'similarity', 'blended']
        self.gamma = gamma
        self.delta_prime = delta_prime
        self.M = M
        self.use_integral_drift = use_integral_drift
        self.verbose = verbose
        
        # Initialize components
        self.bootstrap_variance = BlockBootstrapVariance(n_bootstrap=n_bootstrap)
        self.integral_drift = IntegralDriftBias(
            decay_half_life=decay_half_life,
            calibration_constant=calibration_constant
        )
        self.soft_weighter = SoftEnsembleWeighter(
            gamma=gamma,
            calibration_window=24
        )
        self.similarity_selector = SimilarityDataSelector(
            target_sample_size=target_sample_size,
            recent_window=recent_window
        )
        
        # State storage
        self.market_states: Dict[int, np.ndarray] = {}
        self.previous_weights: Optional[np.ndarray] = None
    
    def compute_market_state(
        self,
        train_data: ValidationData,
        t: int
    ) -> np.ndarray:
        """Compute market state vector for period t."""
        if t in self.market_states:
            return self.market_states[t]
        
        # Stack all returns up to t
        if t > 0:
            returns = np.vstack([train_data.y[j] for j in range(t) if len(train_data.y[j]) > 0])
            if returns.ndim == 1:
                returns = returns.reshape(-1, 1)
        else:
            returns = np.zeros((1, 1))
        
        state = self.similarity_selector.state_constructor.compute_state_simple(
            returns, len(returns) - 1
        )
        
        self.market_states[t] = state
        return state
    
    def _train_candidate(
        self,
        spec: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> BaseModel:
        """Train a single candidate model."""
        model_class = spec['class']
        params = {k: v for k, v in spec.items() if k != 'class'}
        
        model = model_class(**params)
        
        # Try to use sample weights if supported
        if sample_weights is not None and hasattr(model, 'fit'):
            try:
                # Check if fit accepts sample_weight
                import inspect
                sig = inspect.signature(model.fit)
                if 'sample_weight' in sig.parameters:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train)
            except:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        return ModelWrapper(model, name=f"{model_class.__name__}")
    
    def _get_training_data(
        self,
        train_data: ValidationData,
        t: int,
        window: int,
        data_source: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get training data based on data source strategy."""
        
        if data_source == 'contiguous':
            X, y = train_data.get_window(t, window)
            return X, y, None
        
        elif data_source == 'similarity':
            current_state = self.compute_market_state(train_data, t)
            
            # Compute states for all previous periods
            for j in range(t):
                if j not in self.market_states:
                    self.compute_market_state(train_data, j)
            
            similar_periods, weights, _ = self.similarity_selector.select_similar_periods(
                self.market_states, current_state, t
            )
            
            if len(similar_periods) == 0:
                X, y = train_data.get_window(t, window)
                return X, y, None
            
            X_list, y_list, w_list = [], [], []
            for p in similar_periods:
                if p < train_data.T:
                    X_list.append(train_data.X[p])
                    y_list.append(train_data.y[p])
                    w_list.extend([weights.get(p, 1.0)] * len(train_data.y[p]))
            
            if len(y_list) == 0:
                X, y = train_data.get_window(t, window)
                return X, y, None
            
            return np.vstack(X_list), np.concatenate(y_list), np.array(w_list)
        
        else:  # blended
            current_state = self.compute_market_state(train_data, t)
            
            for j in range(t):
                if j not in self.market_states:
                    self.compute_market_state(train_data, j)
            
            X, y, weights = self.similarity_selector.construct_blended_dataset(
                train_data, self.market_states, current_state, t
            )
            return X, y, weights
    
    def train_candidates(
        self,
        train_data: ValidationData,
        t: int
    ) -> List[CandidateModel]:
        """
        Train all candidate models (Algorithm 1, Phase 2).
        """
        candidates = []
        
        for spec in self.model_specs:
            for window in self.window_sizes:
                if window > t:
                    continue
                
                for data_source in self.data_sources:
                    try:
                        X_train, y_train, sample_weights = self._get_training_data(
                            train_data, t, window, data_source
                        )
                        
                        if len(y_train) < 5:  # Minimum samples
                            continue
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = self._train_candidate(
                                spec, X_train, y_train, sample_weights
                            )
                        
                        # Get parameters for drift computation
                        params = model.get_params()
                        
                        candidates.append(CandidateModel(
                            model=model,
                            spec=spec,
                            training_window=window,
                            data_source=data_source,
                            n_train=len(y_train),
                            params=params
                        ))
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Training failed: {spec}, window={window}, "
                                  f"source={data_source}: {e}")
                        continue
        
        return candidates
    
    def compute_risk_scores(
        self,
        candidates: List[CandidateModel],
        val_data: ValidationData,
        t: int
    ) -> np.ndarray:
        """
        Compute risk scores for all candidates (Algorithm 1, Phase 3).
        
        Risk score R_λ = φ̂_soft(f_λ) + ψ̂_boot(f_λ)
        """
        n_candidates = len(candidates)
        risk_scores = np.full(n_candidates, np.inf)
        
        # Compute local volatility for standardization
        X_recent, y_recent = val_data.get_window(t, min(t, 12))
        sigma_local = np.std(y_recent) if len(y_recent) > 1 else 1.0
        sigma_local = max(sigma_local, 1e-8)
        
        for idx, candidate in enumerate(candidates):
            try:
                # Compute for all windows up to current
                delta_hats = {}
                psi_hats = {}
                params_by_window = {}
                
                max_window = min(t, candidate.training_window * 2)
                
                for ell in range(1, max_window + 1):
                    X_val, y_val = val_data.get_window(t, ell)
                    n_ell = len(y_val)
                    
                    if n_ell == 0:
                        continue
                    
                    # Use historical mean as baseline for comparison
                    pred = candidate.model.predict(X_val)
                    baseline_pred = np.zeros_like(pred)  # Zero benchmark
                    
                    u = (pred - y_val) ** 2 - (baseline_pred - y_val) ** 2
                    
                    delta_hats[ell] = np.mean(u)
                    
                    # Bootstrap variance proxy
                    psi_boot, psi_std, _ = self.bootstrap_variance.compute_variance_proxy(
                        u, sigma_local
                    )
                    psi_hats[ell] = psi_std
                    
                    # Store parameters for drift computation
                    if candidate.params is not None:
                        params_by_window[ell] = candidate.params
                
                if len(delta_hats) == 0:
                    continue
                
                # Find optimal window
                best_ell = None
                best_criterion = np.inf
                
                for ell in delta_hats.keys():
                    # Compute bias proxy
                    phi_hat = self.integral_drift.compute_bias_proxy(
                        delta_hats, psi_hats, params_by_window, ell,
                        sigma_local, self.use_integral_drift
                    )
                    
                    criterion = phi_hat + psi_hats[ell]
                    
                    if criterion < best_criterion:
                        best_criterion = criterion
                        best_ell = ell
                
                risk_scores[idx] = best_criterion
                candidate.risk_score = best_criterion
                
            except Exception as e:
                if self.verbose:
                    print(f"Risk score computation failed for candidate {idx}: {e}")
                continue
        
        return risk_scores
    
    def select(
        self,
        train_data: ValidationData,
        val_data: ValidationData,
        t: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Main S-ATOMS selection (Algorithm 1).
        
        Parameters
        ----------
        train_data : ValidationData
            Training data
        val_data : ValidationData
            Validation data
        t : int
            Current time period
        
        Returns
        -------
        ensemble_weights : np.ndarray
            Soft ensemble weights for each candidate
        risk_scores : np.ndarray
            Risk scores for each candidate
        info : dict
            Selection metadata
        """
        if self.verbose:
            print(f"\nS-ATOMS: Period {t}")
        
        # Phase 1 & 2: Train candidates
        candidates = self.train_candidates(train_data, t)
        
        if len(candidates) == 0:
            raise ValueError("No valid candidates trained")
        
        if self.verbose:
            print(f"  Trained {len(candidates)} candidates")
        
        # Phase 3: Compute risk scores
        risk_scores = self.compute_risk_scores(candidates, val_data, t)
        
        # Handle infinite risk scores
        valid_mask = np.isfinite(risk_scores)
        if not np.any(valid_mask):
            # All failed - use uniform weights
            ensemble_weights = np.ones(len(candidates)) / len(candidates)
        else:
            # Phase 4: Compute ensemble weights
            gamma = self.soft_weighter.select_gamma_adaptive(
                [c.model for c in candidates],
                risk_scores,
                val_data,
                t
            )
            
            # Only consider valid candidates for weight computation
            risk_scores_valid = np.where(valid_mask, risk_scores, np.max(risk_scores[valid_mask]) + 1)
            ensemble_weights = self.soft_weighter.compute_weights(risk_scores_valid, gamma)
        
        # Compute turnover
        turnover = self.soft_weighter.compute_turnover(
            self.previous_weights, ensemble_weights
        )
        self.previous_weights = ensemble_weights.copy()
        
        # Prepare info dict
        info = {
            'candidates': candidates,
            'n_candidates': len(candidates),
            'gamma': gamma if 'gamma' in dir() else 2.0,
            'turnover': turnover,
            'effective_n_models': SoftEnsembleWeighter.effective_n_models(ensemble_weights),
            'winner_idx': np.argmin(risk_scores),
            'winner_spec': candidates[np.argmin(risk_scores)].spec,
            'winner_window': candidates[np.argmin(risk_scores)].training_window,
            'winner_source': candidates[np.argmin(risk_scores)].data_source
        }
        
        return ensemble_weights, risk_scores, info
    
    def predict(
        self,
        X: np.ndarray,
        candidates: List[CandidateModel],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Generate ensemble prediction.
        """
        return self.soft_weighter.compute_ensemble_prediction(
            [c.model for c in candidates],
            weights,
            X
        )


# =============================================================================
# High-Level Interface for Industry Portfolios
# =============================================================================

class IndustrySATOMS:
    """
    S-ATOMS for Industry Portfolio Returns.
    
    Mirrors the paper's empirical setup with S-ATOMS extensions.
    
    Parameters
    ----------
    delta_prime : float
        Confidence parameter
    M : float
        Bound on returns (typical monthly ~0.1)
    gamma : float or 'adaptive'
        Ensemble sharpness
    use_similarity : bool
        Enable similarity-based data selection
    use_soft_ensemble : bool
        Enable soft ensemble weighting
    use_empirical_proxies : bool
        Use bootstrap/integral drift vs theoretical proxies
    verbose : bool
        Print progress
    """
    
    def __init__(
        self,
        delta_prime: float = 0.1,
        M: float = 0.1,
        gamma: Union[float, str] = 'adaptive',
        use_similarity: bool = True,
        use_soft_ensemble: bool = True,
        use_empirical_proxies: bool = True,
        verbose: bool = False
    ):
        self.delta_prime = delta_prime
        self.M = M
        self.gamma = gamma
        self.use_similarity = use_similarity
        self.use_soft_ensemble = use_soft_ensemble
        self.use_empirical_proxies = use_empirical_proxies
        self.verbose = verbose
        
        self._setup_model_specs()
        
        # Window sizes: 4^k for k=0,...,5
        self.window_sizes = [1, 4, 16, 64, 256]
        
        # Data sources
        if use_similarity:
            self.data_sources = ['contiguous', 'similarity', 'blended']
        else:
            self.data_sources = ['contiguous']
    
    def _setup_model_specs(self):
        """Setup model specifications from the paper."""
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
        
        self.model_specs = []
        
        # Ridge
        for alpha in [1e-3, 10**(-1.5), 1, 10**1.5, 1e3]:
            self.model_specs.append({'class': Ridge, 'alpha': alpha})
        
        # LASSO
        for alpha in [1e-5, 10**(-3.5), 1e-2, 10**(-0.5), 10]:
            self.model_specs.append({'class': Lasso, 'alpha': alpha, 'max_iter': 10000})
        
        # Elastic Net
        for alpha in [1e-3, 1, 1e3]:
            for l1_ratio in [0.01, 0.05, 0.1]:
                self.model_specs.append({
                    'class': ElasticNet,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'max_iter': 10000
                })
        
        # Random Forest
        for n_estimators in [10, 100, 200]:
            for max_depth in [3, 5, 10]:
                self.model_specs.append({
                    'class': RandomForestRegressor,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': 42,
                    'n_jobs': -1
                })
    
    def create_selector(self) -> SATOMSSelector:
        """Create S-ATOMS selector with current settings."""
        return SATOMSSelector(
            model_specs=self.model_specs,
            window_sizes=self.window_sizes,
            data_sources=self.data_sources,
            gamma=self.gamma if self.use_soft_ensemble else 100.0,  # High gamma = hard selection
            n_bootstrap=500 if self.use_empirical_proxies else 100,
            use_integral_drift=self.use_empirical_proxies,
            delta_prime=self.delta_prime,
            M=self.M,
            verbose=self.verbose
        )
    
    def run_single_industry(
        self,
        train_data: ValidationData,
        val_data: ValidationData,
        test_periods: List[int],
        industry_name: str = ""
    ) -> Dict:
        """
        Run S-ATOMS for a single industry.
        
        Returns
        -------
        results : dict
            predictions, actuals, weights, turnovers, etc.
        """
        selector = self.create_selector()
        
        results = {
            'predictions': [],
            'actuals': [],
            'ensemble_weights': [],
            'risk_scores': [],
            'turnovers': [],
            'effective_n_models': [],
            'winner_specs': [],
            'winner_windows': [],
            'winner_sources': [],
            'n_candidates': []
        }
        
        for t in test_periods:
            if t >= val_data.T:
                continue
            
            try:
                # Run S-ATOMS selection
                weights, risk_scores, info = selector.select(
                    train_data, val_data, t
                )
                
                # Get test data
                X_t = val_data.X[t]
                y_t = val_data.y[t]
                
                if len(y_t) == 0:
                    continue
                
                # Generate prediction
                if self.use_soft_ensemble:
                    pred = selector.predict(X_t, info['candidates'], weights)
                else:
                    # Hard selection - use winner only
                    winner = info['candidates'][info['winner_idx']]
                    pred = winner.model.predict(X_t)
                
                # Store results
                results['predictions'].extend(pred)
                results['actuals'].extend(y_t)
                results['ensemble_weights'].append(weights)
                results['risk_scores'].append(risk_scores)
                results['turnovers'].append(info['turnover'])
                results['effective_n_models'].append(info['effective_n_models'])
                results['winner_specs'].append(info['winner_spec'])
                results['winner_windows'].append(info['winner_window'])
                results['winner_sources'].append(info['winner_source'])
                results['n_candidates'].append(info['n_candidates'])
                
                if self.verbose and t % 10 == 0:
                    print(f"  Period {t}: n_cand={info['n_candidates']}, "
                          f"eff_n={info['effective_n_models']:.2f}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  Period {t} failed: {e}")
                continue
        
        # Compute summary metrics
        if len(results['predictions']) > 0:
            results['oos_r2'] = self._compute_oos_r2(
                results['predictions'], results['actuals']
            )
            results['avg_turnover'] = np.mean(results['turnovers'])
            results['avg_effective_n'] = np.mean(results['effective_n_models'])
        
        return results
    
    def _compute_oos_r2(
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


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_atoms_vs_satoms(
    train_data: ValidationData,
    val_data: ValidationData,
    test_periods: List[int],
    model_specs: List[Dict],
    window_sizes: List[int],
    M: float = 0.1,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare ATOMS vs S-ATOMS (and ablated variants).
    
    Returns results for:
    - ATOMS: Original algorithm
    - S-ATOMS: Full S-ATOMS
    - S-ATOMS (no similarity): Without similarity-based selection
    - S-ATOMS (no ensemble): Hard selection only
    - S-ATOMS (no empirical): Theoretical proxies
    """
    results = {}
    
    # 1. Standard ATOMS (approximate with hard selection, no similarity)
    if verbose:
        print("\n1. Running ATOMS baseline...")
    atoms_selector = IndustrySATOMS(
        M=M,
        gamma=100.0,  # Hard selection
        use_similarity=False,
        use_soft_ensemble=False,
        use_empirical_proxies=False,
        verbose=False
    )
    results['ATOMS'] = atoms_selector.run_single_industry(
        train_data, val_data, test_periods
    )
    if verbose:
        print(f"   R² = {results['ATOMS'].get('oos_r2', 0):.4f}")
    
    # 2. S-ATOMS (full)
    if verbose:
        print("\n2. Running S-ATOMS (full)...")
    satoms_selector = IndustrySATOMS(
        M=M,
        gamma='adaptive',
        use_similarity=True,
        use_soft_ensemble=True,
        use_empirical_proxies=True,
        verbose=False
    )
    results['S-ATOMS'] = satoms_selector.run_single_industry(
        train_data, val_data, test_periods
    )
    if verbose:
        print(f"   R² = {results['S-ATOMS'].get('oos_r2', 0):.4f}")
    
    # 3. S-ATOMS without similarity
    if verbose:
        print("\n3. Running S-ATOMS (no similarity)...")
    satoms_no_sim = IndustrySATOMS(
        M=M,
        gamma='adaptive',
        use_similarity=False,
        use_soft_ensemble=True,
        use_empirical_proxies=True,
        verbose=False
    )
    results['S-ATOMS (no sim)'] = satoms_no_sim.run_single_industry(
        train_data, val_data, test_periods
    )
    if verbose:
        print(f"   R² = {results['S-ATOMS (no sim)'].get('oos_r2', 0):.4f}")
    
    # 4. S-ATOMS without ensemble
    if verbose:
        print("\n4. Running S-ATOMS (no ensemble)...")
    satoms_no_ens = IndustrySATOMS(
        M=M,
        gamma=100.0,
        use_similarity=True,
        use_soft_ensemble=False,
        use_empirical_proxies=True,
        verbose=False
    )
    results['S-ATOMS (no ens)'] = satoms_no_ens.run_single_industry(
        train_data, val_data, test_periods
    )
    if verbose:
        print(f"   R² = {results['S-ATOMS (no ens)'].get('oos_r2', 0):.4f}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        for method, res in results.items():
            r2 = res.get('oos_r2', 0)
            turnover = res.get('avg_turnover', 0)
            eff_n = res.get('avg_effective_n', 1)
            print(f"{method:20s}: R²={r2:.4f}, Turnover={turnover:.3f}, Eff_N={eff_n:.2f}")
    
    return results


# =============================================================================
# Test and Demonstration
# =============================================================================

if __name__ == "__main__":
    print("S-ATOMS: Soft, Similarity-based Adaptive Tournament Model Selection")
    print("=" * 70)
    
    # Check dependencies
    try:
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        exit(1)
    
    np.random.seed(42)
    
    # Generate synthetic regime-switching data
    print("\n1. Generating synthetic regime-switching data...")
    
    T = 150
    n_per_period = 30
    d = 10
    
    # Regime changes
    regime_changes = [50, 100]
    regime_betas = {
        0: np.array([1.0, 0.5, -0.3, 0.2, -0.1] + [0.0] * (d - 5)),
        1: np.array([-0.5, 1.0, 0.8, -0.4, 0.3] + [0.0] * (d - 5)),
        2: np.array([0.8, 0.3, -0.2, 0.4, -0.2] + [0.0] * (d - 5)),
    }
    
    X_list, y_list = [], []
    current_regime = 0
    change_idx = 0
    
    for t in range(T):
        if change_idx < len(regime_changes) and t >= regime_changes[change_idx]:
            current_regime += 1
            change_idx += 1
        
        X_t = np.random.randn(n_per_period, d)
        y_t = X_t @ regime_betas[current_regime] + 0.5 * np.random.randn(n_per_period)
        X_list.append(X_t)
        y_list.append(y_t)
    
    # Split train/val
    train_X = [X[:n_per_period//2] for X in X_list]
    train_y = [y[:n_per_period//2] for y in y_list]
    val_X = [X[n_per_period//2:] for X in X_list]
    val_y = [y[n_per_period//2:] for y in y_list]
    
    train_data = ValidationData(train_X, train_y)
    val_data = ValidationData(val_X, val_y)
    
    print(f"   Periods: {T}")
    print(f"   Regime changes at: {regime_changes}")
    print(f"   Features: {d}")
    
    # Define model specs
    model_specs = [
        {'class': Ridge, 'alpha': 1.0},
        {'class': Ridge, 'alpha': 0.1},
        {'class': Lasso, 'alpha': 0.01},
    ]
    window_sizes = [4, 16, 64]
    test_periods = list(range(60, T))
    
    print(f"\n2. Running comparison...")
    print(f"   Test periods: {test_periods[0]} to {test_periods[-1]}")
    print(f"   Model specs: {len(model_specs)}")
    print(f"   Window sizes: {window_sizes}")
    
    # Run comparison
    results = compare_atoms_vs_satoms(
        train_data, val_data,
        test_periods,
        model_specs,
        window_sizes,
        M=5.0,
        verbose=True
    )
    
    # Compute improvement
    atoms_r2 = results['ATOMS'].get('oos_r2', 0)
    satoms_r2 = results['S-ATOMS'].get('oos_r2', 0)
    
    if atoms_r2 > 0:
        improvement = (satoms_r2 - atoms_r2) / atoms_r2 * 100
        print(f"\nS-ATOMS improvement over ATOMS: {improvement:.1f}%")
    
    print("\nDone!")

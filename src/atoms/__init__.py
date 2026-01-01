"""
ATOMS: Adaptive Tournament Model Selection

Original implementation from:
"The nonstationarity-complexity tradeoff in return prediction"
Capponi, Huang, Sidaoui, Wang, Zou (2025)

This module provides adaptive model selection for non-stationary environments
by jointly optimizing model complexity and training window size.
"""

from .atoms import (
    ValidationData,
    BaseModel,
    ModelWrapper,
    atoms,
    adaptive_rolling_window_comparison,
    fixed_validation_selection,
    ATOMSSelector
)

from .atoms_r2 import (
    atoms_r2,
    adaptive_rolling_window_comparison_r2,
    ATOMSR2Selector
)

__version__ = '1.0.0'

__all__ = [
    # Core data structures
    'ValidationData',
    'BaseModel',
    'ModelWrapper',

    # ATOMS algorithms
    'atoms',
    'adaptive_rolling_window_comparison',
    'fixed_validation_selection',
    'ATOMSSelector',

    # R²-based variant
    'atoms_r2',
    'adaptive_rolling_window_comparison_r2',
    'ATOMSR2Selector',
]

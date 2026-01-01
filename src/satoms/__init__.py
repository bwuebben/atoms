"""
S-ATOMS: Soft, Similarity-based Adaptive Tournament Model Selection

Enhanced implementation from:
"When History Rhymes: Ensemble Learning and Regime-Aware Estimation
under Nonstationarity" - Wuebben (2025)

S-ATOMS extends ATOMS with three key innovations:
1. Soft Ensemble Weighting (Section 3.2)
2. Empirical Proxy Estimation via Block Bootstrap (Section 3.1)
3. Similarity-Based Data Selection (Section 3.3)
"""

from .s_atoms import (
    # Base data structures
    ValidationData,
    BaseModel,
    ModelWrapper,

    # Section 3.1: Empirical Proxy Estimation
    BlockBootstrapVariance,
    IntegralDriftBias,

    # Section 3.2: Soft Ensemble Weighting
    SoftEnsembleWeighter,

    # Section 3.3: Similarity-Based Data Selection
    MarketState,
    MarketStateVector,
    SimilarityDataSelector,

    # Main algorithm
    CandidateModel,
    SATOMSSelector,
    IndustrySATOMS,

    # Utilities
    compare_atoms_vs_satoms,
)

__version__ = '1.0.0'

__all__ = [
    # Core data structures
    'ValidationData',
    'BaseModel',
    'ModelWrapper',
    'CandidateModel',

    # Empirical proxy estimation (Section 3.1)
    'BlockBootstrapVariance',
    'IntegralDriftBias',

    # Soft ensemble weighting (Section 3.2)
    'SoftEnsembleWeighter',

    # Similarity-based selection (Section 3.3)
    'MarketState',
    'MarketStateVector',
    'SimilarityDataSelector',

    # Main interfaces
    'SATOMSSelector',
    'IndustrySATOMS',

    # Utilities
    'compare_atoms_vs_satoms',
]

"""
LPPL — Log-Periodic Power Law for Market Bubble Detection.

Usage:
    from lppl import analyze
    analyze(prices, date_index, name="TSLA")
"""

__version__ = "3.0.0"
__author__ = "Yeonghun"

from .model import LPPLModel, LPPLParams
from .ensemble import analyze, run_ensemble, EnsembleResult, WindowFit

__all__ = [
    "LPPLModel",
    "LPPLParams",
    "analyze",
    "run_ensemble",
    "EnsembleResult",
    "WindowFit",
]

"""
Core LPPL (Log-Periodic Power Law) Model Implementation.

This module provides the main LPPL model class for detecting financial bubbles
using hybrid optimization combining global and local search algorithms.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any, List
import warnings


# Default constraint thresholds
DEFAULT_CB_RATIO_MAX = 1.0  # |C/B| must be < this value


@dataclass
class LPPLParams:
    """
    LPPL model parameters.

    The LPPL formula:
        ln(p(t)) = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) + φ)

    Attributes:
        tc: Critical time (predicted crash date)
        m: Power law exponent (typically 0.1 - 0.9)
        omega: Log-periodic frequency (typically 6 - 13)
        A: Log price at critical time
        B: Power law amplitude (negative for bubbles)
        C: Oscillation amplitude
        phi: Phase
    """
    tc: float
    m: float
    omega: float
    A: float
    B: float
    C: float
    phi: float

    def is_valid(self, cb_ratio_max: float = DEFAULT_CB_RATIO_MAX) -> bool:
        """
        Check if parameters satisfy bubble conditions.

        Valid bubble parameters require:
        - m in [0.1, 0.9]: Ensures finite-time singularity
        - omega in [6, 13]: Typical log-periodic frequency range
        - B < 0: Price increasing faster than exponential (bubble)
        - |C/B| < cb_ratio_max: Oscillations not too large vs growth

        Args:
            cb_ratio_max: Maximum allowed |C/B| ratio (default 1.0)
                         If oscillations (C) are too large relative to
                         growth (B), model is fitting volatility not bubble.
        """
        # Basic constraints
        if not (0.1 <= self.m <= 0.9):
            return False
        if not (6.0 <= self.omega <= 13.0):
            return False
        if not (self.B < 0):
            return False

        # C/B ratio constraint
        # Prevents fitting volatility instead of bubble
        if abs(self.B) > 1e-10:
            cb_ratio = abs(self.C / self.B)
            if cb_ratio > cb_ratio_max:
                return False

        return True

    def is_anti_bubble(self, cb_ratio_max: float = DEFAULT_CB_RATIO_MAX) -> bool:
        """
        Check if parameters satisfy anti-bubble conditions.

        Anti-bubble = prices accelerating downward toward tc (the bottom).
        B > 0: price term declines toward tc, signalling a recovery point.
        """
        if not (0.1 <= self.m <= 0.9):
            return False
        if not (6.0 <= self.omega <= 13.0):
            return False
        if not (self.B > 0):
            return False
        if abs(self.B) > 1e-10:
            if abs(self.C / self.B) > cb_ratio_max:
                return False
        return True

    def get_cb_ratio(self) -> float:
        """Get the |C/B| ratio."""
        if abs(self.B) < 1e-10:
            return float('inf')
        return abs(self.C / self.B)

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """Convert parameters to numpy array."""
        return np.array([self.tc, self.m, self.omega, self.A, self.B, self.C, self.phi])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'LPPLParams':
        """Create LPPLParams from numpy array."""
        return cls(*arr)


@dataclass
class MultiSeedResult:
    """Result from multi-seed fitting."""
    best_params: LPPLParams
    best_ssr: float
    tc_mean: float
    tc_std: float
    tc_values: List[float]
    is_stable: bool  # True if tc converged across seeds
    n_seeds: int
    n_valid: int
    convergence_ratio: float  # Fraction of seeds that converged


class LPPLModel:
    """
    Log-Periodic Power Law model for market bubble detection.

    Features:
    - Hybrid optimization: Differential Evolution (global) + L-BFGS-B (local)
    - Multi-seed validation for robust tc estimation
    - C/B ratio constraint to avoid fitting volatility
    - Numerical stability controls
    - Parameter bounds based on market conditions

    Example:
        >>> prices = np.array([100, 105, 112, 120, ...])
        >>> model = LPPLModel(prices)
        >>> params, ssr = model.fit()
        >>> if params.is_valid():
        ...     print(f"Bubble detected! Critical time: {params.tc}")

        # Multi-seed validation (recommended for production)
        >>> result = model.fit_multi_seed(n_seeds=10)
        >>> if result.is_stable:
        ...     print(f"High confidence tc: {result.tc_mean:.1f} +/- {result.tc_std:.1f}")
    """

    EPSILON = 1e-10  # Numerical stability constant

    def __init__(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None
    ):
        """
        Initialize LPPL model.

        Args:
            prices: Array of price data (must be positive)
            dates: Optional time indices (defaults to 0, 1, 2, ...)

        Raises:
            ValueError: If prices contain non-positive values
        """
        self.prices = np.asarray(prices, dtype=np.float64)

        if np.any(self.prices <= 0):
            raise ValueError("Prices must be positive")

        self.log_prices = np.log(self.prices)
        self.t = dates if dates is not None else np.arange(len(prices), dtype=np.float64)
        self.n = len(prices)

        # Cache for fitted results
        self._fitted_params: Optional[LPPLParams] = None
        self._fitted_ssr: Optional[float] = None

    def lppl_function(self, t: np.ndarray, params: LPPLParams) -> np.ndarray:
        """
        Compute LPPL log-price values.

        Args:
            t: Time points
            params: LPPL parameters

        Returns:
            Predicted log prices
        """
        dt = params.tc - t
        dt = np.maximum(dt, self.EPSILON)  # Numerical stability

        dt_m = np.power(dt, params.m)
        oscillation = np.cos(params.omega * np.log(dt) + params.phi)

        return params.A + params.B * dt_m + params.C * dt_m * oscillation

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function: Sum of Squared Residuals (SSR).

        Args:
            x: Parameter array [tc, m, omega, A, B, C, phi]

        Returns:
            SSR value (large penalty if invalid)
        """
        params = LPPLParams.from_array(x)

        # Penalize if tc is not in the future
        if params.tc <= self.t[-1]:
            return 1e10

        try:
            predicted = self.lppl_function(self.t, params)
            residuals = self.log_prices - predicted
            return np.sum(residuals ** 2)
        except (RuntimeWarning, FloatingPointError):
            return 1e10

    def _get_bounds(self, tc_window: Optional[Tuple[float, float]] = None, anti_bubble: bool = False, full_range: bool = False) -> list:
        """
        Get parameter bounds based on market conditions.

        Args:
            tc_window: Optional custom (min, max) for critical time

        Returns:
            List of (min, max) bounds for each parameter
        """
        t_range = self.t[-1] - self.t[0]
        price_range = self.log_prices.max() - self.log_prices.min()

        if tc_window is None:
            tc_min = self.t[-1] + 1
            tc_max = self.t[-1] + t_range
        else:
            tc_min, tc_max = tc_window

        return [
            (tc_min, tc_max),                          # tc
            (0.1, 0.9),                                # m
            (6.0, 13.0),                               # omega
            (self.log_prices.min() - price_range,
             self.log_prices.max() + price_range),     # A
            (-price_range * 2, price_range * 2) if full_range else (0.001, price_range * 2) if anti_bubble else (-price_range * 2, -0.001),  # B
            (-price_range, price_range),               # C
            (0.0, 2 * np.pi),                          # phi
        ]

    def fit(
        self,
        max_iterations: int = 1000,
        polish: bool = True,
        seed: Optional[int] = None,
        tc_window: Optional[Tuple[float, float]] = None,
        population_size: int = 15,
        tol: float = 1e-6,
        anti_bubble: bool = False,
        full_range: bool = False,
    ) -> Tuple[LPPLParams, float]:
        """
        Fit LPPL model using hybrid optimization.

        Strategy:
        1. Global search with Differential Evolution
        2. Local refinement with L-BFGS-B (if polish=True)

        Args:
            max_iterations: Maximum iterations for global search
            polish: Apply local optimization after global search
            seed: Random seed for reproducibility
            tc_window: Optional (min, max) bounds for critical time
            population_size: DE population size multiplier
            tol: Convergence tolerance

        Returns:
            Tuple of (fitted parameters, final SSR)
        """
        bounds = self._get_bounds(tc_window, anti_bubble=anti_bubble, full_range=full_range)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Global optimization with Differential Evolution
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                maxiter=max_iterations,
                seed=seed,
                polish=False,
                workers=1,
                updating='deferred',
                popsize=population_size,
                tol=tol,
                mutation=(0.5, 1.0),
                recombination=0.7
            )

        # Local refinement with L-BFGS-B
        if polish:
            local_result = minimize(
                self._objective,
                x0=result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if local_result.fun < result.fun:
                result = local_result

        self._fitted_params = LPPLParams.from_array(result.x)
        self._fitted_ssr = result.fun

        return self._fitted_params, self._fitted_ssr

    def fit_fast(
        self,
        max_iterations: int = 150,
        seed: Optional[int] = None,
    ) -> Tuple[LPPLParams, float]:
        """
        Fast fit using OLS linearization (Filimonov & Sornette, 2013).

        For fixed (tc, m, ω), the parameters A, B, C, φ are solved analytically
        via ordinary least squares — reducing the search space from 7D to 3D.

        This is typically 4–6× faster than the full 7D fit with comparable accuracy.

        The sign of B is unconstrained, so a single call detects both
        bubble (B < 0) and anti-bubble (B > 0) patterns.
        """
        t_range = self.t[-1] - self.t[0]

        bounds_3d = [
            (self.t[-1] + 1, self.t[-1] + t_range),  # tc
            (0.1, 0.9),                                # m
            (6.0, 13.0),                               # omega
        ]

        def objective(x):
            tc, m, omega = x
            if tc <= self.t[-1]:
                return 1e10
            dt = np.maximum(tc - self.t, self.EPSILON)
            dt_m = np.power(dt, m)
            log_dt = np.log(dt)
            X = np.column_stack([
                np.ones(self.n),
                dt_m,
                dt_m * np.cos(omega * log_dt),
                dt_m * np.sin(omega * log_dt),
            ])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, self.log_prices, rcond=None)
                return float(np.sum((self.log_prices - X @ coeffs) ** 2))
            except Exception:
                return 1e10

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds_3d,
                maxiter=max_iterations,
                seed=seed,
                polish=True,
                workers=1,
                updating='deferred',
                popsize=10,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
            )

        tc, m, omega = result.x

        # Recover A, B, C, φ analytically from the optimal (tc, m, ω)
        dt = np.maximum(tc - self.t, self.EPSILON)
        dt_m = np.power(dt, m)
        log_dt = np.log(dt)
        X = np.column_stack([
            np.ones(self.n),
            dt_m,
            dt_m * np.cos(omega * log_dt),
            dt_m * np.sin(omega * log_dt),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(X, self.log_prices, rcond=None)
        A, B, C1, C2 = coeffs.tolist()

        C   = float(np.sqrt(C1 ** 2 + C2 ** 2))
        phi = float(np.arctan2(-C2, C1))
        if phi < 0:
            phi += 2 * np.pi

        params = LPPLParams(
            tc=float(tc), m=float(m), omega=float(omega),
            A=float(A), B=float(B), C=C, phi=phi,
        )
        self._fitted_params = params
        self._fitted_ssr = float(result.fun)
        return params, self._fitted_ssr

    def fit_multi_seed(
        self,
        n_seeds: int = 10,
        base_seed: int = 42,
        max_iterations: int = 500,
        tc_tolerance: float = 10.0,
        cb_ratio_max: float = DEFAULT_CB_RATIO_MAX,
        verbose: bool = False
    ) -> MultiSeedResult:
        """
        Fit LPPL with multiple seeds for robust tc estimation.

        Production best practice: Run DE with different seeds and check
        if tc converges. If tc jumps around, the signal is noise.

        Args:
            n_seeds: Number of different seeds to try (default 10)
            base_seed: Starting seed value
            max_iterations: Max iterations per fit
            tc_tolerance: Max allowed tc std for "stable" classification (days)
            cb_ratio_max: Max |C/B| ratio for valid parameters
            verbose: Print progress

        Returns:
            MultiSeedResult with convergence analysis
        """
        results = []
        tc_values = []

        if verbose:
            print(f"Multi-seed fitting with {n_seeds} seeds...")

        for i in range(n_seeds):
            seed = base_seed + i * 7  # Different seeds

            try:
                params, ssr = self.fit(
                    max_iterations=max_iterations,
                    seed=seed,
                    polish=True
                )

                # Check validity with C/B constraint
                if params.is_valid(cb_ratio_max=cb_ratio_max):
                    results.append((params, ssr))
                    tc_values.append(params.tc)

                    if verbose:
                        print(f"  Seed {i+1}: tc={params.tc:.1f}, SSR={ssr:.4f}, C/B={params.get_cb_ratio():.3f}")
                else:
                    if verbose:
                        print(f"  Seed {i+1}: Invalid params (C/B={params.get_cb_ratio():.3f})")

            except Exception as e:
                if verbose:
                    print(f"  Seed {i+1}: Failed - {e}")

        n_valid = len(results)

        if n_valid == 0:
            # No valid fits
            return MultiSeedResult(
                best_params=None,
                best_ssr=float('inf'),
                tc_mean=float('nan'),
                tc_std=float('inf'),
                tc_values=[],
                is_stable=False,
                n_seeds=n_seeds,
                n_valid=0,
                convergence_ratio=0.0
            )

        # Find best fit (lowest SSR)
        best_idx = np.argmin([r[1] for r in results])
        best_params, best_ssr = results[best_idx]

        # Compute tc statistics
        tc_values = np.array(tc_values)
        tc_mean = np.mean(tc_values)
        tc_std = np.std(tc_values)

        # Check stability: tc_std < tolerance
        is_stable = tc_std < tc_tolerance

        # Convergence ratio: how many are within 1 std of mean
        within_1std = np.sum(np.abs(tc_values - tc_mean) < tc_std)
        convergence_ratio = within_1std / n_valid if n_valid > 0 else 0.0

        if verbose:
            print(f"\nMulti-seed results:")
            print(f"  Valid fits: {n_valid}/{n_seeds}")
            print(f"  tc mean: {tc_mean:.1f}")
            print(f"  tc std: {tc_std:.1f}")
            print(f"  Stable: {is_stable}")
            print(f"  Convergence: {convergence_ratio:.1%}")

        return MultiSeedResult(
            best_params=best_params,
            best_ssr=best_ssr,
            tc_mean=tc_mean,
            tc_std=tc_std,
            tc_values=tc_values.tolist(),
            is_stable=is_stable,
            n_seeds=n_seeds,
            n_valid=n_valid,
            convergence_ratio=convergence_ratio
        )

    def predict(
        self,
        params: Optional[LPPLParams] = None,
        t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict prices for given time points.

        Args:
            params: LPPL parameters (uses fitted if None)
            t: Time points (uses training times if None)

        Returns:
            Predicted prices

        Raises:
            ValueError: If no parameters available
        """
        if params is None:
            if self._fitted_params is None:
                raise ValueError("Model not fitted. Call fit() first or provide params.")
            params = self._fitted_params

        if t is None:
            t = self.t

        log_pred = self.lppl_function(t, params)
        return np.exp(log_pred)

    def get_residuals(self, params: Optional[LPPLParams] = None) -> np.ndarray:
        """
        Compute residuals (observed - predicted log prices).

        Args:
            params: LPPL parameters (uses fitted if None)

        Returns:
            Array of residuals
        """
        if params is None:
            if self._fitted_params is None:
                raise ValueError("Model not fitted. Call fit() first or provide params.")
            params = self._fitted_params

        predicted = self.lppl_function(self.t, params)
        return self.log_prices - predicted

    @property
    def fitted_params(self) -> Optional[LPPLParams]:
        """Get fitted parameters."""
        return self._fitted_params

    @property
    def fitted_ssr(self) -> Optional[float]:
        """Get fitted SSR."""
        return self._fitted_ssr


def fit_lppl(
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[LPPLParams, float, LPPLModel]:
    """
    Convenience function to fit LPPL model.

    Args:
        prices: Price data
        dates: Optional time indices
        **kwargs: Additional arguments passed to LPPLModel.fit()

    Returns:
        Tuple of (parameters, SSR, model)
    """
    model = LPPLModel(prices, dates)
    params, ssr = model.fit(**kwargs)
    return params, ssr, model


def fit_lppl_multi_seed(
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    n_seeds: int = 10,
    **kwargs
) -> Tuple[MultiSeedResult, LPPLModel]:
    """
    Convenience function for multi-seed LPPL fitting.

    Args:
        prices: Price data
        dates: Optional time indices
        n_seeds: Number of seeds
        **kwargs: Additional arguments passed to fit_multi_seed()

    Returns:
        Tuple of (MultiSeedResult, model)
    """
    model = LPPLModel(prices, dates)
    result = model.fit_multi_seed(n_seeds=n_seeds, **kwargs)
    return result, model

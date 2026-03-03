"""
Multi-Window Ensemble LPPL Analysis with Bayesian MCMC.

Scientific approach (Sornette et al.):
  Vary window start, fix end = today.
  Credible only when tc estimates converge across windows.

Bayesian MCMC (optional, requires emcee):
  Samples posterior P(tc | data) on top windows via OLS-marginalized likelihood.
  Gives calibrated monthly probability estimates for the peak date.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from .model import LPPLModel, LPPLParams


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WindowFit:
    """LPPL fit result for one window."""
    window_size: int
    t_start: int        # Start index in full price array
    tc: float           # Critical time in ABSOLUTE index (from data start)
    r_squared: float
    is_valid: bool
    params: LPPLParams  # Fitted params (tc is in LOCAL window time 0…window_size)


@dataclass
class EnsembleResult:
    """Result from multi-window ensemble."""
    n_windows: int
    n_valid: int
    bubble_fraction: float
    tc_values: np.ndarray       # Valid tc estimates (absolute indices)
    tc_median: Optional[float]
    tc_std: float
    tc_p10: Optional[float]     # 10th percentile — lower bound of 80% range
    tc_p90: Optional[float]     # 90th percentile — upper bound of 80% range
    signal: str                 # 'strong' | 'moderate' | 'weak' | 'none'
    best_fit: Optional[WindowFit]
    valid_fits: List[WindowFit] = field(default_factory=list)  # Raw fits for MCMC


# ── Core fitting ──────────────────────────────────────────────────────────────

def _fit_window(window_prices: np.ndarray, t_start: int, seed: int = 42) -> Optional[WindowFit]:
    """
    Fit LPPL on a single window using OLS-accelerated 3D optimisation.

    B is unconstrained — single pass detects bubble (B<0) and anti-bubble (B>0).
    """
    n = len(window_prices)
    t = np.arange(n, dtype=np.float64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LPPLModel(window_prices, t)
            params, ssr = model.fit_fast(max_iterations=150, seed=seed)

        ss_tot = np.sum((model.log_prices - model.log_prices.mean()) ** 2)
        r2 = float(np.clip(1.0 - ssr / (ss_tot + 1e-10), 0.0, 1.0))

        is_valid = params.is_valid() or params.is_anti_bubble()

        return WindowFit(
            window_size=n,
            t_start=t_start,
            tc=params.tc + t_start,
            r_squared=r2,
            is_valid=is_valid,
            params=params,
        )
    except Exception:
        return None


def _fit_window_worker(args):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    window_prices, t_start, seed = args
    return _fit_window(window_prices, t_start, seed)


def _build_result(fits: List[WindowFit], n_total: int) -> EnsembleResult:
    """Build an EnsembleResult from a list of valid fits."""
    n_valid = len(fits)
    bubble_fraction = n_valid / n_total if n_total > 0 else 0.0

    if n_valid < 5:
        return EnsembleResult(
            n_windows=n_total, n_valid=n_valid,
            bubble_fraction=bubble_fraction,
            tc_values=np.array([]),
            tc_median=None, tc_std=float("inf"),
            tc_p10=None, tc_p90=None,
            signal="none", best_fit=None,
            valid_fits=fits,
        )

    tc_vals   = np.array([f.tc for f in fits])
    tc_median = float(np.median(tc_vals))
    tc_std    = float(np.std(tc_vals))
    tc_p10    = float(np.percentile(tc_vals, 10))
    tc_p90    = float(np.percentile(tc_vals, 90))

    cv = tc_std / (abs(tc_median) + 1e-10)
    if bubble_fraction > 0.5 and cv < 0.20 and n_valid >= 20:
        signal = "strong"
    elif bubble_fraction > 0.3 and cv < 0.35 and n_valid >= 10:
        signal = "moderate"
    elif bubble_fraction > 0.15 and n_valid >= 5:
        signal = "weak"
    else:
        signal = "none"

    near_median = [f for f in fits if abs(f.tc - tc_median) < tc_std]
    best_fit = max(near_median or fits, key=lambda f: f.r_squared)

    return EnsembleResult(
        n_windows=n_total, n_valid=n_valid,
        bubble_fraction=bubble_fraction,
        tc_values=tc_vals,
        tc_median=tc_median, tc_std=tc_std,
        tc_p10=tc_p10, tc_p90=tc_p90,
        signal=signal, best_fit=best_fit,
        valid_fits=fits,
    )


def run_ensemble(
    prices: np.ndarray,
    min_window: int = 125,
    max_window: int = 750,
    step: int = 5,
    seed: int = 42,
) -> tuple:
    """
    Single-pass multi-window ensemble.

    Fits LPPL with unconstrained B sign, then splits by sign:
      - B < 0  →  bubble fits  (accelerating rise toward tc)
      - B > 0  →  anti-bubble fits  (accelerating decline toward tc)

    Returns (bubble_result, anti_result) as EnsembleResult.
    Each result has .valid_fits for downstream MCMC.
    """
    n = len(prices)
    max_window = min(max_window, n)
    window_sizes = list(range(min_window, max_window + 1, step))
    n_total = len(window_sizes)

    n_workers = min(os.cpu_count() or 4, 8)
    print(f"Fitting {n_total} windows ({min_window}–{max_window} days) on {n_workers} cores…")

    args_list = [(prices[n - ws : n].copy(), n - ws, seed) for ws in window_sizes]

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            all_fits = [f for f in executor.map(_fit_window_worker, args_list) if f is not None]
    except Exception:
        all_fits = [f for f in (_fit_window(*a) for a in args_list) if f is not None]

    bubble_fits = [f for f in all_fits if f.is_valid and f.params.B < 0]
    anti_fits   = [f for f in all_fits if f.is_valid and f.params.B > 0]

    return _build_result(bubble_fits, n_total), _build_result(anti_fits, n_total)


# ── Bayesian MCMC ─────────────────────────────────────────────────────────────

def _run_mcmc(
    prices: np.ndarray,
    window_fits: List[WindowFit],
    n_top: int = 5,
    n_walkers: int = 16,
    n_steps: int = 400,
) -> Optional[np.ndarray]:
    """
    Bayesian MCMC on top-N window fits using emcee.

    Samples (tc, m, ω) with A, B, C, φ analytically marginalized via OLS.
    This is the "profile likelihood" approach — fast and exact for linear params.

    Returns absolute-index tc posterior samples, or None if emcee is not installed.
    """
    try:
        import emcee
    except ImportError:
        return None

    if not window_fits:
        return None

    top_fits = sorted(window_fits, key=lambda f: f.r_squared, reverse=True)[:n_top]
    EPSILON  = 1e-10
    all_tc_samples = []

    for i, fit in enumerate(top_fits):
        n_pts   = fit.window_size
        t       = np.arange(n_pts, dtype=np.float64)
        log_p   = np.log(prices[fit.t_start : fit.t_start + n_pts])
        t_range = t[-1] - t[0]
        tc_lo   = t[-1] + 1.0
        tc_hi   = t[-1] + t_range

        # OLS-marginalized profile log-likelihood
        # Default args capture loop variables by value (avoids late-binding)
        def _log_prob(
            x,
            t=t, log_p=log_p, n_pts=n_pts,
            tc_lo=tc_lo, tc_hi=tc_hi,
        ):
            tc, m, omega = x
            if not (tc_lo <= tc <= tc_hi):
                return -np.inf
            if not (0.1 <= m <= 0.9):
                return -np.inf
            if not (6.0 <= omega <= 13.0):
                return -np.inf

            dt    = np.maximum(tc - t, EPSILON)
            dt_m  = np.power(dt, m)
            log_dt = np.log(dt)
            X = np.column_stack([
                np.ones(n_pts),
                dt_m,
                dt_m * np.cos(omega * log_dt),
                dt_m * np.sin(omega * log_dt),
            ])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, log_p, rcond=None)
                resid  = log_p - X @ coeffs
                sigma2 = float(np.var(resid))
                if sigma2 <= 0:
                    return -np.inf
                return -0.5 * n_pts * np.log(sigma2)  # profile log-likelihood
            except Exception:
                return -np.inf

        # Initialise walkers near the best-fit point
        rng   = np.random.default_rng(42 + i)
        p0_tc = float(np.clip(fit.params.tc, tc_lo + 0.1, tc_hi - 0.1))
        p0    = np.column_stack([
            rng.normal(p0_tc,             max(1.0, t_range * 0.01), n_walkers),
            rng.normal(fit.params.m,      0.02, n_walkers),
            rng.normal(fit.params.omega,  0.30, n_walkers),
        ])
        p0[:, 0] = np.clip(p0[:, 0], tc_lo + 0.1, tc_hi - 0.1)
        p0[:, 1] = np.clip(p0[:, 1], 0.11, 0.89)
        p0[:, 2] = np.clip(p0[:, 2], 6.1,  12.9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = emcee.EnsembleSampler(n_walkers, 3, _log_prob)
            sampler.run_mcmc(p0, n_steps, progress=False)

        burn_in = n_steps // 2
        flat    = sampler.get_chain(discard=burn_in, flat=True)
        tc_abs  = flat[:, 0] + fit.t_start   # local → absolute index
        all_tc_samples.append(tc_abs)

    return np.concatenate(all_tc_samples) if all_tc_samples else None


# ── Plain-language helpers ────────────────────────────────────────────────────

def _pattern_confidence(bubble_fraction: float) -> str:
    pct = f"{bubble_fraction:.0%}"
    if bubble_fraction >= 0.8:
        return f"Very high  — {pct} of all timeframes detect the pattern"
    elif bubble_fraction >= 0.5:
        return f"High  — {pct} of all timeframes detect the pattern"
    elif bubble_fraction >= 0.3:
        return f"Moderate  — {pct} of all timeframes detect the pattern"
    else:
        return f"Low  — only {pct} of timeframes detect the pattern"


def _timing_confidence(tc_std: float) -> str:
    months = tc_std / 21
    if months < 2:
        return f"High  — estimates cluster within ±{months:.1f} months"
    elif months < 5:
        return f"Medium  — estimates spread ±{months:.1f} months either way"
    else:
        return f"Low  — estimates are widely spread (±{months:.1f} months)"


def _bottom_line(signal: str, tc_date, days_out: int) -> str:
    month = tc_date.strftime("%b %Y")
    if signal == "strong":
        if days_out <= 30:
            return f"The bubble pattern is mature and the peak may be imminent (around {month})."
        return f"Strong bubble pattern detected. Watch for a peak around {month}."
    elif signal == "moderate":
        return f"A possible bubble pattern exists, but the timing is uncertain. Monitor monthly."
    elif signal == "weak":
        return f"Weak bubble signal. Not strong enough to draw conclusions."
    return "No consistent bubble pattern found in the price data."


# ── Date utilities ────────────────────────────────────────────────────────────

def _day_to_date(day_idx: float, date_index) -> pd.Timestamp:
    """Convert absolute day index to a calendar date."""
    n = len(date_index)
    k = int(round(day_idx))
    k = max(0, k)

    if k < n:
        return pd.Timestamp(date_index[k])

    avg_days = (date_index[-1] - date_index[0]).days / max(n - 1, 1)
    delta = pd.Timedelta(days=int((k - (n - 1)) * avg_days))
    return pd.Timestamp(date_index[-1]) + delta


# ── Visualization ─────────────────────────────────────────────────────────────

def _plot(
    prices: np.ndarray,
    date_index,
    result: EnsembleResult,
    name: str,
    anti_result: Optional[EnsembleResult] = None,
    mcmc_samples: Optional[np.ndarray] = None,
    future_prices: Optional[np.ndarray] = None,
    future_dates=None,
    save_path: Optional[str] = None,
):
    """
    Two-panel chart:
      Top    — price history with LPPL fit and state badge
      Bottom — Bayesian MCMC monthly probability bars (or histogram fallback)
    """
    n = len(prices)
    dates_pd = [pd.Timestamp(d) for d in date_index]

    # ── Determine market state ────────────────────────────────────────────────
    bubble_active = (
        result.signal in ("strong", "moderate")
        and result.tc_median is not None
        and result.tc_median > n
    )
    anti_active = (
        anti_result is not None
        and anti_result.signal in ("strong", "moderate")
        and anti_result.tc_median is not None
        and anti_result.tc_median > n
    )

    if bubble_active and anti_active:
        if result.bubble_fraction >= anti_result.bubble_fraction:
            anti_active = False
        else:
            bubble_active = False

    if bubble_active:
        state_label = "BUBBLE DETECTED"
        main_color  = "#c62828"
        bg_color    = "#ffebee"
        fit_color   = "#d32f2f"
        active_res  = result
    elif anti_active:
        state_label = "ANTI-BUBBLE  (buy signal)"
        main_color  = "#2e7d32"
        bg_color    = "#e8f5e9"
        fit_color   = "#388e3c"
        active_res  = anti_result
    else:
        state_label = "NO CLEAR SIGNAL"
        main_color  = "#616161"
        bg_color    = "#f5f5f5"
        fit_color   = "#9e9e9e"
        active_res  = result if result.tc_median else None

    conf_line = (
        f"{active_res.signal.upper()} SIGNAL  •  {active_res.bubble_fraction:.0%} of windows agree"
        if active_res and active_res.signal != "none"
        else "Insufficient data for a confident signal"
    )

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d0d0d0")
        ax.tick_params(colors="#555555", labelsize=8)
        ax.yaxis.label.set_color("#555555")

    # ── Panel 1: Price + LPPL fit ─────────────────────────────────────────────

    # Subtle background for the active pattern window
    if active_res is not None and active_res.best_fit is not None:
        ax1.axvspan(
            dates_pd[active_res.best_fit.t_start], dates_pd[-1],
            alpha=0.04, color=main_color, zorder=1,
        )

    ax1.plot(dates_pd, prices, color="#1a6faf", linewidth=1.8, label="Price", zorder=3)

    # Future actual prices (historical-mode only) — plotted before the fit so fit stays on top
    has_future = future_prices is not None and len(future_prices) > 0
    if has_future:
        future_dates_pd = [pd.Timestamp(d) for d in future_dates]
        # Connect seamlessly from the last historical close
        ax1.plot(
            [dates_pd[-1]] + future_dates_pd,
            [prices[-1]]   + list(future_prices),
            color="#ef6c00", linewidth=1.6, linestyle="-",
            label="Actual price (after analysis date)", zorder=3, alpha=0.9,
        )

    # Best LPPL fit curve
    bf = result.best_fit
    if bf is not None:
        w_prices = prices[bf.t_start : bf.t_start + bf.window_size]
        # Extend fit further when we have future data so users can compare trajectories
        if has_future:
            extra = len(future_prices) + 60
        else:
            extra = min(90, max(0, int(bf.params.tc) - bf.window_size - 5))
        t_plot   = np.linspace(0, bf.window_size + extra - 1, 400)
        t_plot   = t_plot[t_plot < bf.params.tc - 2]
        try:
            mdl    = LPPLModel(w_prices, np.arange(bf.window_size, dtype=float))
            fitted = np.exp(mdl.lppl_function(t_plot, bf.params))
            fit_dates = [_day_to_date(bf.t_start + ti, date_index) for ti in t_plot]
            ax1.plot(
                fit_dates, fitted,
                color=fit_color, linewidth=2, linestyle="--", alpha=0.85, zorder=4,
                label=f"LPPL fit  (R²={bf.r_squared:.2f})",
            )
        except Exception:
            pass

    # Predicted peak: 80% confidence band + median line
    if result.tc_median is not None:
        tc_date  = _day_to_date(result.tc_median, date_index)
        p10_date = _day_to_date(result.tc_p10, date_index)
        p90_date = _day_to_date(result.tc_p90, date_index)
        ax1.axvspan(
            p10_date, p90_date,
            alpha=0.10, color="#c62828",
            label=f"Peak window  {p10_date.strftime('%b %Y')}–{p90_date.strftime('%b %Y')}",
        )
        ax1.axvline(
            tc_date, color="#c62828", linewidth=2, linestyle=":",
            zorder=5, label=f"Median peak  {tc_date.strftime('%b %Y')}",
        )

    # Anti-bubble recovery (subtle secondary marker)
    if anti_result is not None and anti_result.tc_median is not None and not anti_active:
        atc = _day_to_date(anti_result.tc_median, date_index)
        ax1.axvline(
            atc, color="#43a047", linewidth=1.2, linestyle=":",
            alpha=0.55, zorder=5, label=f"Est. recovery  {atc.strftime('%b %Y')}",
        )

    analysis_label = "Analysis date" if has_future else "Today"
    ax1.axvline(dates_pd[-1], color="#aaaaaa", linewidth=1.2,
                linestyle="-", alpha=0.7, zorder=2, label=analysis_label)

    # State badge
    ax1.text(
        0.99, 0.97, f"{state_label}\n{conf_line}",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=main_color,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=bg_color, edgecolor=main_color,
            linewidth=1.8, alpha=0.95,
        ),
        zorder=10,
    )

    ax1.set_title(f"{name} — LPPL Bubble Analysis",
                  fontsize=13, fontweight="bold", color="#222222", pad=10)
    ax1.set_ylabel("Price", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9,
               frameon=True, edgecolor="#e0e0e0")
    ax1.grid(True, alpha=0.18, color="#aaaaaa")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── Panel 2: MCMC probability bars or fallback histogram ─────────────────
    if mcmc_samples is not None and len(mcmc_samples) > 0:
        _plot_mcmc_bars(ax2, mcmc_samples, date_index, main_color,
                        future_prices=future_prices, future_dates=future_dates)
    else:
        _plot_tc_histogram(ax2, result, date_index)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Chart saved → {save_path}")

    plt.show()
    return fig


def _plot_mcmc_bars(ax, tc_samples, date_index, main_color,
                    future_prices=None, future_dates=None):
    """Monthly probability bar chart from MCMC posterior samples."""
    today = pd.Timestamp(date_index[-1])

    # Convert posterior tc samples to calendar months (strip tz for period conversion)
    tc_dates  = [_day_to_date(tc, date_index) for tc in tc_samples]
    dti       = pd.DatetimeIndex(tc_dates)
    tc_months = (dti.tz_convert(None) if dti.tz is not None else dti).to_period("M")
    counts    = tc_months.value_counts().sort_index()
    total     = len(tc_months)
    probs     = counts / total * 100   # percent per month

    if len(probs) == 0:
        ax.text(0.5, 0.5, "Insufficient MCMC samples",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888")
        return

    months    = [p.to_timestamp() for p in probs.index]
    prob_vals = probs.values
    cum_vals  = np.cumsum(prob_vals)

    # Bars coloured by cumulative probability (blue → yellow → red)
    cmap       = plt.get_cmap("RdYlBu_r")
    bar_colors = [cmap(c / 100.0) for c in cum_vals]

    ax.bar(months, prob_vals, width=20, color=bar_colors,
           edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)

    # Cumulative % line on secondary y-axis
    ax_twin = ax.twinx()
    ax_twin.plot(months, cum_vals, color="#333333", linewidth=2,
                 linestyle="-", marker="o", markersize=3, zorder=4)
    ax_twin.set_ylim(0, 115)
    ax_twin.set_ylabel("Cumulative probability (%)", fontsize=9, color="#444444")
    ax_twin.tick_params(colors="#555555", labelsize=8)
    ax_twin.spines["top"].set_visible(False)
    ax_twin.spines["left"].set_visible(False)
    ax_twin.spines["bottom"].set_visible(False)
    ax_twin.spines["right"].set_color("#d0d0d0")

    # 50% and 90% threshold reference lines
    for threshold, label_prefix in [(50, "50% by"), (90, "90% by")]:
        idx = int(np.searchsorted(cum_vals, threshold))
        if 0 < idx < len(months):
            m_cross = months[idx]
            ax_twin.axhline(threshold, color="#cccccc", linewidth=1,
                            linestyle="--", alpha=0.9)
            ax_twin.text(
                months[-1], threshold + 2,
                f"{label_prefix} {m_cross.strftime('%b %Y')}",
                fontsize=7.5, color="#666666", ha="right", va="bottom",
            )

    # Analysis-date / Today marker
    today_label = "Analysis date" if (future_prices is not None and len(future_prices) > 0) else "Today"
    ax.axvline(today, color="#888888", linewidth=1.2,
               linestyle="-", alpha=0.65, zorder=5)
    if len(prob_vals) > 0:
        ax.text(today, max(prob_vals) * 0.95, f"  {today_label}",
                fontsize=7.5, color="#888888", va="top")

    # Actual high marker (historical mode) — shows where prices actually peaked
    if future_prices is not None and len(future_prices) > 0:
        future_dates_pd = [pd.Timestamp(d) for d in future_dates]
        peak_idx   = int(np.argmax(future_prices))
        peak_date  = future_dates_pd[peak_idx]
        still_rising = (peak_idx == len(future_prices) - 1)
        peak_label = (f"Highest so far  {peak_date.strftime('%b %Y')}"
                      if still_rising else
                      f"Actual high  {peak_date.strftime('%b %Y')}")
        ax.axvline(peak_date.to_pydatetime(), color="#ef6c00", linewidth=1.8,
                   linestyle="--", zorder=6)
        ax.text(peak_date.to_pydatetime(), max(prob_vals) * 1.15, f"  {peak_label}",
                fontsize=7.5, color="#ef6c00", va="top", fontweight="bold")

    # Info badge
    ax.text(
        0.01, 0.97,
        f"Bayesian MCMC  •  {len(tc_samples):,} posterior samples",
        transform=ax.transAxes, fontsize=8, color=main_color,
        va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=main_color, alpha=0.85, linewidth=1.2),
    )

    ax.set_ylim(0, max(prob_vals) * 1.35)
    ax.set_ylabel("Probability per month (%)", fontsize=9)
    ax.set_title(
        "When is the peak most likely?  (Bayesian posterior)",
        fontsize=9.5, color="#444444", pad=6,
    )
    ax.grid(True, alpha=0.18, color="#aaaaaa", axis="y")

    n_months = len(months)
    interval = 1 if n_months <= 18 else 2 if n_months <= 36 else 3
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d0d0d0")


def _plot_tc_histogram(ax, result: EnsembleResult, date_index):
    """Fallback: ensemble tc histogram when emcee is not installed."""
    if len(result.tc_values) == 0:
        ax.text(0.5, 0.5, "No bubble pattern detected",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888888")
        ax.set_title("Peak date distribution", fontsize=9.5, color="#888888", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d0d0d0")
        return

    tc_dt  = [_day_to_date(tc, date_index).to_pydatetime() for tc in result.tc_values]
    n_bins = min(40, max(10, len(tc_dt) // 3))
    ax.hist(tc_dt, bins=n_bins, color="#4a90d9",
            edgecolor="white", linewidth=0.5, alpha=0.8)

    if result.tc_median is not None:
        tc_date  = _day_to_date(result.tc_median, date_index)
        p10_date = _day_to_date(result.tc_p10, date_index)
        p90_date = _day_to_date(result.tc_p90, date_index)
        ax.axvline(tc_date.to_pydatetime(), color="crimson",
                   linewidth=2, linestyle="--",
                   label=f"Median: {tc_date.strftime('%b %Y')}")
        ax.axvspan(p10_date.to_pydatetime(), p90_date.to_pydatetime(),
                   alpha=0.12, color="crimson",
                   label=f"80% range: {p10_date.strftime('%b %Y')} – {p90_date.strftime('%b %Y')}")
        ax.legend(fontsize=8.5)

    ax.set_ylabel("Windows agreeing", fontsize=9)
    ax.set_title(
        "Peak date distribution  (install emcee for Bayesian probability bars)",
        fontsize=9, color="#888888", pad=6,
    )
    ax.grid(True, alpha=0.18, color="#aaaaaa")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d0d0d0")


# ── Main entry point ──────────────────────────────────────────────────────────

def _tick(level: str) -> str:
    """Visual reliability indicator: ✓ high / ◎ medium / ✗ low."""
    return {"high": "✓", "medium": "◎", "low": "✗"}.get(level, "?")


def _pattern_level(frac: float) -> str:
    if frac >= 0.50: return "high"
    if frac >= 0.30: return "medium"
    return "low"


def _timing_level(tc_std: float) -> str:
    months = tc_std / 21
    if months < 2: return "high"
    if months < 5: return "medium"
    return "low"


def _mcmc_level(p5: pd.Timestamp, p95: pd.Timestamp) -> str:
    months = max((p95 - p5).days / 30, 0)
    if months < 3: return "high"
    if months < 6: return "medium"
    return "low"


def _overall_summary(p_level: str, t_level: str, m_level: str) -> str:
    scores = {"high": 2, "medium": 1, "low": 0}
    total  = scores[p_level] + scores[t_level] + scores[m_level]
    header = (f"  Overall:  "
              f"Pattern {_tick(p_level)}  ·  "
              f"Timing {_tick(t_level)}  ·  "
              f"MCMC {_tick(m_level)}")
    if p_level == "low":
        verdict = "→ Pattern too weak — do not act on this signal"
    elif total >= 5:
        verdict = "→ HIGH CONVICTION — all three signals agree"
    elif total >= 4:
        verdict = "→ Good signal — act if other factors align"
    elif total >= 3:
        verdict = "→ Moderate — watch monthly, wait for timing to tighten"
    else:
        verdict = "→ Pattern real but timing uncertain — revisit monthly"
    return f"{header}\n     {verdict}"


def _print_guide():
    """Print a plain-English interpretation guide at the end of the report."""
    g = "═" * 54
    d = "─" * 54
    print(f"{g}")
    print(f"  HOW TO READ THIS REPORT")
    print(f"{g}")

    print(f"""
  PATTERN CONFIDENCE  (how many timeframes see the pattern)
    > 80%   Very high  — pattern is clearly present
    50–80%  High       — pattern is likely real
    30–50%  Moderate   — uncertain, needs timing confirmation
    < 30%   Low        — treat as noise, do not act

  TIMING CONFIDENCE  (how spread-out the peak/bottom estimates are)
    < ±2 months   High   — date is reliable
    ±2–5 months   Medium — right direction, wrong exact date
    > ±5 months   Low    — pattern exists but timing is guesswork

  MCMC CREDIBLE INTERVAL  (the sharpest timing signal — trust this most)
    < 1 month wide  — Imminent, very high confidence on timing
    1–3 months wide — Reliable range, act if pattern is also strong
    3–6 months wide — Useful ballpark, keep monitoring monthly
    > 6 months wide — Timing too uncertain, do not use for decisions

  {d}
  DECISION MATRIX

    Strong pattern  +  Tight MCMC (< 3 months)
      → High-conviction signal. Monitor closely.

    Strong pattern  +  Wide MCMC (> 6 months)
      → Pattern is real but timing unknown.
        Do not act yet — revisit monthly.

    Moderate/Weak pattern  (any timing)
      → Not actionable on its own.

    Anti-bubble imminent  (⚠ warning showing)
      → Potential buy zone forming.
        Less validated than bubble signals — use as supporting evidence.

  {d}
  KNOWN LIMITATIONS

    · Tight MCMC (narrow credible interval) = reliable TIMING,
      but only if pattern confidence is also High or Very high.
      Tight MCMC with Low pattern confidence can be a model artefact.

    · Assets in a permanent uptrend (indices, Bitcoin bull runs) may
      always show a bubble pattern. Weight TIMING confidence heavily —
      the signal only becomes actionable when tc estimates converge.

    · External shocks (policy changes, geopolitical events, earnings)
      can override any statistical signal instantly.

    · Anti-bubble detection is less validated in academic literature
      than bubble detection. Treat anti-bubble signals with more caution.

    · LPPL is one quantitative signal. Never use it as your only input.
""")
    print(f"{g}\n")


def analyze(
    prices: np.ndarray,
    date_index,
    name: str = "Asset",
    save_path: str = "lppl_result.png",
    future_prices: Optional[np.ndarray] = None,
    future_dates=None,
) -> tuple:
    """
    Run LPPL ensemble + optional Bayesian MCMC and display results.

    Args:
        prices:        1-D array of closing prices (up to the analysis date)
        date_index:    pandas DatetimeIndex of the same length
        name:          Asset name (used in chart title and printed output)
        save_path:     Where to save the chart PNG
        future_prices: Optional prices after the analysis date (historical mode)
        future_dates:  Corresponding DatetimeIndex for future_prices

    Returns:
        (bubble_result, anti_result) as EnsembleResult objects
    """
    n = len(prices)
    result, anti_result = run_ensemble(prices)

    _avg_cal_per_td = (date_index[-1] - date_index[0]).days / max(len(date_index) - 1, 1)

    def _near_tc_warning(ens: EnsembleResult):
        if ens.tc_median is None or len(ens.tc_values) == 0:
            return None
        today    = pd.Timestamp(date_index[-1])
        p10_date = _day_to_date(ens.tc_p10, date_index)
        d_to_p10 = (p10_date - today).days
        near_td  = int(90 / max(_avg_cal_per_td, 0.5))
        imm_frac = float(np.mean(ens.tc_values <= n + near_td))
        if d_to_p10 <= 0:
            return (f"  ⚠  Earliest estimate already passed — peak may have occurred recently\n"
                    f"     {imm_frac:.0%} of windows predict tc within 3 months")
        elif d_to_p10 <= 90:
            return (f"  ⚠  Earliest estimate only {d_to_p10} days away\n"
                    f"     {imm_frac:.0%} of windows predict tc within 3 months — entering critical zone")
        return None

    # ── Printed summary ───────────────────────────────────────────────────────
    bar = "═" * 54
    print(f"\n{bar}\n  BUBBLE ANALYSIS: {name}\n{bar}")

    print(f"\n  BUBBLE  (market peak — sell warning)\n  {'─'*44}")
    if result.tc_median is not None:
        tc_d = _day_to_date(result.tc_median, date_index)
        p10d = _day_to_date(result.tc_p10, date_index)
        p90d = _day_to_date(result.tc_p90, date_index)
        print(f"  Pattern confidence: {_tick(_pattern_level(result.bubble_fraction))}  {_pattern_confidence(result.bubble_fraction)}")
        print(f"  Timing confidence:  {_tick(_timing_level(result.tc_std))}  {_timing_confidence(result.tc_std)}")
        print(f"  Most likely peak:   {tc_d.strftime('%b %Y')}")
        print(f"  Likely range:       {p10d.strftime('%b %Y')}  →  {p90d.strftime('%b %Y')}")
        w = _near_tc_warning(result)
        if w:
            print(w)
        print(f"  → {_bottom_line(result.signal, tc_d, int(result.tc_median) - n)}")
    else:
        print(f"  → No upward bubble pattern detected.")

    print(f"\n  ANTI-BUBBLE  (market bottom — buy opportunity)\n  {'─'*44}")
    if anti_result.tc_median is not None:
        atc_d = _day_to_date(anti_result.tc_median, date_index)
        ap10d = _day_to_date(anti_result.tc_p10, date_index)
        ap90d = _day_to_date(anti_result.tc_p90, date_index)
        print(f"  Pattern confidence: {_tick(_pattern_level(anti_result.bubble_fraction))}  {_pattern_confidence(anti_result.bubble_fraction)}")
        print(f"  Timing confidence:  {_tick(_timing_level(anti_result.tc_std))}  {_timing_confidence(anti_result.tc_std)}")
        print(f"  Most likely bottom: {atc_d.strftime('%b %Y')}")
        print(f"  Likely range:       {ap10d.strftime('%b %Y')}  →  {ap90d.strftime('%b %Y')}")
        w = _near_tc_warning(anti_result)
        if w:
            print(w)
        print(f"  → {_bottom_line(anti_result.signal, atc_d, int(anti_result.tc_median) - n)}")
    else:
        print(f"  → No downward recovery pattern detected.")

    # ── Bayesian MCMC ─────────────────────────────────────────────────────────
    # Run on the dominant signal's top windows
    if (result.signal in ("strong", "moderate")
            and result.tc_median is not None and result.tc_median > n):
        mcmc_fits  = result.valid_fits
        mcmc_label = "bubble"
    elif (anti_result.signal in ("strong", "moderate")
            and anti_result.tc_median is not None and anti_result.tc_median > n):
        mcmc_fits  = anti_result.valid_fits
        mcmc_label = "anti-bubble"
    else:
        mcmc_fits  = []
        mcmc_label = None

    mcmc_samples = None
    if mcmc_fits:
        print(f"\n  Running Bayesian MCMC on top-5 {mcmc_label} windows…")
        try:
            mcmc_samples = _run_mcmc(prices, mcmc_fits, n_top=5, n_walkers=16, n_steps=400)
        except Exception as exc:
            print(f"  MCMC failed: {exc}")

        if mcmc_samples is not None:
            tc_dates  = [_day_to_date(tc, date_index) for tc in mcmc_samples]
            _dti      = pd.DatetimeIndex(tc_dates)
            tc_months = (_dti.tz_convert(None) if _dti.tz is not None else _dti).to_period("M")
            mc        = tc_months.value_counts().sort_index()
            total_smp = len(tc_months)
            cum = 0.0
            p50_m = p90_m = None
            for period, cnt in mc.items():
                cum += cnt / total_smp
                if p50_m is None and cum >= 0.50:
                    p50_m = period.to_timestamp()
                if p90_m is None and cum >= 0.90:
                    p90_m = period.to_timestamp()

            mc_median = _day_to_date(float(np.median(mcmc_samples)), date_index)
            mc_p5     = _day_to_date(float(np.percentile(mcmc_samples,  5)), date_index)
            mc_p95    = _day_to_date(float(np.percentile(mcmc_samples, 95)), date_index)

            ml           = _mcmc_level(mc_p5, mc_p95)
            mcmc_months  = max((mc_p95 - mc_p5).days / 30, 0)
            active_ens   = result if mcmc_label == "bubble" else anti_result
            p_lvl        = _pattern_level(active_ens.bubble_fraction)
            t_lvl        = _timing_level(active_ens.tc_std)

            label_upper  = mcmc_label.upper()
            peak_word    = "bottom" if mcmc_label == "anti-bubble" else "peak"
            print(f"\n  ─── Bayesian MCMC  [{label_upper}] ──────────────────────")
            print(f"  Posterior samples:       {len(mcmc_samples):,}")
            print(f"  Median {peak_word} estimate:  {mc_median.strftime('%b %Y')}")
            print(f"  90% credible interval:   {_tick(ml)}  {mc_p5.strftime('%b %Y')} → {mc_p95.strftime('%b %Y')}  ({mcmc_months:.0f} months wide)")
            if p50_m:
                print(f"  50% probability by:      {p50_m.strftime('%b %Y')}")
            if p90_m:
                print(f"  90% probability by:      {p90_m.strftime('%b %Y')}")
            print(f"\n  Overall {label_upper} signal:")
            print(f"{_overall_summary(p_lvl, t_lvl, ml)}")
            print(f"  ─────────────────────────────────────────────────────")
        else:
            print(f"  (emcee not installed — run: pip install emcee)")

    print(f"\n{bar}\n")
    _print_guide()

    _plot(
        prices, date_index, result, name,
        anti_result=anti_result,
        mcmc_samples=mcmc_samples,
        future_prices=future_prices,
        future_dates=future_dates,
        save_path=save_path,
    )

    return result, anti_result

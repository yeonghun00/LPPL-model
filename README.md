# LPPL Bubble Detector

Detect financial market bubbles and recovery signals using the **Log-Periodic Power Law (LPPL)** model — the same framework used by physicist Didier Sornette and his team to study the 2008 crash, the Nasdaq dot-com bubble, and others.

---

## What is LPPL?

When markets form a bubble, prices don't just rise exponentially — they accelerate **faster than exponential** while oscillating in a log-periodic pattern. This specific mathematical fingerprint tends to appear before major crashes.

```
ln(p(t)) = A + B(tc - t)^m + C(tc - t)^m · cos(ω · ln(tc - t) + φ)
```

The model fits this pattern across hundreds of analysis windows and asks: *do they all agree on when the crash (`tc`) is coming?* That convergence is the signal.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-window ensemble** | Fits LPPL across 125–750 day windows, all ending today. Signal is credible only when `tc` estimates converge |
| **Anti-bubble detection** | Detects accelerating *downward* patterns — potential market bottoms / buy signals |
| **Bayesian MCMC** | Uses `emcee` to produce a proper posterior probability distribution over the peak/bottom date |
| **Historical backtesting** | Re-run any analysis as of a past date with the actual future price overlaid on the chart |
| **Signal grading** | ✓ / ◎ / ✗ ticks on pattern confidence, timing confidence, and MCMC reliability |
| **Plain-English output** | Every number is translated into a clear, actionable interpretation |

---

## Installation

```bash
git clone https://github.com/yourusername/lppl.git
cd lppl
pip install -r requirements.txt
pip install emcee   # optional but strongly recommended — enables Bayesian probability bars
```

**Requirements:** Python 3.9+, `numpy`, `scipy`, `matplotlib`, `yfinance`, `pandas`

---

## Usage

```bash
# Current analysis
python run.py                    # S&P 500 (default)
python run.py TSLA               # Any ticker
python run.py BTC                # Bitcoin
python run.py NASDAQ             # Nasdaq

# Historical backtesting — analyse as if that date were today
# Shows actual future prices on the chart so you can judge model accuracy
python run.py TSLA 2021-11-01       # Right before Tesla's 2021 peak
python run.py "^GSPC" 2024-12-31   # Did the model see the Feb 2025 crash coming?
python run.py BTC 2021-03-01       # Crypto bubble
```

### Supported shorthands

| Input | Maps to |
|-------|---------|
| `sp500`, `s&p500` | `^GSPC` |
| `nasdaq` | `^IXIC` |
| `dow` | `^DJI` |
| `btc`, `bitcoin` | `BTC-USD` |
| `eth`, `ethereum` | `ETH-USD` |

Data window is automatically sized by asset type: **indices → 5 years**, **stocks → 3 years**, **crypto → 2 years**.

---

## Example output

### S&P 500 as of Dec 31, 2024 — correctly predicted the Feb 2025 peak

```
══════════════════════════════════════════════════════
  BUBBLE ANALYSIS: S&P 500  [as of 2024-12-31]
══════════════════════════════════════════════════════

  BUBBLE  (market peak — sell warning)
  ────────────────────────────────────────────
  Pattern confidence: ✓  Very high  — 97% of all timeframes detect the pattern
  Timing confidence:  ✓  High  — estimates cluster within ±1.5 months
  Most likely peak:   Feb 2025
  Likely range:       Jan 2025  →  Feb 2025
  ⚠  Earliest estimate only 1 days away
     84% of windows predict tc within 3 months — entering critical zone
  → The bubble pattern is mature and the peak may be imminent (around Feb 2025).

  ─── Bayesian MCMC  [BUBBLE] ──────────────────────
  Posterior samples:       16,000
  Median peak estimate:    Jan 2025
  90% credible interval:   ✓  Jan 2025 → Feb 2025  (1 months wide)
  50% probability by:      Jan 2025
  90% probability by:      Feb 2025

  Overall BUBBLE signal:
  Overall:  Pattern ✓  ·  Timing ✓  ·  MCMC ✓
     → HIGH CONVICTION — all three signals agree
```

> **Actual S&P 500 peak: February 19, 2025.** The model predicted Jan–Feb 2025 seven weeks in advance.

---

## How to interpret the report

### The three signals explained

| Signal | What it measures | ✓ if |
|--------|-----------------|------|
| **Pattern confidence** | % of analysis windows that detect the LPPL pattern | > 50% |
| **Timing confidence** | How tightly `tc` estimates cluster across different windows | Spread < ±2 months |
| **MCMC credible interval** | Bayesian posterior width for the peak/bottom date | < 3 months wide |

### Tick reference

| Tick | Level | Meaning |
|------|-------|---------|
| ✓ | High | Reliable — use this |
| ◎ | Medium | Useful direction, imprecise timing |
| ✗ | Low | Unreliable — do not act on this alone |

### Decision matrix

| Pattern | Timing | MCMC | What to do |
|---------|--------|------|------------|
| ✓ | ✓ | ✓ | **High conviction** — monitor closely |
| ✓ | ◎ | ✓ | Good signal — act if other factors align |
| ✓ | ✗ | any | Pattern is real but timing unknown — revisit monthly |
| ◎ or ✗ | any | any | Not actionable on its own |

### The ⚠ critical zone warning

Fires when the **earliest** `tc` estimate (10th percentile) is within 90 calendar days of today. Combined with high pattern confidence, this is the most urgent signal the model can produce.

---

## Known limitations

> **This is a research tool, not financial advice.**

**Secular uptrends** — assets that grow continuously (major indices, Bitcoin in a bull market) may always appear to be in a bubble. In these cases, weight **timing confidence** heavily. The signal is only actionable when timing tightens around a specific date.

**Anti-bubble is less validated** — LPPL was originally developed for upward bubble detection. Anti-bubble signals (accelerating declines) are less supported in academic literature. Treat them as supporting evidence, not a primary signal.

**External shocks** — geopolitical events, earnings surprises, and policy changes can break any statistical pattern instantly. LPPL cannot predict these.

**`tc` is a risk zone, not a trading date** — the model predicts when the accelerating pattern *ends*, not the exact price reversal date. Treat it as a window of elevated risk.

**One signal among many** — never use LPPL as your only input.

---

## Technical details

| Component | Approach |
|-----------|----------|
| **Optimisation** | Differential Evolution on 3D space (tc, m, ω) with OLS-linearised A, B, C, φ |
| **Speedup** | OLS linearisation (Filimonov & Sornette 2013) reduces 7D → 3D search (~4–6× faster) |
| **Parallelism** | `ProcessPoolExecutor` across all CPU cores |
| **Ensemble** | 125–750 day windows, all ending at the analysis date, step = 5 days |
| **Signal grading** | Bubble fraction + coefficient of variation (cv = std / median tc) |
| **Bayesian MCMC** | `emcee` EnsembleSampler, top-5 windows, 16 walkers × 400 steps, OLS-marginalized likelihood |
| **Data** | `yfinance` — stocks (3y default), indices (5y), crypto (2y) |

---

## Project structure

```
lppl/
├── run.py              # CLI entry point
└── lppl/
    ├── __init__.py     # Package exports
    ├── model.py        # Core LPPL model — fit(), fit_fast(), parameter constraints
    └── ensemble.py     # Multi-window ensemble, Bayesian MCMC, chart, report
```

---

## References

1. Sornette, D. (2003). *Why Stock Markets Crash* — Princeton University Press
2. Johansen, A. & Sornette, D. (2010). *Shocks, Crashes and Bubbles in Financial Markets*
3. Filimonov, V. & Sornette, D. (2013). *A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model* — Quantitative Finance

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Past pattern detection does not guarantee future results.

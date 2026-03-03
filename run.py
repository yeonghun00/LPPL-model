#!/usr/bin/env python3
"""
LPPL Bubble Detector

Usage:
    python run.py                    # S&P 500 (default), latest data
    python run.py TSLA               # Tesla, latest data
    python run.py TSLA 2024-06-01   # Tesla as of Jun 1 2024  (historical)
    python run.py 2024-06-01        # S&P 500 as of Jun 1 2024 (historical)

Historical mode re-runs the analysis exactly as it would have appeared on that date,
using only data available up to that day — useful for backtesting model signals.
"""

import re
import sys
import warnings

# yfinance uses pandas internals that trigger deprecation warnings we can't fix
warnings.filterwarnings("ignore", message=".*utcnow.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

import pandas as pd
import yfinance as yf

TICKER_MAP = {
    "sp500": "^GSPC", "s&p500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eth": "ETH-USD", "ethereum": "ETH-USD",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_period(symbol: str) -> str:
    if symbol.startswith("^"):
        return "5y"   # Index — bubbles develop over years
    if symbol.endswith("-USD"):
        return "2y"   # Crypto — fast bubble cycles
    return "3y"       # Stock default


def main():
    args = sys.argv[1:]

    symbol      = "^GSPC"
    target_date = None

    for arg in args:
        if _DATE_RE.match(arg):
            target_date = arg
        else:
            symbol = TICKER_MAP.get(arg.lower(), arg.upper())

    ticker_obj = yf.Ticker(symbol)

    if target_date:
        end_dt   = pd.Timestamp(target_date)
        # 875 calendar days covers the 750-day max window plus a ~125-day buffer
        start_dt = end_dt - pd.Timedelta(days=875)
        # yfinance end is exclusive — add one day to include the target date
        end_fetch = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"Fetching {symbol} for historical analysis (as of {target_date})…")
        df = ticker_obj.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_fetch,
        )
    else:
        period = get_period(symbol)
        print(f"Fetching {symbol} ({period})…")
        df = ticker_obj.history(period=period)

    if df.empty:
        print(f"No data returned for {symbol}. Check the ticker symbol.")
        sys.exit(1)

    prices     = df["Close"].values
    date_index = df.index

    try:
        name = ticker_obj.info.get("shortName", symbol)
    except Exception:
        name = symbol

    if len(prices) < 125:
        print(f"Only {len(prices)} days of data — need at least 125 for LPPL analysis.")
        sys.exit(1)

    print(f"Loaded {len(prices)} days for {name}")
    print(f"Period: {date_index[0].strftime('%Y-%m-%d')} → {date_index[-1].strftime('%Y-%m-%d')}")
    print(f"Price:  ${prices.min():.2f} – ${prices.max():.2f}\n")

    if target_date:
        as_of     = date_index[-1].strftime("%Y-%m-%d")
        label     = f"{name}  [as of {as_of}]"
        safe_sym  = symbol.replace("^", "").replace("-", "")
        save_path = f"lppl_{safe_sym}_{as_of}.png"
        print(f"Historical mode: analysing as if today = {as_of}")

        # Fetch what actually happened after the analysis date
        future_start = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        today_str    = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        future_prices = None
        future_dates  = None
        try:
            future_df = ticker_obj.history(start=future_start, end=today_str)
            if not future_df.empty:
                future_prices = future_df["Close"].values
                future_dates  = future_df.index
                print(f"Loaded {len(future_prices)} future days "
                      f"({future_df.index[0].strftime('%Y-%m-%d')} → "
                      f"{future_df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"Could not fetch future data: {e}")
        print()
    else:
        label         = name
        save_path     = "lppl_result.png"
        future_prices = None
        future_dates  = None

    from lppl import analyze
    analyze(prices, date_index, name=label, save_path=save_path,
            future_prices=future_prices, future_dates=future_dates)


if __name__ == "__main__":
    main()

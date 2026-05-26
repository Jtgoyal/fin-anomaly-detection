"""
data_loader.py — pulls daily OHLCV from Yahoo Finance and caches to disk.

Design choices:
- Use Ticker(...).history() instead of yf.download() — more reliable rate-limit handling.
- Cache to data/raw/{ticker}.csv so we never refetch unless forced.
- 5-year window — enough to include 2021 GME squeeze + COVID volatility.
- Auto-adjusted prices (handles splits/dividends without us thinking about it).
- Retry with backoff on transient failures.
"""

from pathlib import Path
import time
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "TSLA", "NVDA", "GME", "AMC"]
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_ticker(
    ticker: str,
    period: str = "5y",
    force: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download daily OHLCV for one ticker. Cached after first call."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{ticker}.csv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return df

    print(f"Downloading {ticker} ({period})...")
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval="1d", auto_adjust=True)
            if not df.empty:
                # Drop tz info from index for cleaner CSV round-trip
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index.name = "Date"
                df.to_csv(cache_path)
                return df
            print(f"  attempt {attempt}: empty data, retrying...")
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt}: {type(e).__name__}: {e}")
        time.sleep(2 * attempt)  # 2s, 4s, 6s backoff

    raise RuntimeError(
        f"Failed to download {ticker} after {max_retries} attempts. Last error: {last_err}"
    )


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download all configured tickers. Returns dict[ticker -> DataFrame]."""
    out = {}
    for t in TICKERS:
        out[t] = download_ticker(t, force=force)
        time.sleep(1)  # gentle pacing between tickers
    return out


if __name__ == "__main__":
    data = download_all()
    for t, df in data.items():
        print(f"{t}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
"""
features.py — turn raw OHLCV into the feature matrix every method consumes.

Features computed per ticker:
    - return            : daily simple return    (close_t / close_{t-1}) - 1
    - log_return        : daily log return       ln(close_t / close_{t-1})
    - volatility_20d    : rolling 20-day std of returns (NOT annualized — raw daily std)
    - volume_ratio      : today's volume / rolling 20-day average volume
    - return_zscore     : (return - rolling_mean_20d) / rolling_std_20d

All rolling windows use trailing data only (no lookahead).

Output: data/processed/{ticker}.csv — one row per trading day.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import download_all, TICKERS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def add_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Add anomaly-detection features to an OHLCV DataFrame."""
    df = df.copy()

    # Returns
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling volatility (raw daily std of returns over a 20-day trailing window)
    df["volatility_20d"] = df["return"].rolling(vol_window).std()

    # Volume ratio: today's volume vs trailing 20-day average
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(vol_window).mean()

    # Rolling z-score of return — feature for the statistical baseline, but also useful for everyone
    roll_mean = df["return"].rolling(vol_window).mean()
    roll_std = df["return"].rolling(vol_window).std()
    df["return_zscore"] = (df["return"] - roll_mean) / roll_std

    return df


def build_all_features(force: bool = False) -> dict[str, pd.DataFrame]:
    """Compute features for all tickers and save to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = download_all(force=force)

    out = {}
    for ticker, df in raw_data.items():
        feats = add_features(df)
        out_path = PROCESSED_DIR / f"{ticker}.csv"
        feats.to_csv(out_path)
        print(f"{ticker}: wrote {len(feats)} rows -> {out_path.name}")
        out[ticker] = feats
    return out


if __name__ == "__main__":
    data = build_all_features()
    # Quick sanity print: features look reasonable for one ticker
    sample = data["GME"][["Close", "return", "volatility_20d", "volume_ratio", "return_zscore"]]
    print("\nSample (GME, last 5 days):")
    print(sample.tail())
    print("\nGME feature stats:")
    print(sample.describe().round(4))
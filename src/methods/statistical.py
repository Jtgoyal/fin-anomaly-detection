"""
statistical.py — z-score and IQR baseline anomaly detection.

Both methods are rolling (trailing window only, no lookahead).
Today's value is NOT included in its own statistics — the window is the
30 days BEFORE today, so today's value can't bias its own threshold.

Input:  pd.Series of daily returns indexed by date
Output: pd.Series of bool flags (True = anomaly) indexed by the same dates
"""

import pandas as pd
import numpy as np


def zscore_flags(returns: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.Series:
    """
    Flag days where the rolling z-score of return exceeds +/- threshold.

    z = (today_return - trailing_mean) / trailing_std
    Flag if |z| > threshold.
    """
    # Shift by 1 so today is excluded from its own window
    trailing = returns.shift(1)
    roll_mean = trailing.rolling(window, min_periods=window).mean()
    roll_std = trailing.rolling(window, min_periods=window).std()

    # Avoid divide-by-zero (rare but happens in flat periods)
    roll_std = roll_std.replace(0, np.nan)

    z = (returns - roll_mean) / roll_std
    flags = z.abs() > threshold
    # NaN at the start (insufficient window) → not flagged
    return flags.fillna(False)


def iqr_flags(returns: pd.Series, window: int = 30, k: float = 3.0) -> pd.Series:
    """
    Flag days outside [Q1 - k*IQR, Q3 + k*IQR] of the trailing window.

    Standard k=1.5 catches "mild" outliers (too many false positives for stocks).
    k=3.0 is "extreme outlier" — better signal-to-noise for our use case.
    """
    trailing = returns.shift(1)
    q1 = trailing.rolling(window, min_periods=window).quantile(0.25)
    q3 = trailing.rolling(window, min_periods=window).quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    flags = (returns < lower) | (returns > upper)
    return flags.fillna(False)


if __name__ == "__main__":
    # Smoke test on GME
    from pathlib import Path
    PROC = Path(__file__).parent.parent.parent / "data" / "processed"
    gme = pd.read_csv(PROC / "GME.csv", parse_dates=["Date"], index_col="Date")

    z = zscore_flags(gme["return"], window=30, threshold=3.0)
    i = iqr_flags(gme["return"], window=30, k=3.0)

    print(f"GME — z-score (window=30, threshold=3.0):")
    print(f"  flagged {z.sum()} days out of {len(z)} ({z.mean()*100:.2f}%)")
    print(f"  first 5 flagged dates: {z[z].index[:5].strftime('%Y-%m-%d').tolist()}")

    print(f"\nGME — IQR (window=30, k=3.0):")
    print(f"  flagged {i.sum()} days out of {len(i)} ({i.mean()*100:.2f}%)")
    print(f"  first 5 flagged dates: {i[i].index[:5].strftime('%Y-%m-%d').tolist()}")

    # Did we catch the famous Jan 27 2021 squeeze?
    print(f"\nDid we catch 2021-01-27?")
    print(f"  z-score:  {z.loc['2021-01-27']}")
    print(f"  IQR:      {i.loc['2021-01-27']}")
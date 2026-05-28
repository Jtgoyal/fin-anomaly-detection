"""
statistical.py — z-score and IQR baseline anomaly detection.

Interface contract (every method in src/methods/ follows this):
    fn(features: pd.DataFrame, **params) -> pd.Series  # bool flags, indexed by date

Both methods are rolling (trailing window only, no lookahead).
Today's value is NOT included in its own statistics — the window is the
30 days BEFORE today, so today's value can't bias its own threshold.
"""

import pandas as pd
import numpy as np


def zscore_flags(features: pd.DataFrame, window: int = 30, threshold: float = 3.0) -> pd.Series:
    """
    Flag days where the rolling z-score of return exceeds +/- threshold.
    Uses only the `return` column from `features`.
    """
    returns = features["return"]
    trailing = returns.shift(1)
    roll_mean = trailing.rolling(window, min_periods=window).mean()
    roll_std = trailing.rolling(window, min_periods=window).std()
    roll_std = roll_std.replace(0, np.nan)

    z = (returns - roll_mean) / roll_std
    flags = z.abs() > threshold
    return flags.fillna(False)


def iqr_flags(features: pd.DataFrame, window: int = 30, k: float = 3.0) -> pd.Series:
    """
    Flag days outside [Q1 - k*IQR, Q3 + k*IQR] of the trailing window.
    Uses only the `return` column from `features`.
    """
    returns = features["return"]
    trailing = returns.shift(1)
    q1 = trailing.rolling(window, min_periods=window).quantile(0.25)
    q3 = trailing.rolling(window, min_periods=window).quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    flags = (returns < lower) | (returns > upper)
    return flags.fillna(False)


if __name__ == "__main__":
    from pathlib import Path
    PROC = Path(__file__).parent.parent.parent / "data" / "processed"
    gme = pd.read_csv(PROC / "GME.csv", parse_dates=["Date"], index_col="Date")

    z = zscore_flags(gme, window=30, threshold=3.0)
    print(f"GME z-score: flagged {z.sum()}/{len(z)} days ({z.mean()*100:.2f}%)")
    print(f"Caught 2021-01-27? {z.loc['2021-01-27']}")
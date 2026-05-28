"""
iforest.py — Isolation Forest anomaly detection.

Uses sklearn.ensemble.IsolationForest on a multi-feature input:
    [return, volatility_20d, volume_ratio]

Per-ticker scaling: StandardScaler fit on each ticker independently.
Rationale: AAPL's natural volatility is much smaller than GME's; a global
scaler would dilute the signal in the calmer tickers.

Interface contract:
    iforest_flags(features: pd.DataFrame, **params) -> pd.Series of bool
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["return", "volatility_20d", "volume_ratio"]


def iforest_flags(
    features: pd.DataFrame,
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.Series:
    """
    Flag anomalies using Isolation Forest on multi-feature input.

    Args:
        features: DataFrame with columns [return, volatility_20d, volume_ratio]
        contamination: expected fraction of anomalies (sets the score threshold)
        n_estimators: number of trees in the forest (more = more stable scores)
        random_state: for reproducibility

    Returns:
        pd.Series of bool flags, indexed by the same dates as features.
    """
    # Build feature matrix, drop rows with any NaN (early days when rolling features aren't ready)
    X = features[FEATURE_COLS].copy()
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    # Standardize per-ticker (this method gets called per-ticker by evaluate.py)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Fit Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    preds = model.fit_predict(X_scaled)  # -1 = anomaly, +1 = normal

    # Build the output series — False everywhere by default, True where the model said anomaly
    flags = pd.Series(False, index=features.index)
    flags.loc[X_valid.index] = (preds == -1)

    return flags


if __name__ == "__main__":
    from pathlib import Path
    PROC = Path(__file__).parent.parent.parent / "data" / "processed"
    gme = pd.read_csv(PROC / "GME.csv", parse_dates=["Date"], index_col="Date")

    flags = iforest_flags(gme, contamination=0.02)
    print(f"GME Isolation Forest (contamination=0.02):")
    print(f"  flagged {flags.sum()}/{len(flags)} days ({flags.mean()*100:.2f}%)")
    print(f"  caught 2021-01-27? {flags.loc['2021-01-27']}")
    print(f"  caught 2024-05-13? {flags.loc['2024-05-13']}")
    print(f"\n  first 10 flagged dates:")
    for d in flags[flags].index[:10]:
        print(f"    {d.date()}  return={gme.loc[d, 'return']*100:.2f}%  vol_ratio={gme.loc[d, 'volume_ratio']:.2f}")
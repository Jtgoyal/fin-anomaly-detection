"""
lof.py — Local Outlier Factor anomaly detection.

LOF scores each point by comparing its local density to its k nearest neighbors.
Lower density relative to neighbors → outlier.

Differs from Isolation Forest in a meaningful way:
- IF measures global isolation (path length across random trees).
- LOF measures local anomaly (density vs neighborhood).

This matters when anomalies cluster (e.g., August 2024 NVDA volatility):
IF might catch the whole cluster as isolated. LOF, because each point's
neighbors are similar, might NOT flag any of them — including the
Sept 3 event that emerged from that cluster. We'll measure.

Interface contract:
    lof_flags(features: pd.DataFrame, **params) -> pd.Series of bool
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["return", "volatility_20d", "volume_ratio"]


def lof_flags(
    features: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.02,
) -> pd.Series:
    """
    Flag anomalies using Local Outlier Factor on multi-feature input.

    Args:
        features: DataFrame with columns [return, volatility_20d, volume_ratio]
        n_neighbors: number of neighbors used in local density estimation.
                     Default 20 (sklearn default).
        contamination: expected fraction of anomalies — sets the decision threshold
                       on LOF scores. Lower = more selective.

    Returns:
        pd.Series of bool flags, indexed by features.index.

    Note: LOF is transductive — it can't easily score a new unseen point without
    refitting on the whole training set. That's a deployment limitation worth
    documenting. For this batch evaluation it's fine.
    """
    # Build feature matrix, drop rows with any NaN (early days)
    X = features[FEATURE_COLS].copy()
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    # Standardize per-ticker (same rationale as Isolation Forest)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Fit + predict in one call (novelty=False is the default for this use case)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    preds = model.fit_predict(X_scaled)  # -1 = anomaly, +1 = normal

    flags = pd.Series(False, index=features.index)
    flags.loc[X_valid.index] = (preds == -1)
    return flags


if __name__ == "__main__":
    from pathlib import Path
    PROC = Path(__file__).parent.parent.parent / "data" / "processed"

    print("=== GME LOF (n_neighbors=20, contamination=0.02) ===")
    gme = pd.read_csv(PROC / "GME.csv", parse_dates=["Date"], index_col="Date")
    flags = lof_flags(gme, n_neighbors=20, contamination=0.02)
    print(f"  flagged {flags.sum()}/{len(flags)} ({flags.mean()*100:.2f}%)")
    print(f"  caught 2021-01-27? {flags.loc['2021-01-27']}")
    print(f"  caught 2024-05-13? {flags.loc['2024-05-13']}")

    print("\n=== NVDA LOF (n_neighbors=20, contamination=0.02) ===")
    nvda = pd.read_csv(PROC / "NVDA.csv", parse_dates=["Date"], index_col="Date")
    flags = lof_flags(nvda, n_neighbors=20, contamination=0.02)
    print(f"  flagged {flags.sum()}/{len(flags)} ({flags.mean()*100:.2f}%)")
    print(f"  caught 2023-05-25 (AI boom)? {flags.loc['2023-05-25']}")
    print(f"  caught 2024-09-03 (mkt cap loss)? {flags.loc['2024-09-03']}")
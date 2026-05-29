"""
evaluate.py — score predicted anomaly flags against ground-truth labels.

Computes precision, recall, F1 per ticker AND in aggregate across all tickers.

Key design decision: "did we catch the labeled anomaly?" uses a tolerance window.
A label on 2021-01-27 is considered "caught" if the method flags any day in
[2021-01-26, 2021-01-28] (default tolerance: ±1 trading day).

Rationale: news catalysts can move prices on the announcement day OR the next
trading day depending on after-hours timing. Strict same-day matching would
over-penalize methods for off-by-one errors that are not meaningful.
"""

from pathlib import Path
from typing import Callable
import pandas as pd
import numpy as np

LABELS_PATH = Path(__file__).parent.parent / "data" / "labels" / "ground_truth.csv"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_ground_truth() -> pd.DataFrame:
    """Load the hand-labeled anomaly set."""
    df = pd.read_csv(LABELS_PATH, parse_dates=["date"])
    return df


def score_one_ticker(
    flags: pd.Series,
    labels: pd.DatetimeIndex,
    tolerance_days: int = 1,
) -> dict:
    """
    Score predicted flags against ground-truth labels for a single ticker.

    Args:
        flags: pd.Series of bool flags indexed by date (True = predicted anomaly)
        labels: DatetimeIndex of true anomaly dates for this ticker
        tolerance_days: how many trading days off a label can still count as caught

    Returns dict with: tp, fp, fn, precision, recall, f1, n_flagged, n_labels
    """
    flagged_dates = set(flags[flags].index)

    # For each label, check if any flag falls within tolerance window
    caught_labels = set()
    matched_flags = set()
    for label in labels:
        # Build the tolerance window around this label
        idx = flags.index
        if label not in idx:
            # Label date not in our data (shouldn't happen but be safe)
            continue
        label_pos = idx.get_loc(label)
        lo = max(0, label_pos - tolerance_days)
        hi = min(len(idx) - 1, label_pos + tolerance_days)
        window = set(idx[lo:hi + 1])
        hits = flagged_dates & window
        if hits:
            caught_labels.add(label)
            matched_flags.update(hits)

    tp = len(caught_labels)
    fn = len(labels) - tp
    # False positives: flags that didn't match any label's tolerance window
    fp = len(flagged_dates - matched_flags)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_flagged": len(flagged_dates),
        "n_labels": len(labels),
    }


def evaluate_method(
    method_fn: Callable[[pd.Series], pd.Series],
    method_name: str,
    tickers: list = None,
    tolerance_days: int = 1,
) -> pd.DataFrame:
    """
    Run a method on all tickers, score against ground truth, return one DataFrame.

    method_fn must take a return Series and return a bool flag Series of same length.
    """
    gt = load_ground_truth()
    if tickers is None:
        tickers = sorted(gt["ticker"].unique())

    rows = []
    for ticker in tickers:
        df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", parse_dates=["Date"], index_col="Date")
        df.attrs["ticker"] = ticker
        flags = method_fn(df)
        labels = pd.DatetimeIndex(gt[gt["ticker"] == ticker]["date"])
        metrics = score_one_ticker(flags, labels, tolerance_days=tolerance_days)
        metrics["method"] = method_name
        metrics["ticker"] = ticker
        rows.append(metrics)

    df_out = pd.DataFrame(rows)
    # Reorder columns
    cols = ["method", "ticker", "n_labels", "n_flagged", "tp", "fp", "fn",
            "precision", "recall", "f1"]
    return df_out[cols]


def aggregate_metrics(df: pd.DataFrame) -> dict:
    """
    Compute aggregate metrics across tickers (micro-averaged precision/recall/F1).

    Micro-average: sum TP/FP/FN across tickers, then compute P/R/F1.
    Rationale: each true positive matters equally regardless of which ticker.
    """
    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_flagged_total": int(df["n_flagged"].sum()),
        "n_labels_total": int(df["n_labels"].sum()),
    }


if __name__ == "__main__":
    # Smoke test: evaluate z-score baseline
    from src.methods.statistical import zscore_flags, iqr_flags

    print("=" * 60)
    print("Evaluating z-score (window=30, threshold=3.0)")
    print("=" * 60)
    results = evaluate_method(
       lambda df: zscore_flags(df, window=30, threshold=3.0),
        method_name="zscore_w30_t3",
    )
    print(results.to_string(index=False))
    print(f"\nAggregate: {aggregate_metrics(results)}")

    print("\n" + "=" * 60)
    print("Evaluating IQR (window=30, k=3.0)")
    print("=" * 60)
    results = evaluate_method(
        lambda df: iqr_flags(df, window=30, k=3.0),
        method_name="iqr_w30_k3",
    )
    print(results.to_string(index=False))
    print(f"\nAggregate: {aggregate_metrics(results)}")
"""
run_all.py — run every registered method on every ticker, save unified results.

Single source of truth for the comparison table in the README.
As you add Isolation Forest, LOF, LSTM AE, register them in METHODS below
and rerun this script. No other code needs to change.

Output:
    results/all_methods.csv — per-(method, ticker) rows
    results/summary.csv     — per-method aggregates for the README table
"""

from pathlib import Path
import pandas as pd

from src.evaluate import evaluate_method, aggregate_metrics
from src.methods.statistical import zscore_flags, iqr_flags

OUT_PATH = Path(__file__).parent.parent / "results" / "all_methods.csv"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"


# Method registry: (name, function)
# Adding a new method later = one new row here + one import.
METHODS = [
    ("zscore_w30_t3.0", lambda df: zscore_flags(df, window=30, threshold=3.0)),
    ("zscore_w30_t3.5", lambda df: zscore_flags(df, window=30, threshold=3.5)),
    ("iqr_w30_k3.0",    lambda df: iqr_flags(df, window=30, k=3.0)),
    # Day 5: ("iforest_c0.02", lambda df: iforest_flags(df, contamination=0.02)),
    # Day 6: ("lof_n20",       lambda df: lof_flags(df, n_neighbors=20)),
    # Day 9: ("lstm_ae_k2.5",  lambda df: lstm_ae_flags(df, threshold_k=2.5)),
]


def main():
    all_rows = []
    summary_rows = []

    for name, fn in METHODS:
        df = evaluate_method(fn, method_name=name)
        all_rows.append(df)

        agg = aggregate_metrics(df)
        summary_rows.append({
            "method": name,
            "tp": int(agg["tp"]),
            "fp": int(agg["fp"]),
            "fn": int(agg["fn"]),
            "precision": round(float(agg["precision"]), 4),
            "recall": round(float(agg["recall"]), 4),
            "f1": round(float(agg["f1"]), 4),
            "n_flagged_total": int(agg["n_flagged_total"]),
        })
        print(f"{name:25s}  P={agg['precision']:.3f}  R={agg['recall']:.3f}  F1={agg['f1']:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT_PATH, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    print(f"\nPer-ticker rows -> {OUT_PATH}")
    print(f"Summary table   -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
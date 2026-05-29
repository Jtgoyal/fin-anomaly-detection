"""
analyze_failures.py — structured failure analysis across all methods.

Answers four questions:
1. For each label, how many methods catch it?
2. For each method, which labels does it miss?
3. Per-ticker F1 spread across methods (which tickers are hardest?)
4. False-positive overlap: are there days many methods flag that aren't labels?

Output: results/failures/ directory with three CSVs.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.evaluate import load_ground_truth
from src.methods.statistical import zscore_flags, iqr_flags
from src.methods.iforest import iforest_flags
from src.methods.lof import lof_flags

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "failures"

# Best config per method family — what we actually want to characterize
METHODS = [
    ("zscore_t3.5",    lambda df: zscore_flags(df, window=30, threshold=3.5)),
    ("iqr_k3.0",       lambda df: iqr_flags(df, window=30, k=3.0)),
    ("iforest_c0.01",  lambda df: iforest_flags(df, contamination=0.01)),
    ("lof_n20_c0.01",  lambda df: lof_flags(df, n_neighbors=20, contamination=0.01)),
]


def label_catch_matrix() -> pd.DataFrame:
    """
    Rows: each labeled anomaly. Columns: each method. Cell: True if caught.
    """
    gt = load_ground_truth()
    rows = []

    # Precompute flags per (method, ticker)
    flags_cache = {}
    for ticker in sorted(gt["ticker"].unique()):
        df = pd.read_csv(PROC / f"{ticker}.csv", parse_dates=["Date"], index_col="Date")
        for name, fn in METHODS:
            flags_cache[(name, ticker)] = (fn(df), df)

    for _, label in gt.iterrows():
        ticker = label["ticker"]
        date = label["date"]
        row = {
            "ticker": ticker,
            "date": date.date(),
            "return_%": None,
            "vol_ratio": None,
            "event": label["event"][:50],
        }
        for name, _ in METHODS:
            flags, df = flags_cache[(name, ticker)]
            if date in df.index:
                pos = df.index.get_loc(date)
                window = df.index[max(0, pos - 1):min(len(df), pos + 2)]
                caught = any(flags.loc[d] for d in window if d in flags.index)
                row[name] = caught
                if row["return_%"] is None:
                    row["return_%"] = round(df.loc[date, "return"] * 100, 2)
                    row["vol_ratio"] = round(df.loc[date, "volume_ratio"], 2)
            else:
                row[name] = None  # date not in data
        row["n_methods_caught"] = sum(1 for n, _ in METHODS if row.get(n) is True)
        rows.append(row)

    return pd.DataFrame(rows)


def ticker_difficulty() -> pd.DataFrame:
    """For each ticker, F1 mean/std across methods. Hardest tickers have highest std or lowest mean."""
    all_results = pd.read_csv(ROOT / "results" / "all_methods.csv")
    # Restrict to best-config methods only for fair comparison
    method_names = [name for name, _ in METHODS]
    # Map our analysis methods to corresponding run_all.py names
    rename_map = {
        "zscore_t3.5":   "zscore_w30_t3.5",
        "iqr_k3.0":      "iqr_w30_k3.0",
        "iforest_c0.01": "iforest_c0.01",
        "lof_n20_c0.01": "lof_n20_c0.01",
    }
    keep = list(rename_map.values())
    df = all_results[all_results["method"].isin(keep)]
    return df.groupby("ticker").agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_min=("f1", "min"),
        f1_max=("f1", "max"),
        n_labels=("n_labels", "first"),
    ).round(3).sort_values("f1_mean")


def fp_overlap() -> pd.DataFrame:
    """
    Days flagged by 3 or 4 of the 4 methods that are NOT in the ground truth.
    Candidates for 'should-have-been-labeled' anomalies — interesting either way.
    """
    gt = load_ground_truth()
    gt_dates_by_ticker = {t: set(gt[gt["ticker"] == t]["date"]) for t in gt["ticker"].unique()}

    rows = []
    for ticker in sorted(gt["ticker"].unique()):
        df = pd.read_csv(PROC / f"{ticker}.csv", parse_dates=["Date"], index_col="Date")
        per_method_flags = {}
        for name, fn in METHODS:
            per_method_flags[name] = fn(df)

        for date in df.index:
            n_flagged = sum(per_method_flags[n].loc[date] for n in per_method_flags)
            if n_flagged >= 3 and date not in gt_dates_by_ticker.get(ticker, set()):
                rows.append({
                    "ticker": ticker,
                    "date": date.date(),
                    "n_methods_flagged": int(n_flagged),
                    "return_%": round(df.loc[date, "return"] * 100, 2),
                    "vol_ratio": round(df.loc[date, "volume_ratio"], 2),
                })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "n_methods_flagged", "return_%", "vol_ratio"])
    return pd.DataFrame(rows).sort_values(
        ["n_methods_flagged", "ticker", "date"],
        ascending=[False, True, True],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("(1) Per-label catch matrix")
    print("=" * 70)
    matrix = label_catch_matrix()
    cols = ["ticker", "date", "return_%", "vol_ratio", "n_methods_caught"] + [n for n, _ in METHODS]
    print(matrix[cols].to_string(index=False))
    matrix.to_csv(OUT_DIR / "label_catch_matrix.csv", index=False)

    print("\n" + "=" * 70)
    print("(2) Per-ticker difficulty (mean F1 across the 4 best-config methods)")
    print("=" * 70)
    difficulty = ticker_difficulty()
    print(difficulty.to_string())
    difficulty.to_csv(OUT_DIR / "ticker_difficulty.csv")

    print("\n" + "=" * 70)
    print("(3) Heavily-flagged non-labels (3+ methods agree, not in ground truth)")
    print("=" * 70)
    fps = fp_overlap()
    print(f"Found {len(fps)} dates flagged by 3+ methods that aren't in ground truth.")
    if len(fps) > 0:
        print("\nTop 15 by agreement:")
        print(fps.head(15).to_string(index=False))
    fps.to_csv(OUT_DIR / "fp_overlap.csv", index=False)

    print(f"\nSaved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
"""
save_baseline.py — run statistical baselines, save per-ticker and aggregate
metrics to results/baseline.json for later reference.
"""

import json
from pathlib import Path
from src.evaluate import evaluate_method, aggregate_metrics
from src.methods.statistical import zscore_flags, iqr_flags

RESULTS_PATH = Path(__file__).parent.parent / "results" / "baseline.json"


def run_baseline():
    output = {}

    configs = [
        ("zscore_w30_t2.5", lambda df: zscore_flags(df, 30, 2.5)),
        ("zscore_w30_t3.0", lambda df: zscore_flags(df, 30, 3.0)),
        ("zscore_w30_t3.5", lambda df: zscore_flags(df, 30, 3.5)),
        ("iqr_w30_k1.5", lambda df: iqr_flags(df, 30, 1.5)),
        ("iqr_w30_k3.0", lambda df: iqr_flags(df, 30, 3.0)),
    ]

    for name, fn in configs:
        df = evaluate_method(fn, method_name=name)
        output[name] = {
            "per_ticker": df.drop(columns=["method"]).to_dict(orient="records"),
            "aggregate": aggregate_metrics(df),
        }
        agg = output[name]["aggregate"]
        print(f"{name:25s}  P={agg['precision']:.3f}  R={agg['recall']:.3f}  F1={agg['f1']:.3f}  (TP={agg['tp']}, FP={agg['fp']}, FN={agg['fn']})")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    run_baseline()
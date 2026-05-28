# Notes — Project 2

## Day 1 

**What I built:**
- Repo skeleton, venv, requirements.
- `src/data_loader.py` — pulls 5y daily OHLCV for AAPL/TSLA/NVDA/GME/AMC, caches to `data/raw/`.
- `notebooks/01_eda.ipynb` — price charts, return distributions, rolling vol.


**What I'm worried about:**
My eval set has only 1 AMC label vs 3 for AAPL. AMC will get fewer chances to score. Need to be careful interpreting AMC method performance — small denominator means high variance in precision/recall.


5 of 10 labels involve macro shocks (COVID, tariff pause) that hit multiple tickers on the same day. If a method catches one, it likely catches all — F1 scores might look inflated.

**Design decisions I made today:**
- Used `auto_adjust=True` in yfinance so prices are split/dividend adjusted — avoids fake "anomalies" on ex-dividend days.
- Cached CSVs but did NOT commit them — reproducible from source.
- Tracked `data/labels/` even though `data/raw/` is gitignored — labels are irreplaceable.


## Day 2 —

**What I built:**
- `src/features.py` — 5 features per ticker: return, log_return, volatility_20d, volume_ratio, return_zscore (all using 20-day rolling window, trailing only).
- Extended data window from 5y to 7y after sanity check showed GME 2021 squeeze was outside original window.
- `data/labels/ground_truth.csv` — 10 hand-labeled anomaly dates across all 5 tickers, each verified against data + news source.

**Labeling protocol (decided today):**
- One label per news event, placed on the peak day (largest |return|).
- Rationale: in real surveillance, one flag per event is enough to trigger human review.
- Tradeoff: methods flagging adjacent days (e.g. 2024-05-14 right after the GME peak) will be marked false positives even when catching the same event. Accepted because the alternative inflates GME's share of the eval set.

**Hardest decision today:**
Whether to label both 2021-01-26 (+93%, 3.27x volume) and 2021-01-27 (+135% peak) for GME. Both were genuinely anomalous days with different news triggers. Picked only the peak per the labeling protocol, accepting that methods catching 1/26 will be marked false positives. Documented this tradeo

**What surprised me:**
GME volume_ratio on the Jan 27 peak was only 1.59x — counterintuitively low. The rolling 20-day average had already exploded during the buildup, so today-vs-trailing-avg normalized away the actual spike. This is a real feature engineering blind spot — volume_ratio systematically underdetects multi-day events.


The Twitter-deal-close candidate (TSLA 2022-10-28) didn't actually produce an anomalous single-day move — biggest move in the window was only +5.3%, not anomalous for TSLA. I dropped this candidate from the eval set. Letting data veto candidates beats forcing labels to match my prior assumptions.

**Design decisions:**
- 7-year window instead of 5-year to capture GME/AMC 2021 squeezes.
- Labeled the SAME news event for multiple tickers (2025-04-09 tariff pause for AAPL+TSLA+NVDA) — necessary because evaluation is per-ticker, so each ticker needs its own anomaly date.
- Skipped the TSLA 2022-10-28 candidate because data didn't support it (largest move in window was only 5%).

**Questions for tomorrow:**
- For z-score baseline: threshold of 2.5, 3, or sweep? Probably sweep, but starting at 3 keeps things conservative.
- Does the eval framework score each ticker separately or in aggregate? Decide on Day 3.

**Labeling protocol (for multi-day events):**
- One label per news event, placed on the peak day (the day with the largest |return|).
- Rationale: in real surveillance, one flag per event is enough to trigger human review.
- Tradeoff: a method that flags 2024-05-14 (still +60%!) gets marked as a false positive.
  Accepted because the alternative (label every day of a run) would let GME dominate the eval set.


## Day 3 —

**What I built:**
- `src/methods/statistical.py` — z-score and IQR with trailing 30-day windows, today excluded from its own window via `.shift(1)`.
- `src/evaluate.py` — per-ticker P/R/F1 with ±1 trading day tolerance.
- `scripts/save_baseline.py` — sweeps 5 configs (3 z-score, 2 IQR), saves to `results/baseline.json`.

**Numbers I shipped:**
- Best baseline: zscore_w30_t3.5 with aggregate F1=0.133, P=0.073, R=0.800 (TP=8, FP=102, FN=2).
- z-score at t=2.5 and t=3.0 both hit R=0.9 — one label is unreachable even at the loosest threshold.
- IQR was strictly worse than z-score at every threshold. Fat-tailed return distributions mean percentile thresholds over-fire.

**Specific failure mode identified:**
- z-score misses NVDA 2024-09-03 (−9.53% return) even at threshold=2.5.
- Root cause: the August 2024 NVDA volatility inflated the trailing 30-day std, so a 9.5% move registered as only ~2σ.
- Rolling-window methods systematically miss anomalies that emerge FROM volatile regimes.
- Prediction for Day 5: Isolation Forest with multi-feature input (return + volatility + volume_ratio together) may catch this where return-only z-score cannot. Will verify.

**Hardest decision today:**
- The ±1 trading day tolerance window in `evaluate.py`. After-hours news lands on day N or N+1 depending on timing; strict same-day matching would over-penalize methods. Documented in the docstring.

**What surprised me:**
- IQR k=1.5 flagged 448 days — way too many. The "textbook" outlier threshold is designed for normal distributions; stock returns are not normal.
- z-score recall plateaus at 0.9 — loosening the threshold from 3.0 to 2.5 adds FPs but doesn't catch the missing label.

**Design decisions:**
- Used `.shift(1)` so today is excluded from its own window. Otherwise anomalies bias their own threshold downward.
- Micro-averaged aggregate (sum TP/FP/FN, then compute). Each true positive matters equally regardless of ticker.
- Tolerance window = ±1 trading day.

**Questions for tomorrow:**
- For Day 4 refactor: should every method return a Series of bool flags or anomaly *scores*? Scores enable PR curves later.
- For classical ML methods (IF, LOF): switch from single-feature to multi-feature input (return + volatility + volume_ratio).


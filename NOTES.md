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

**Questions for tomorrow:**
- How do I decide which days to hand-label as anomalies without leaking info into eval?


## Day 2 — <today's date>

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
GME volume_ratio on the Jan 27 peak was only 1.59x — counterintuitively low. The rolling 20-day average had already exploded during the buildup, so today-vs-trailing-avg normalized away the actual spike. This is a real feature engineering blind spot — volume_ratio systematically underdetects multi-day events. Worth flagging on Day 13.


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
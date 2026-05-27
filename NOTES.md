# Notes — Project 2

## Day 1 

**What I built:**
- Repo skeleton, venv, requirements.
- `src/data_loader.py` — pulls 5y daily OHLCV for AAPL/TSLA/NVDA/GME/AMC, caches to `data/raw/`.
- `notebooks/01_eda.ipynb` — price charts, return distributions, rolling vol.

**What surprised me:**
- <fill in — e.g. "GME's max single-day return was X% on Y date" or "TSLA's vol is higher than I expected">

**What I'm worried about:**
- <fill in honestly — e.g. "5 years isn't a lot of training data for an LSTM" or "the eval set on Day 2 will be subjective">

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
- [Write yours — example: "Whether to label both 2021-01-26 (+93%) and 2021-01-27 (+135%) for GME. Picked only 2021-01-27 per the peak-only rule, even though both were genuinely anomalous days."]

**What surprised me:**
- [Write yours — example: "GME volume_ratio on the Jan 27 peak was only 1.6x, because the rolling 20-day avg had already exploded during the buildup. Features that compare to recent history can MASK multi-day events."]
- [And: "TSLA Twitter deal close (2022-10-28) didn't produce a notable single-day move — the market priced it in over weeks. I dropped this candidate from the eval set."]

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
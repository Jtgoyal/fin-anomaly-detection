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
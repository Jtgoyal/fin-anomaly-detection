# Financial Time-Series Anomaly Detection

> Compare 4+ anomaly detection methods on real stock price data, with a hand-labeled evaluation set and a live dashboard.

**Status:** 🚧 Day 1 of 14 — data pipeline + EDA complete.

## What this is

Statistical anomalies in daily stock returns — days where return deviates significantly from recent behavior. Goal: flag unusual days that would warrant human review, similar to what fintech surveillance teams do. **Not** a price-prediction project.

## Tickers

AAPL, TSLA, NVDA, GME, AMC — chosen for a mix of large-cap stability (AAPL, NVDA), high volatility (TSLA), and famous anomalies (GME, AMC 2021 short squeeze).

## Methods (planned)

| Method | Family | Status |
|---|---|---|
| Z-score / IQR | Statistical baseline | ⏳ |
| Isolation Forest | Classical ML | ⏳ |
| Local Outlier Factor | Classical ML | ⏳ |
| LSTM Autoencoder | Deep learning (trained from scratch) | ⏳ |

Final comparison table with measured precision / recall / F1 on a hand-labeled eval set lands by Day 13.

## Repo layout
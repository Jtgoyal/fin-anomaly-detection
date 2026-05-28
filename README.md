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

## Results (preliminary — Day 4 of 14)

Evaluated against a hand-labeled set of 10 verified anomaly dates across 5 tickers (AAPL, AMC, GME, NVDA, TSLA). Matches use a ±1 trading day tolerance to account for after-hours news.

Aggregate metrics (micro-averaged across tickers):

| Method | Precision | Recall | F1 | Flagged | Notes |
|---|---|---|---|---|---|
| z-score, t=3.0 | 0.049 | 0.90 | 0.092 | 187 | Best baseline recall |
| z-score, t=3.5 | 0.073 | 0.80 | **0.133** | 112 | Best baseline F1 |
| IQR, k=3.0 | 0.057 | 0.60 | 0.104 | 109 | Misses more labels than z-score |
| Isolation Forest | TBD | TBD | TBD | — | Day 5 |
| Local Outlier Factor | TBD | TBD | TBD | — | Day 6 |
| LSTM Autoencoder | TBD | TBD | TBD | — | Day 8–9 |

### Known limitations so far

- **z-score blind spot:** misses NVDA 2024-09-03 (−9.5% return, largest single-day market cap loss in history at the time) even at threshold=2.5. The August 2024 NVDA volatility had inflated the trailing 30-day std, so a 9.5% drop only registered at ~2σ. Rolling-window adaptive methods systematically miss anomalies that emerge *from* volatile regimes.
- **IQR over-fires at textbook k=1.5:** flagged 448 days. Stock returns are fat-tailed, so percentile-based outlier thresholds designed for normal distributions over-fire.
- **IQR misses both TSLA labels at k=3.0:** TSLA's return distribution has wide quartiles, making the k=3 bounds too loose to flag the labeled events. IQR's poor performance on TSLA drags its aggregate recall to 0.6.
- **Eval set bias:** 10 labels skew toward news-driven extreme moves (squeezes, earnings surprises, macro shocks). Methods are likely benchmarked against easier anomalies than they would face in production.
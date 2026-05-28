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

## Results

Evaluated against a hand-labeled set of 10 verified anomaly dates across 5 tickers (AAPL, AMC, GME, NVDA, TSLA). Matches use a ±1 trading day tolerance to account for after-hours news.

Aggregate metrics (micro-averaged across tickers):

| Method | Precision | Recall | F1 | Flagged | Notes |
|---|---|---|---|---|---|
| z-score, t=3.0 | 0.049 | 0.90 | 0.092 | 187 | Best baseline recall |
| z-score, t=3.5 | 0.073 | 0.80 | 0.133 | 112 | Best baseline F1 |
| IQR, k=3.0 | 0.057 | 0.60 | 0.104 | 109 | Misses both TSLA labels |
| **Isolation Forest, c=0.01** | **0.095** | **0.80** | **0.170** | 105 | **Best F1 so far** |
| Isolation Forest, c=0.02 | 0.054 | 0.90 | 0.101 | 166 | Catches 9/10 at cost of precision |
| Isolation Forest, c=0.05 | 0.024 | 1.00 | 0.046 | 416 | Perfect recall, precision collapses |
| LOF, n=20, c=0.01 | 0.091 | 0.80 | 0.163 | 110 | Within 4% of IF |
| LOF, n=20, c=0.02 | 0.047 | 0.80 | 0.089 | 170 | |
| LOF, n=50, c=0.02 | 0.054 | 0.90 | 0.101 | 167 | Wider neighborhood catches 1 more label |
| LSTM Autoencoder | TBD | TBD | TBD | — | Day 8–9 |

### Key finding: a shared blind spot

Three independent point-wise methods — z-score (rolling statistic), Isolation Forest (global feature-space isolation), and LOF (local-density comparison) — all miss the **same two labeled anomalies** at their optimal configurations:

- **NVDA 2024-09-03** (−9.5% return, 1.4× volume) — the largest single-day market cap loss in history at the time
- **TSLA 2022-04-26** (−12.2% return, 1.9× volume) — Tesla drops on Musk's Twitter acquisition announcement

Both have moderate-magnitude returns (~10%) on stocks whose distributions frequently produce moves that size. Compare to the labels these methods *catch* (+24% NVDA on AI guidance, +23% TSLA on tariff pause): the caught events are simply larger in magnitude.

**The structural limitation:** point-wise methods — whether based on rolling statistics, isolation paths, or density comparisons — cannot distinguish "a real anomaly in a stable stock" from "an ordinary day in a volatile stock." They measure *whether today is unusual*, not *whether the pattern leading up to today is unusual*. This motivates the temporal approach: the LSTM autoencoder (Day 8–9) operates on 30-day sequences, asking whether the entire recent shape is something the model has seen before. We will measure whether it closes the gap.

### Known limitations so far

- **z-score blind spot:** misses NVDA 2024-09-03 (−9.5% return, largest single-day market cap loss in history at the time) even at threshold=2.5. The August 2024 NVDA volatility had inflated the trailing 30-day std, so a 9.5% drop only registered at ~2σ. Rolling-window adaptive methods systematically miss anomalies that emerge *from* volatile regimes.
- **IQR over-fires at textbook k=1.5:** flagged 448 days. Stock returns are fat-tailed, so percentile-based outlier thresholds designed for normal distributions over-fire.
- **IQR misses both TSLA labels at k=3.0:** TSLA's return distribution has wide quartiles, making the k=3 bounds too loose to flag the labeled events. IQR's poor performance on TSLA drags its aggregate recall to 0.6.
- **Eval set bias:** 10 labels skew toward news-driven extreme moves (squeezes, earnings surprises, macro shocks). Methods are likely benchmarked against easier anomalies than they would face in production.
- **Isolation Forest blind spot:** at the optimal F1 config (c=0.01), IF misses NVDA 2024-09-03 and TSLA 2022-04-26 — the same NVDA label z-score missed, plus one TSLA label. Both share a pattern: large-but-not-extreme moves (~10%) in stocks whose return distribution often produces moves that size. Point-wise isolation methods can't distinguish "an anomaly in a calm stock" from "a normal move in a volatile stock." This motivates the temporal approach (LSTM autoencoder).
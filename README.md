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

Evaluated against a hand-labeled set of 13 verified anomaly dates across 5 tickers (AAPL, AMC, GME, NVDA, TSLA). Matches use a ±1 trading day tolerance to account for after-hours news.

Aggregate metrics (micro-averaged across tickers):

| Method | Precision | Recall | F1 | Flagged | Notes |
|---|---|---|---|---|---|
| z-score, t=3.0 | 0.065 | 0.923 | 0.121 | 187 | Best baseline recall |
| z-score, t=3.5 | 0.100 | 0.846 | 0.179 | 112 | Best baseline F1 |
| IQR, k=3.0 | 0.086 | 0.692 | 0.152 | 109 | Misses both TSLA labels |
| **Isolation Forest, c=0.01** | **0.131** | **0.846** | **0.227** | 105 | **Best F1** |
| Isolation Forest, c=0.02 | 0.071 | 0.923 | 0.133 | 166 | |
| Isolation Forest, c=0.05 | 0.031 | 1.000 | 0.060 | 416 | Perfect recall, precision collapses |
| LOF, n=20, c=0.01 | 0.125 | 0.846 | 0.218 | 110 | Within 4% of IF |
| LOF, n=20, c=0.02 | 0.065 | 0.846 | 0.121 | 170 | |
| LOF, n=50, c=0.02 | 0.071 | 0.923 | 0.133 | 167 | Wider neighborhood catches 1 more label |
| LSTM Autoencoder | TBD | TBD | TBD | — | Day 8–9 |

See **[FAILURE_ANALYSIS.md](./FAILURE_ANALYSIS.md)** for the structured breakdown of which methods catch which labels and why.

### Key findings so far

- **One universally hard label.** NVDA 2024-09-03 (−9.5%, largest single-day market cap loss in history at the time) is missed by all four point-wise methods at their best configurations. The August 2024 NVDA volatility had already raised the local baseline, so adaptive methods became blind to the actual event.
- **Method-specific complementarity.** Z-score catches TSLA 2022-04-26 (Twitter announcement) that IF and LOF both miss. IF and LOF catch AAPL 2020-03-16 (COVID crash) that statistical methods both miss. Methods are *not* strictly ordered — they have complementary failure modes. A z-score + IF ensemble would catch 12/13 labels.
- **TSLA is the hardest ticker.** Highest F1 variance across methods (std 0.074). IQR scores F1=0.0 on TSLA. Method choice matters most for this ticker.

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
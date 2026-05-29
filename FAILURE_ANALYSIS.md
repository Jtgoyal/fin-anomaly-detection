# Failure Analysis

A structured account of what each anomaly-detection method gets right, what it gets wrong, and why. The point isn't to show which method is best — it's to show *under what conditions each method fails*, so a future deployment can choose appropriately.

## Methodology

Each method is evaluated against a hand-labeled ground-truth set of **13 anomaly dates** across 5 tickers (AAPL, AMC, GME, NVDA, TSLA). Matches use a ±1 trading day tolerance to account for after-hours news. The analysis below compares the best-F1 configuration per method family:

| Family | Configuration | Aggregate F1 |
|---|---|---|
| Statistical (z-score) | window=30, threshold=3.5 | 0.179 |
| Statistical (IQR) | window=30, k=3.0 | 0.152 |
| Isolation Forest | contamination=0.01 | **0.227** |
| Local Outlier Factor | n_neighbors=20, contamination=0.01 | 0.218 |

## Finding 1 — One universally hard label

**NVDA 2024-09-03 (return −9.5%, volume 1.4×)** is the only labeled anomaly that ALL four methods miss at their optimal configurations.

This date was the largest single-day market-cap loss in history at the time (~$280B), driven by AI demand concerns and an antitrust subpoena. The news catalyst is unambiguous; the data signal is not.

Why all four methods miss it:
- **z-score (3.5σ):** the August 2024 NVDA price action had already inflated the trailing 30-day standard deviation. A −9.5% move only registered at ~2σ against that elevated baseline.
- **Isolation Forest:** the (return, volatility, volume) point for Sept 3 sits near other August 2024 points in feature space. Not isolated enough to score in the top 1% of anomalies.
- **LOF:** the 20-nearest-neighbors of Sept 3 are mostly other volatile early-September days. Local density is similar to those neighbors, so Sept 3 doesn't stand out locally.
- **IQR (k=3.0):** Sept 3's return falls within the trailing 30-day Q1±3·IQR bounds because the recent IQR was wide.

**Common root cause:** all four methods are *adaptive* — they calibrate against recent local context. When a real anomaly emerges from a period that already established elevated baseline behavior, every adaptive method becomes blind to it. This is a class-level limitation of point-wise scoring.

This motivates Day 8–9: an LSTM autoencoder operates on **30-day sequences**, asking whether the entire recent shape is something the model has seen before. Whether that closes the gap is the open question.

## Finding 2 — Method-specific complementarity

Three labels are caught by *some* methods but not others. This refutes the cleaner-but-wrong story that methods are strictly ordered.

| Label | Return | Methods catching | Pattern |
|---|---|---|---|
| TSLA 2022-04-26 (Twitter) | −12.2% | z-score only (1/4) | Statistical wins; ML methods all miss |
| AAPL 2020-03-16 (COVID) | −12.9% | IF + LOF only (2/4) | ML wins; statistical methods miss |
| TSLA 2025-04-09 (tariff) | +22.7% | all except IQR (3/4) | IQR's wide quartile bounds hurt it on TSLA |

The takeaway: **a deployment that ran z-score + IF in parallel would catch 12/13 labels** — every label except the universally-hard NVDA 2024-09-03. Neither method alone catches 12; their failures are complementary. This is a real opportunity for ensembling that the project does not explore further but documents.

## Finding 3 — Per-ticker difficulty

| Ticker | Mean F1 | F1 range | Reason |
|---|---|---|---|
| **NVDA** | 0.340 | [0.273, 0.500] | Easiest. Anomalies are either extreme earnings or macro shocks — both readily detected. |
| **AAPL** | 0.253 | [0.160, 0.316] | AAPL's calm baseline makes anomalies stand out in feature space. |
| **GME** | 0.174 | [0.121, 0.235] | High baseline volatility raises the noise floor. |
| **AMC** | 0.169 | [0.111, 0.210] | Similar dynamics to GME. |
| **TSLA** | 0.095 | [0.000, 0.182] | **Highest F1 variance across methods.** IQR scores 0.0 on TSLA. Method choice matters most here. |

TSLA's variance reflects that its return distribution is wide enough to defeat percentile methods but the moderate-magnitude moves (Twitter, tariff response) can still be caught by methods that use multiple features (z-score on returns alone misses Twitter; IQR misses both).

## Finding 4 — False-positive overlap suggests labeling incompleteness

Across all four methods at best configurations, **38 dates outside the ground-truth set are flagged by 3 or all 4 methods**. Spot-checks of the highest-agreement dates revealed several real anomalies that had been missed during the initial 90-minute labeling session:

| Date | Return | News |
|---|---|---|
| AMC 2021-06-02 | +95.2% | AMC June squeeze peak |
| NVDA 2024-02-22 | +16.4% | Q4 FY2024 earnings (data center revenue beat) |
| NVDA 2025-04-09 | +18.7% | Tariff pause (sister event to existing AAPL/TSLA labels) |

These three were verified against published news and added to the ground-truth set. The remaining 35 high-agreement non-labels include further likely-real anomalies (e.g., AMC 2024-05-13 sympathy rally with the labeled GME event), but were not added to avoid the impression of moving goalposts. Their existence is documented but their resolution is left as future work.

**Methodological note.** Using model agreement to *find* labels is methodologically OK only if every added label is verified against an independent source (news article, press release). The label set is not "what the models agree on" — it is "real events documented in news, some of which were initially missed during manual labeling." The verification breaks the circularity.

## Finding 5 — IF c=0.05 sets the recall ceiling

The most permissive IF configuration (`contamination=0.05`) achieves **perfect recall (1.000)** — including the universally-hard NVDA 2024-09-03. But aggregate precision drops to 0.031 (one true positive per ~30 flags). The label is detectable; the threshold required to detect it is the threshold at which everything else gets detected too.

This implies the issue with Sept 3 isn't that the methods *can't* see it — it's that no per-method threshold separates Sept 3 from a large set of normal-but-volatile days.

## Limitations of this analysis

- The eval set (13 labels) is too small for statistically significant comparisons between methods within 1–2 F1 points of each other. The headline numbers should be read as *directional*, not *precise*.
- All labels are large news-driven events. Quiet/subtle anomalies (e.g., gradual regime shifts) are not represented.
- Ticker selection skews toward high-volatility names (GME, AMC, TSLA). A broader portfolio (SPY, MSFT, KO) would test more "calm baseline" behavior.

## Practical implications for the LSTM autoencoder (Days 8–9)

Three predictions to test:

1. **NVDA 2024-09-03 prediction:** the LSTM, operating on 30-day sequences, may distinguish "a calm pre-Sept regime followed by a sharp drop" from "an extended volatile regime that continued." If it catches this label, that's evidence that temporal context resolves the universal blind spot. If it does not, the limitation is more fundamental than method class.

2. **TSLA 2022-04-26 prediction:** the LSTM should catch this because the surrounding context is calm — TSLA had been steady for weeks before the Twitter announcement, then dropped 12%. A sequence-aware method should see that shape as anomalous even when the point itself isn't extreme.

3. **AAPL 2020-03-16 prediction:** already caught by IF and LOF. LSTM should also catch it. If LSTM misses something IF catches, the project's narrative becomes more interesting (specialization, not strict ordering).
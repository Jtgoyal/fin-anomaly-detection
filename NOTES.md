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


## Day 4 — May 28, 2026

**What I built:**
- Refactored all methods to take `pd.DataFrame` instead of `pd.Series`. Single consistent interface: `fn(features, **params) -> pd.Series of bool`. Lets future methods (IF, LSTM AE) read multiple feature columns without changing the eval contract.
- `scripts/run_all.py` — single script with a METHODS registry. Adding a new method on Day 5+ is a 2-line change here, no other code touched.
- `results/all_methods.csv` and `results/summary.csv` — single source of truth that the README results table reads from.
- README "Results" section v0 with the baseline filled in and IF/LOF/LSTM TBD rows.

**Hardest decision today:**
- Whether to pass DataFrames or to keep the Series interface and have IF/LOF compute features inside the method. Picked DataFrame in / Series out because (a) all methods share the same feature pipeline so duplicating feature code per method = bugs waiting to happen, (b) the LSTM autoencoder will need sliding windows of multiple features anyway.

**What surprised me:**
- Refactoring a 30-line method took 5 minutes. Wiring it through evaluate.py + save_baseline.py + run_all.py and re-verifying every number unchanged took 90% of the session. Plumbing dominates new code for systems work — a thing I underestimate every time.

**Design decisions:**
- METHODS registry as a list of (name, lambda) tuples instead of a dict. Order preserved for the comparison table.
- One results CSV with per-ticker rows + one summary CSV with per-method aggregates. Both exist because README wants summary; failure analysis wants per-ticker.

**Questions for tomorrow:**
- For Isolation Forest contamination parameter: sweep {0.01, 0.02, 0.05} or pick one and defend it? Probably sweep, mirror the baseline pattern.


## Day 5 — May 29, 2026

**What I built:**
- `src/methods/iforest.py` — Isolation Forest on multi-feature input (return, volatility_20d, volume_ratio). Per-ticker StandardScaler.
- Plugged 3 IF configs (c=0.01, 0.02, 0.05) into `scripts/run_all.py`. Adding the method took ~5 minutes of work in run_all.py thanks to Day 4's refactor.

**Prediction vs reality:**
- Predicted IF would catch NVDA 2024-09-03 because multi-feature view would isolate the combo. WRONG at c=0.01 and c=0.02. Only c=0.05 catches it (and that's the config with 1.0 recall on everything).
- Predicted IF would catch both TSLA labels. RIGHT at c=0.02 and looser; missed 2022-04-26 at the optimal c=0.01.

**Best result so far:**
- iforest_c0.01: F1=0.170 (vs prior best 0.133 from zscore_w30_t3.5). 28% relative F1 improvement. Same recall (0.8) but precision lifted from 0.073 to 0.095. The multi-feature view earns its keep.

**Failure mode #3 identified:**
- IF at optimal config misses NVDA 2024-09-03 AND TSLA 2022-04-26. Both labels share a pattern: ~10% moves on stocks whose distribution often produces moves that size. IF measures "how isolated is this point" — but a normal-sized move for that ticker isn't isolated, even if it has a clear news catalyst. Point-wise methods (z-score, IQR, IF) all share this blind spot in different forms. This is precisely the gap LSTM AE needs to close.

**What surprised me:**
- At c=0.05, IF achieves perfect recall (1.0) but F1 collapses to 0.046. The PR trade-off is steep — there's no contamination value that keeps both high.
- Per-ticker performance is wildly different: AAPL F1=0.316, TSLA F1=0.100. Aggregate F1 hides this. The README should probably show per-ticker breakdowns too on Day 13.

**Design decisions:**
- StandardScaler fit per-ticker, not globally. Global would dilute AAPL's signal (small ranges) under GME's (huge ranges).
- n_estimators=200 (the sklearn default of 100 produces noticeably noisier scores between runs).
- random_state=42 for reproducibility.

**Questions for tomorrow:**
- LOF should behave differently because it scores points against local neighborhoods, not global isolation. Will it catch NVDA 2024-09-03 where IF didn't? Prediction: probably also no, for the same reason — the neighborhood IS volatile, so the point isn't locally anomalous either.


## Day 6 

**What I built:**
- `src/methods/lof.py` — Local Outlier Factor on the same multi-feature input as IF, per-ticker StandardScaler.
- Three LOF configs in `scripts/run_all.py` (n_neighbors=20 with c=0.01 and 0.02; n_neighbors=50 with c=0.02).

**Best result:**
- LOF n=20, c=0.01: F1=0.163, P=0.091, R=0.800. Caught 8/10.
- Within 4% of IF's F1 (0.170). Multi-feature view dominates the choice of algorithm.

**Prediction vs reality:**
- Predicted LOF would also miss NVDA 2024-09-03 because the neighborhood IS volatile. CORRECT.
- Predicted nothing about TSLA 2022-04-26. LOF also misses it.
- Most striking: LOF and IF agree on EVERY one of the 4 NVDA + TSLA label cases. Same catches, same misses. Two algorithms with very different mechanisms, identical outcomes on the hard labels.

**The headline finding:**
- Three point-wise methods (z-score, IF, LOF) all miss the same two anomalies at their optimal configs: NVDA 2024-09-03 and TSLA 2022-04-26. The caught labels in those tickers (+24%, +23%) are large. The missed labels (-10%, -12%) are moderate.
- Pattern: point-wise methods catch only the EXTREMES; they miss MODERATE-magnitude anomalies in volatile stocks.
- This is a class-level limitation, not an algorithm choice. They all measure "is today unusual," not "is the pattern leading to today unusual." The LSTM autoencoder works on sequences, so it has a real shot at closing this gap.

**Per-ticker performance (best LOF config):**
- AAPL: F1=0.300 (3/3 caught) — easiest because anomalies stand out
- GME:  F1=0.211 (2/2 caught)
- AMC:  F1=0.105 (1/1 caught) — perfect recall but only 1 label = low F1 ceiling
- NVDA: F1=0.100 (1/2 caught) — misses 2024-09-03
- TSLA: F1=0.100 (1/2 caught) — misses 2022-04-26

**Design decisions:**
- LOF n_neighbors=20 (sklearn default). Tried n=50 — caught 1 more label but precision halved. Wider neighborhood = more context = catches clustered anomalies but also more false alarms.
- Same per-ticker StandardScaler as IF, same rationale (AAPL ranges vs GME ranges).

**Note on deployment:**
- LOF is transductive — to score a new unseen day, you'd refit on the full training set + the new point. That's a real production limitation compared to IF, which fits once and can score new data instantly. Worth flagging in the eval doc.

**Questions for Day 7 (failure analysis day):**
- Are NVDA 2024-09-03 and TSLA 2022-04-26 the ONLY anomalies all 3 methods agree on missing, or are there others?
- Should I write a dedicated FAILURE_ANALYSIS.md doc tomorrow to capture this systematically?


## Day 7 — May 31, 2026

**What I built:**
- `scripts/analyze_failures.py` — diagnostic script producing 3 tables: per-label catch matrix, per-ticker difficulty, and false-positive overlap.
- `FAILURE_ANALYSIS.md` — structured failure analysis doc cataloging where each method wins and loses, with one universally-hard label identified and complementary strengths documented.

**Key methodological moment:**
- The FP overlap script surfaced 3 dates that all 4 methods flagged but weren't in my ground truth: AMC 2021-06-02, NVDA 2024-02-22, NVDA 2025-04-09. Each had a verified news catalyst, so I added them to the label set (now 13, was 10).
- This is methodologically defensible only because each new label was verified against an independent news source — not "where models agreed." I wrote this distinction into FAILURE_ANALYSIS.md so it's explicit.

**Eval set after refinement:**
- 13 labels across 5 tickers .
- NVDA went from 2 → 4 labels, AMC from 1 → 2. AAPL/GME/TSLA unchanged.
- Years now span 2020-2025 (was 2020-2025 but lighter on 2024-2025).

**Updated leaderboard (13 labels):**
- iforest_c0.01: F1=0.227 (was 0.170 on 10 labels)
- lof_n20_c0.01: F1=0.218 (was 0.163)
- zscore_t3.5: F1=0.179 (was 0.133)
- iqr_k3.0: F1=0.152 (was 0.104)
- iforest_c0.05: still perfect recall (1.0), F1=0.060

Every method's F1 improved because 3 previously-counted false positives became true positives. Relative ordering held (IF still #1, LOF still #2). Reassuring — the additions didn't reshuffle the table.

**Findings I got wrong yesterday:**
- Yesterday I said "z-score, IF, LOF all miss NVDA 2024-09-03 AND TSLA 2022-04-26 — a class-level failure mode." This was wrong on the second label. **Z-score actually catches TSLA 2022-04-26**; only IF and LOF miss it. The truly universal hard case is just NVDA 2024-09-03.
- I implied ML strictly dominates statistical methods. Actually wrong — on TSLA 2022-04-26, statistical wins. On AAPL 2020-03-16 (COVID), ML wins. They're complementary, not ordered. An ensemble of z-score + IF would catch 12/13 labels.

**Per-ticker difficulty (updated):**
- NVDA 0.340 (was 0.129 — biggest jump, because adding 2 labels the methods catch lifted the mean)
- AAPL 0.253 (unchanged)
- GME 0.174 (unchanged)
- AMC 0.169 (was 0.088 — also jumped on added label)
- TSLA 0.095 (unchanged — no labels added; still the hardest ticker)

**Decision I'm leaving on the table:**
- 38 dates remain flagged by 3+ methods but unlabeled. Spot-checks suggest some are real anomalies (e.g. AMC 2024-05-13 Roaring Kitty sympathy rally, TSLA 2019-10-24 earnings). I'm NOT labeling more — moving the goalposts twice looks like fishing. Documented as future work.

**Predictions for LSTM AE (Days 8-9):**
1. NVDA 2024-09-03 — temporal context might distinguish "calm regime → sudden drop" patterns. Maybe catches it.
2. TSLA 2022-04-26 — surrounding context was calm. Sequence-aware method should catch this easily.
3. AAPL 2020-03-16 — already caught by IF/LOF. LSTM should too.

If LSTM catches Sept 3, that's evidence the temporal hypothesis works. If it doesn't, the limitation is deeper than method class.

**Questions for tomorrow (Day 8 starts the LSTM):**
- Sliding window length: 30 days standard, but should I try 20 or 60?
- Architecture: 2-layer LSTM with hidden=16 (per the original plan). If training is slow, drop to hidden=8.
- Train/test split: by date (last 20% as test) or by ticker? Probably by date — each ticker gets its own train history and test period.


## Day 8

**What I built:**
- `src/methods/lstm_ae.py` — LSTM autoencoder (encoder + decoder, hidden=16, 1 layer, batch_first=True).
- Training loop with Adam, MSE loss, early stopping (patience=5), 80/20 temporal train/val split.
- Trained 5 per-ticker models, saved to `models/lstm_ae_{ticker}.pt` (gitignored).
- Per-ticker StandardScaler fit on train only (no val leakage), stats saved alongside weights.

**Total training time:** 35.5s on CPU for all 5 tickers combined.

**Tuning attempt that taught me something:**
- First run with default hidden=16: AAPL converged at val_loss=0.888 (18% reduction from initial).
- Suspected underfitting. Bumped to hidden=32 → AAPL converged at val_loss=0.891 (same).
- Doubled the model, got identical loss. Capacity was not the bottleneck.
- Initially concluded "daily returns are noise-dominated, hitting irreducible floor."
- **Then trained all 5 tickers and was wrong.** Val loss ranges 0.21 (AMC) to 0.89 (AAPL).

**The actual finding:**
- AMC and GME (meme stocks): val MSE 0.21 and 0.38. Highly learnable structure.
- AAPL: val MSE 0.89. Essentially noise floor — efficient pricing, near random-walk returns.
- TSLA, NVDA in between.
- The LSTM learns where the structure is, not where the volatility is.
- This is OPPOSITE intuition — calm stocks (AAPL) have less learnable structure than volatile ones (AMC/GME), because efficient large-caps approach random-walk behavior at daily frequency.

**Prediction for Day 9:**
- LSTM should dominate IF on AMC and GME (clean baseline → anomalies stick out).
- LSTM may underperform IF on AAPL (everything reconstructs poorly → anomaly signal drowns in noise).
- This would invert the classical-methods pattern (where AAPL was the "easiest" ticker).
- If this holds, the project narrative becomes ticker-dependent method selection, not single-best-method.

**Design decisions:**
- hidden=16, 1 layer: small enough to train fast on CPU, ~3K parameters. Capacity test (32) confirmed 16 is sufficient.
- 30-day window: matches the rolling windows used by z-score / IF, apples-to-apples.
- 2 features (return, volume_ratio): skipped volatility_20d to avoid correlation with rolling return variance.
- Temporal split (no shuffling): val period is the most recent 20% of data. Realistic for time series.
- Early stopping with patience=5: stopped between epochs 28-50 across tickers, no overfitting observed.

**Questions for Day 9:**
- What threshold to use? Mean + k*std of train reconstruction errors? Per-window, per-ticker?
- Flag the END date of each high-error window (the day that "broke" the reconstruction)?
- Will my LSTM-vs-classical prediction hold per-ticker?
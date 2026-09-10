# Short-Horizon Price Prediction from Order Flow and Limit-Order-Book Data

## Research question

Can contemporaneously observable order-flow and market-microstructure features predict 1-, 5-, and 10-second mid-price direction, and can that signal produce positive held-out PnL after spread, fees, and slippage?

The answer on the supplied six-day sample is two-part: some specifications show predictive association and positive gross PnL, but **no active strategy is profitable under the realistic-base cost scenario**.

## Assignment summary

This repository is the submission for CMF Data Science Stage 2. It contains:

- a complete audit and EDA of the raw HFT files;
- paper-motivated order-flow and LOB features;
- naive, rule-based, linear, and nonlinear models;
- purged chronological train/validation/test evaluation;
- a delayed-execution backtester with spread, fees, slippage, turnover, sizing, and risk limits;
- cost sensitivity and feature/horizon ablations;
- persisted results, figures, a final report, and a 10-minute presentation script.

## Dataset

| File | Rows | Content |
|---|---:|---|
| `data/lob.csv` | 1,036,690 | 25 ask and 25 bid price/amount levels |
| `data/trades.csv` | 21,864,989 | local timestamp, side, price, amount |

The data cover 2024-08-01 through 2024-08-06 UTC. Timestamps are ordered integer microseconds. There are no missing values, crossed/locked top-of-book observations, or depth-ordering violations. The data do not identify the instrument, venue, exchange timestamp, trade ID, fee tier, or tick-size metadata. The `side` field is treated as aggressor direction.

The audit finds 176,998 potential duplicate trade payloads after excluding the CSV row index. They are retained because simultaneous identical fills cannot be distinguished from duplication without an exchange trade identifier. Full details are in [`reports/data_audit.json`](reports/data_audit.json).

Raw CSVs are intentionally ignored by Git. Place the supplied files at exactly `data/lob.csv` and `data/trades.csv` before reproducing the project.

## Research hypotheses and academic motivation

| Paper | Hypothesis adapted to this dataset | Implementation | Test |
|---|---|---|---|
| Cont, Kukanov & Stoikov, [The Price Impact of Order Book Events](https://doi.org/10.1093/jjfinec/nbt003) | Positive/negative OFI predicts upward/downward short-horizon price moves | Snapshot-based top-of-book OFI, lags, EMA and rolling sums | `ofi_threshold`; OFI versus no-OFI ML ablation |
| Cartea, Donnelly & Jaimungal, [Enhancing Trading Strategies with Order Book Signals](https://doi.org/10.1080/1350486X.2018.1434009) | Depth imbalance and microprice contain short-horizon directional information | 1/5/10-level imbalance, depth, microprice, signed trade flow | Logistic and LightGBM versus OFI-only rule |
| Sirignano & Cont, [Universal Features of Price Formation](https://doi.org/10.1080/14697688.2019.1622295) | Price response to order-flow history can be nonlinear | Lagged/rolling state and gradient-boosted trees | `lightgbm_all` versus linear baselines |

This is an adaptation rather than a claim of exact replication. The files contain snapshots rather than add/cancel event messages, so OFI is reconstructed from changes at the best bid and ask. The stochastic-control market maker and deep neural architecture from the papers are outside the information and sample size supported here.

## Methodology

### Causal one-second bars

A bar labelled `T` contains the last quote and all trades observed in `(T-1s, T]`. Missing quote seconds are forward-filled, and `quote_age_seconds` records staleness. This produces 518,398 decision rows. No future quote or trade enters a feature at `T`.

### Features

- Price state: mid-price, spread, spread in bps, microprice displacement.
- Book state: bid/ask depth and imbalance over 1, 5, and 10 levels.
- Order flow: normalized OFI, 1/5-second lags, 5/20-second EMA, 10/60-second rolling OFI.
- Dynamics: trailing returns, momentum, rolling volatility, depth changes.
- Trades: count, volume, volume change, signed volume, imbalance, intensity, rolling signed flow.
- Context: quote age and cyclic UTC time-of-day.

All rolling, differenced, and lagged features use only the current or preceding observations. The causality test modifies future inputs and asserts that earlier feature rows remain identical.

### Target

For horizon `h`:

```text
future_return[t, h] = log(mid[t+h] / mid[t])
```

The label is down/neutral/up using a fixed ±3 bps band. The threshold is an ex-ante rounded approximation to a realistic-base round trip and is not estimated from the test set. Horizons are 1, 5, and 10 seconds.

### Temporal splitting and leakage prevention

- 60% train, 20% validation, 20% test in chronological order.
- No random shuffle or cross-time resampling.
- A purge at least as long as the target horizon before each boundary.
- An equally conservative embargo after each boundary.
- Imputation and scaling fitted exclusively on train.
- Logistic `C`, OFI threshold, LightGBM parameters, and early stopping selected on validation.
- Test used only for final predictive and trading evaluation.
- A signal created at `T` becomes a position at `T+1`.

See [`src/targets.py`](src/targets.py), [`src/models.py`](src/models.py), and [`tests/test_pipeline.py`](tests/test_pipeline.py).

## Models and baselines

1. `naive_majority`: most frequent training label; in this sample it is the neutral/no-trade benchmark.
2. `ofi_threshold`: validation-selected symmetric threshold on causal smoothed OFI.
3. `logistic_no_ofi`: one-vs-rest logistic regression without the OFI feature family.
4. `logistic_all`: the same linear model with OFI features.
5. `lightgbm_all`: validation-tuned nonlinear tree model with all features.

Macro-F1 is the primary predictive comparison because the 1-second target is 76.8% neutral. Accuracy is retained but is not sufficient by itself.

## Backtesting methodology

Predictions are target positions in `{-1, 0, +1}`, clipped by the configured maximum normalized position. If `s_t` is the signal made at `t`, the position earning the next return is delayed:

```text
position[t] = clip(signal[t-1], -max_position, +max_position)
gross_pnl[t] = position[t] * (mid[t+1] / mid[t] - 1)
turnover[t] = abs(position[t] - position[t-1])
cost[t] = turnover[t] * (half_spread[t] + fee + slippage)
net_pnl[t] = gross_pnl[t] - cost[t]
```

A reversal has turnover 2, and terminal inventory is liquidated. PnL is additive normalized return, not currency PnL. Reported Sharpe is computed from daily PnL and annualized with `sqrt(365)`; the held-out block contains only two partial/calendar days, so Sharpe is descriptive and statistically fragile.

Five cost scenarios are evaluated: zero cost, spread only, realistic low, realistic base, and stressed. The realistic-base assumption is observed half-spread plus 1.0 bps fee and 0.5 bps slippage per traded notional unit.

## Main held-out results

Realistic-base costs, 1-second target:

| Strategy | Accuracy | Macro-F1 | Gross PnL | Net PnL | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| naive majority | 0.7206 | 0.2792 | 0.0000 | 0.0000 | 0 | 0 |
| OFI threshold | 0.4471 | 0.3283 | 0.1966 | -2.4671 | 16,538 | 16,133 |
| logistic, no OFI | 0.6246 | 0.4179 | 0.0965 | -5.4772 | 34,276 | 23,266 |
| logistic, all | 0.6249 | **0.4184** | 0.0936 | -5.4682 | 34,212 | 23,238 |
| LightGBM, all | **0.7214** | 0.2913 | -0.0101 | **-0.1746** | 990 | 982 |

The bold net result is the least-negative active strategy, not a profit. The majority baseline predicts neutral, never trades, and has zero PnL.

The clearest cost result is the 1-second OFI rule: PnL is +0.1966 before costs, +0.0136 after spread only, and -2.4671 under realistic-base fee/slippage. Adding OFI to logistic regression changes macro-F1 by only -0.0008 to +0.0005 across horizons. Thus the data support a weak predictive OFI association, but not an economically viable active strategy under the tested execution assumptions.

Complete tables: [`reports/tables/final_results.md`](reports/tables/final_results.md) and [`reports/results.csv`](reports/results.csv).

## Key figures and interpretations

- [`price_and_spread.png`](reports/figures/price_and_spread.png): the mid-price declines materially across the six-day sample, while spread spikes cluster around the most volatile interval.
- [`distributions.png`](reports/figures/distributions.png): one-second returns are sharply concentrated with heavy tails; spread is usually tight but has rare extreme values.
- [`autocorrelation.png`](reports/figures/autocorrelation.png): raw return dependence is weak, whereas absolute-return dependence is much stronger, consistent with volatility clustering.
- [`target_balance.png`](reports/figures/target_balance.png): the neutral class dominates at 1 second and shrinks as the horizon increases, motivating macro-F1.
- [`test_equity_curves.png`](reports/figures/test_equity_curves.png): realistic costs drive every active strategy below the no-trade benchmark.

Interpretations for every generated figure are collected in [`reports/FIGURE_GUIDE.md`](reports/FIGURE_GUIDE.md).

## Conclusions

- Order-flow and LOB state have measurable short-horizon predictive content.
- Predictive metrics alone overstate economic usefulness.
- OFI barely covers the observed spread at 1 second but not fee/slippage.
- No active model demonstrates positive held-out PnL under realistic-base costs.
- Lower-turnover, confidence-aware execution is the most plausible next research direction.

## Limitations

- Six days and one anonymous instrument do not establish temporal or cross-asset generalization.
- Local timestamps may contain feed latency; exchange timestamps and sequence numbers are absent.
- Trade-side semantics and potential duplicate trades cannot be resolved definitively.
- Fees, queue position, fill probability, partial fills, latency, and nonlinear market impact are unknown.
- One-second aggregation discards sub-second event ordering.
- Overlapping labels reduce effective sample size even with purge/embargo.
- Daily Sharpe is based on only two held-out day buckets.

## Repository structure

```text
README.md                       project overview and reproduction guide
SUBMISSION_AUDIT.md             requirement-by-requirement audit
FINAL_CHECKLIST.md              final handoff and verification
configs/experiment.json         fixed experimental configuration
src/                            data, features, labels, models, backtest, metrics
scripts/train.py                complete reproducible experiment
scripts/evaluate.py             concise result viewer
notebooks/01_eda.ipynb          executable EDA/results reader
reports/final_report.md         full scientific report
reports/FINAL_RESULTS.md        compact final findings
reports/PRESENTATION.md         10-minute presentation material
reports/FIGURE_GUIDE.md         interpretation for each figure
reports/tables/                 slide-ready result and sensitivity tables
reports/figures/                final EDA and evaluation figures
tests/test_pipeline.py          causality, split, and execution checks
```

## Reproduction

Python 3.12 was used for the final run. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py
pytest -q
```

`python scripts/train.py` audits the raw files, builds or loads causal bars, regenerates EDA, trains all models, executes all 75 model/horizon/cost experiments, saves predictions and model metadata, writes final tables, and rebuilds the final report. Use `python scripts/train.py --force-bars` to invalidate the bar cache.

Large reproducible caches, raw data, and binary model objects are Git-ignored. Generated reports, figures, compact tables, configuration, and model-parameter JSON files are intended for submission.

An isolated virtual environment is recommended. The development machine contains unrelated globally installed ML packages with mutually incompatible requirements; they are not imported by this project and are intentionally excluded from `requirements.txt`.

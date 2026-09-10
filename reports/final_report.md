# Order-flow prediction and HFT backtest

## Research question

Can contemporaneously observable order-flow and limit-order-book state predict short-horizon mid-price direction, and does that signal retain positive held-out PnL after spread, fee, and slippage assumptions?

## Dataset and audit

The anonymous dataset contains **1,036,690** 25-level LOB snapshots and **21,864,989** trades from 2024-08-01T00:00:02.038431+00:00 to 2024-08-06T23:59:59.947411+00:00. Timestamps are integer microseconds and both files are chronologically ordered. LOB timestamps are unique; trade timestamps need not be. There is no instrument, venue, exchange timestamp, tick-size metadata, fee schedule, or aggressor-side definition. The `side` field is therefore treated as aggressor direction, an explicit assumption.

Null counts are recorded in `reports/data_audit.json`. The touch has 0 locked/crossed observations and adjacent-level ordering checks found 0 violations. The supplied unnamed column is a unique row identifier, not a feature. No rows were silently deduplicated.

Hashing every payload field except the CSV row-id finds 0 duplicate LOB payload rows and 176,998 potential duplicate trade payload rows. Trades are not deduplicated: simultaneous identical fills may be legitimate, and no exchange trade identifier exists to resolve the ambiguity.

The event streams are converted to 1-second bars. A bar labelled T uses the last quote and trades in (T-1s, T]; absent quote seconds are forward-filled and quote age is retained. This yields 518,398 decision rows. The median quoted spread is 0.11673 bps and 15.80% of one-second returns are zero.

## EDA findings

One-second return lag-1 autocorrelation is 0.037777; absolute-return lag-1 autocorrelation is 0.333551, which measures short volatility clustering. The return ADF p-value is 0 and KPSS p-value is 0.1; the corresponding mid-price p-values are 0.481991 and 0.01. Test p-values are descriptive because microstructure dependence violates ideal iid assumptions and the sample covers only six days.

Figures in `reports/figures/` cover price/spread, distributions, depth, rank correlations, autocorrelation, intraday seasonality, class balance, confusion matrix, and test equity curves. Heavy-tailed plots clip only for display; modeling uses unclipped inputs with median imputation learned on training data.

## Related work and hypotheses

- Cont, Kukanov & Stoikov, [The Price Impact of Order Book Events](https://doi.org/10.1093/jjfinec/nbt003), motivates top-of-book OFI. **H1:** positive (negative) OFI predicts upward (downward) short-horizon moves.
- Cartea, Donnelly & Jaimungal, [Enhancing trading strategies with order book signals](https://doi.org/10.1080/1350486X.2018.1434009), motivates depth imbalance and microprice signals. **H2:** combining imbalance, microprice, spread, and flow improves on an OFI threshold.
- Sirignano & Cont, [Universal features of price formation in financial markets](https://doi.org/10.1080/14697688.2019.1622295), motivates nonlinear LOB models; here a gradient-boosted tree tests nonlinearity without introducing a sequence model before the tabular baseline is sound.

## Methodology

Features include mid, spread, microprice displacement, 1/5/10-level imbalance, depth and changes, Cont-style OFI with explicit lags and causal rolling OFI, lagged returns, momentum, realized volatility, trade intensity/signed flow and volume change, staleness, and cyclic UTC time. Every rolling feature is backward-looking and includes no negative shift. Targets are future log-mid returns over [1, 5, 10] seconds, mapped to down/neutral/up using ±3.0 bps. The 3 bps default is an ex-ante rounded approximation to a base-cost round trip (two executions, each paying fee, slippage, and half-spread), rather than a threshold estimated from the test set.

The chronological split is 60%/20%/20%. At each boundary, a gap at least as large as the target horizon is purged on the left and embargoed on the right. Imputation/scaling fit only on training data; standardized logistic inputs are capped at ±20 to bound regime-shift outlier influence without learning test quantiles. Hyperparameters and the OFI threshold use validation only; the test block is touched once for reported metrics.

Models are a training-majority baseline, validation-tuned OFI threshold, one-vs-rest logistic regression without OFI, logistic regression with OFI, and LightGBM with all features. This directly supplies the requested naive, rule-based, linear, nonlinear, and OFI ablation comparisons.

## Backtesting

A signal computed at T is executed at T+1 and earns returns only from T+1 onward. Desired positions are clipped to 1.0 normalized notional unit, the explicit sizing/risk limit. Turnover pays the contemporaneous half-spread per traded unit plus scenario-specific fee and adverse slippage; reversal costs two units and final inventory is liquidated. PnL is normalized return on capital, not currency PnL. Daily Sharpe is annualized with sqrt(365), appropriate to a presumed continuously traded asset, but each held-out run contains only the `sharpe_daily_observations` shown in the table, so it is descriptive rather than reliable.

## Held-out results (realistic base cost)

The base assumption is spread crossing + 1.0 bps fee + 0.5 bps slippage per traded notional unit.

|   horizon | model           |   accuracy |   macro_f1 |   net_pnl |   sharpe_daily_annualized |   sharpe_daily_observations |   max_drawdown |   turnover |   number_of_trades |
|----------:|:----------------|-----------:|-----------:|----------:|--------------------------:|----------------------------:|---------------:|-----------:|-------------------:|
|         1 | lightgbm_all    |    0.72143 |    0.29129 |  -0.17457 |                   -20.349 |                           2 |       -0.17457 |        990 |                982 |
|         1 | logistic_all    |    0.62487 |    0.41837 |  -5.4682  |                   -28.191 |                           2 |       -5.4682  |      34212 |              23238 |
|         1 | logistic_no_ofi |    0.62455 |    0.41787 |  -5.4772  |                   -28.215 |                           2 |       -5.4772  |      34276 |              23266 |
|         1 | naive_majority  |    0.72059 |    0.2792  |   0       |                     0     |                           2 |        0       |          0 |                  0 |
|         1 | ofi_threshold   |    0.44705 |    0.32829 |  -2.4671  |                   -20.624 |                           2 |       -2.4672  |      16538 |              16133 |
|         5 | lightgbm_all    |    0.40876 |    0.35189 |  -4.651   |                   -24.617 |                           2 |       -4.6521  |      29708 |              20341 |
|         5 | logistic_all    |    0.39141 |    0.39075 | -10.838   |                   -21.854 |                           2 |      -10.839   |      68598 |              39710 |
|         5 | logistic_no_ofi |    0.39134 |    0.39064 | -10.854   |                   -21.782 |                           2 |      -10.854   |      68852 |              39824 |
|         5 | naive_majority  |    0.37769 |    0.18277 |   0       |                     0     |                           2 |        0       |          0 |                  0 |
|         5 | ofi_threshold   |    0.3443  |    0.34333 |  -2.9578  |                   -20.332 |                           2 |       -2.9584  |      20058 |              19121 |
|        10 | lightgbm_all    |    0.38943 |    0.36522 |  -7.9094  |                   -18.884 |                           2 |       -7.9104  |      51296 |              30087 |
|        10 | logistic_all    |    0.39317 |    0.37251 | -11.545   |                   -20.651 |                           2 |      -11.545   |      74172 |              40510 |
|        10 | logistic_no_ofi |    0.39418 |    0.37329 | -11.711   |                   -20.559 |                           2 |      -11.711   |      75232 |              41046 |
|        10 | naive_majority  |    0.26536 |    0.13981 |   0       |                     0     |                           2 |        0       |          0 |                  0 |
|        10 | ofi_threshold   |    0.34044 |    0.33906 |  -2.9574  |                   -20.333 |                           2 |       -2.9584  |      20058 |              19121 |

The best base-cost held-out result is **naive_majority** at horizon 1s with net PnL 0; it is **non-positive**. The best non-naive strategy is **lightgbm_all** at 1s with net PnL -0.174572. The winning majority baseline predicts the neutral class, holds no position, and is therefore the no-trade benchmark—not evidence of a predictive edge. These statements are mechanical and do not generalize beyond the supplied test period.

## Ablations and cost sensitivity

`logistic_no_ofi` versus `logistic_all` isolates the OFI feature family; `ofi_threshold` tests H1 directly; the three horizons test decay. The `zero_cost` rows are an upper bound, while spread-only, low, base, and stressed scenarios expose turnover sensitivity. Mean net PnL across all model/horizon combinations is:

| cost_scenario   |   fee_bps |   slippage_bps | include_spread   |   net_pnl |
|:----------------|----------:|---------------:|:-----------------|----------:|
| realistic_base  |       1   |           0.5  | True             |  -5.1341  |
| realistic_low   |       0.5 |           0.25 | True             |  -2.6642  |
| spread_only     |       0   |           0    | True             |  -0.19421 |
| stressed        |       2   |           1    | True             | -10.074   |
| zero_cost       |       0   |           0    | False            |   0.1765  |

Zero-cost mean net PnL is 0.176501. Full per-run prediction, trading, class, validation, and cost outputs are in `reports/results.csv`.

At 1s, the OFI threshold earns 0.19658 before costs and 0.0136281 after quoted spread alone, but -2.46707 under base fees/slippage. This supports a weak H1 association while rejecting economic viability under the base implementation. Adding OFI to logistic regression changes held-out macro-F1 by only -0.000784193 to 0.00049149 across horizons and does not make any logistic strategy profitable after base costs; H2 is therefore not supported in economically meaningful form here.

## Limitations

- Six days and one unnamed instrument are too narrow for claims of stable profitability or cross-asset generalization.
- Local timestamps may include feed/network latency; exchange timestamps and sequence numbers are absent.
- Trade-side semantics, fees, tick size, queue position, available displayed liquidity, latency, and market impact are not documented.
- The one-second aggregation discards sub-second ordering. Market orders are approximated by next-bar spread crossing and bps slippage, without partial fills.
- Overlapping horizon labels and serial dependence reduce effective sample size despite purging.
- Hyperparameter search is deliberately small; repeated validation choices can still overfit this short sample.

## Conclusion

The experiment tests predictive association and executable economics separately. Classification above a naive baseline is not itself evidence of profitability: the decisive result is the held-out, delayed-execution, cost-aware PnL above. Given the short sample and execution omissions, any positive result should be treated as a hypothesis for evaluation on additional dates and venues; any non-positive result rejects profitability only for the tested implementation and assumptions.

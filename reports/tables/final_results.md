# Final held-out results

All rows below use the untouched chronological test block. PnL is normalized additive return. The primary table uses the realistic-base scenario: observed half-spread + 1.0 bps fee + 0.5 bps slippage per traded notional unit.

## Slide-ready primary comparison — 1-second horizon

| strategy        |   macro_f1 |   gross_pnl |   net_pnl |   turnover |
|:----------------|-----------:|------------:|----------:|-----------:|
| naive_majority  |    0.2792  |    0        |   0       |          0 |
| ofi_threshold   |    0.32829 |    0.19658  |  -2.4671  |      16538 |
| logistic_no_ofi |    0.41787 |    0.096475 |  -5.4772  |      34276 |
| logistic_all    |    0.41837 |    0.093596 |  -5.4682  |      34212 |
| lightgbm_all    |    0.29129 |   -0.010056 |  -0.17457 |        990 |

## Detailed primary comparison

|   horizon_seconds | strategy        |   accuracy |   macro_f1 |   gross_pnl |   net_pnl |   sharpe |   max_drawdown |   turnover |   trades |
|------------------:|:----------------|-----------:|-----------:|------------:|----------:|---------:|---------------:|-----------:|---------:|
|                 1 | naive_majority  |    0.72059 |    0.2792  |    0        |   0       |    0     |        0       |          0 |        0 |
|                 1 | ofi_threshold   |    0.44705 |    0.32829 |    0.19658  |  -2.4671  |  -20.624 |       -2.4672  |      16538 |    16133 |
|                 1 | logistic_no_ofi |    0.62455 |    0.41787 |    0.096475 |  -5.4772  |  -28.215 |       -5.4772  |      34276 |    23266 |
|                 1 | logistic_all    |    0.62487 |    0.41837 |    0.093596 |  -5.4682  |  -28.191 |       -5.4682  |      34212 |    23238 |
|                 1 | lightgbm_all    |    0.72143 |    0.29129 |   -0.010056 |  -0.17457 |  -20.349 |       -0.17457 |        990 |      982 |

## All horizons — realistic-base costs

|   horizon_seconds | strategy        |   accuracy |   macro_f1 |   gross_pnl |   net_pnl |   sharpe |   max_drawdown |   turnover |   trades |
|------------------:|:----------------|-----------:|-----------:|------------:|----------:|---------:|---------------:|-----------:|---------:|
|                 1 | naive_majority  |    0.72059 |    0.2792  |    0        |   0       |    0     |        0       |          0 |        0 |
|                 1 | ofi_threshold   |    0.44705 |    0.32829 |    0.19658  |  -2.4671  |  -20.624 |       -2.4672  |      16538 |    16133 |
|                 1 | logistic_no_ofi |    0.62455 |    0.41787 |    0.096475 |  -5.4772  |  -28.215 |       -5.4772  |      34276 |    23266 |
|                 1 | logistic_all    |    0.62487 |    0.41837 |    0.093596 |  -5.4682  |  -28.191 |       -5.4682  |      34212 |    23238 |
|                 1 | lightgbm_all    |    0.72143 |    0.29129 |   -0.010056 |  -0.17457 |  -20.349 |       -0.17457 |        990 |      982 |
|                 5 | naive_majority  |    0.37769 |    0.18277 |    0        |   0       |    0     |        0       |          0 |        0 |
|                 5 | ofi_threshold   |    0.3443  |    0.34333 |    0.26982  |  -2.9578  |  -20.332 |       -2.9584  |      20058 |    19121 |
|                 5 | logistic_no_ofi |    0.39134 |    0.39064 |    0.24556  | -10.854   |  -21.782 |      -10.854   |      68852 |    39824 |
|                 5 | logistic_all    |    0.39141 |    0.39075 |    0.21878  | -10.838   |  -21.854 |      -10.839   |      68598 |    39710 |
|                 5 | lightgbm_all    |    0.40876 |    0.35189 |    0.16309  |  -4.651   |  -24.617 |       -4.6521  |      29708 |    20341 |
|                10 | naive_majority  |    0.26536 |    0.13981 |    0        |   0       |    0     |        0       |          0 |        0 |
|                10 | ofi_threshold   |    0.34044 |    0.33906 |    0.27019  |  -2.9574  |  -20.333 |       -2.9584  |      20058 |    19121 |
|                10 | logistic_no_ofi |    0.39418 |    0.37329 |    0.39169  | -11.711   |  -20.559 |      -11.711   |      75232 |    41046 |
|                10 | logistic_all    |    0.39317 |    0.37251 |    0.38994  | -11.545   |  -20.651 |      -11.545   |      74172 |    40510 |
|                10 | lightgbm_all    |    0.38943 |    0.36522 |    0.32184  |  -7.9094  |  -18.884 |       -7.9104  |      51296 |    30087 |

## Cost sensitivity — 1-second horizon

| strategy        |   realistic_base |   realistic_low |   spread_only |   stressed |   zero_cost |
|:----------------|-----------------:|----------------:|--------------:|-----------:|------------:|
| naive_majority  |          0       |         0       |      0        |    0       |    0        |
| ofi_threshold   |         -2.4671  |        -1.2267  |      0.013628 |   -4.9478  |    0.19658  |
| logistic_no_ofi |         -5.4772  |        -2.9065  |     -0.33584  |  -10.619   |    0.096475 |
| logistic_all    |         -5.4682  |        -2.9023  |     -0.33644  |  -10.6     |    0.093596 |
| lightgbm_all    |         -0.17457 |        -0.10032 |     -0.026072 |   -0.32307 |   -0.010056 |

## Conclusions

1. The highest test macro-F1 is **0.4184** from `logistic_all` at 1s; accuracy alone is misleading because the neutral class dominates at 1s.
2. Every active strategy has negative net PnL under realistic-base costs. The least-negative active result is `lightgbm_all` at 1s: **-0.1746**.
3. The 1s OFI rule earns **0.1966** before costs and **0.0136** after spread only, but falls to **-2.4671** under base fee/slippage.
4. Adding the OFI family to logistic regression changes macro-F1 by only **-0.0008 to 0.0005** across horizons; this ablation shows no material predictive improvement.
5. The neutral majority baseline has zero turnover and zero PnL. It is a no-trade benchmark, not a profitable strategy.

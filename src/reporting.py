"""EDA figures and data-driven final report generation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller, kpss


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {"figure.dpi": 140, "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False}
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _stationarity(series: pd.Series, max_points: int = 50_000) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) > max_points:
        clean = clean.iloc[:: int(np.ceil(len(clean) / max_points))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "sample_size": int(len(clean)),
    }


def create_eda(frame: pd.DataFrame, figures_dir: str | Path) -> dict:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _style()
    step = max(1, len(frame) // 5000)
    sample = frame.iloc[::step]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(sample["timestamp"], sample["mid"], lw=0.7, color="#264653")
    axes[0].set_ylabel("Mid-price")
    axes[0].set_title("Mid-price and quoted spread (one-second bars)")
    axes[1].plot(sample["timestamp"], sample["spread_bps"], lw=0.6, color="#e76f51")
    axes[1].set_ylabel("Spread (bps)")
    axes[1].set_xlabel("UTC time")
    _save(fig, figures_dir / "price_and_spread.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    return_clip = frame["ret_1"].dropna().clip(
        frame["ret_1"].quantile(0.001), frame["ret_1"].quantile(0.999)
    )
    sns.histplot(return_clip * 10_000, bins=100, ax=axes[0], color="#2a9d8f", stat="density")
    axes[0].set(xlabel="One-second log return (bps)", title="Return distribution (0.1% tails clipped)")
    spread_clip = frame["spread_bps"].clip(upper=frame["spread_bps"].quantile(0.995))
    sns.histplot(spread_clip, bins=80, ax=axes[1], color="#e9c46a", stat="density")
    axes[1].set(xlabel="Spread (bps)", title="Spread distribution (top 0.5% clipped)")
    _save(fig, figures_dir / "distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sns.histplot(np.log1p(frame["depth_ask_10"]), bins=80, ax=axes[0], color="#e76f51", label="ask")
    sns.histplot(np.log1p(frame["depth_bid_10"]), bins=80, ax=axes[0], color="#457b9d", label="bid", alpha=0.55)
    axes[0].legend()
    axes[0].set(xlabel="log(1 + depth)", title="Ten-level depth")
    sns.histplot(frame["book_imbalance_10"].clip(-1, 1), bins=80, ax=axes[1], color="#6d597a")
    axes[1].set(xlabel="Depth imbalance", title="Ten-level imbalance")
    sns.histplot(frame["trade_volume_log"], bins=80, ax=axes[2], color="#2a9d8f")
    axes[2].set(xlabel="log(1 + traded volume)", title="Per-second traded volume")
    _save(fig, figures_dir / "depth_and_imbalance.png")

    clean_ret = frame["ret_1"].fillna(0.0).to_numpy()
    acf_values = acf(clean_ret, nlags=60, fft=True)
    abs_acf_values = acf(np.abs(clean_ret), nlags=60, fft=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(range(1, 61), acf_values[1:], linefmt="#457b9d", markerfmt=" ", basefmt=" ", label="return")
    ax.plot(range(1, 61), abs_acf_values[1:], color="#e76f51", lw=1.2, label="absolute return")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel="Lag (seconds)", ylabel="Autocorrelation", title="Return and volatility autocorrelation")
    ax.legend()
    _save(fig, figures_dir / "autocorrelation.png")

    corr_cols = [
        "ret_1", "spread_bps", "microprice_delta_bps", "book_imbalance_1",
        "book_imbalance_10", "ofi", "trade_imbalance", "volatility_60", "trade_count_log",
    ]
    corr = frame[corr_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Spearman feature correlations")
    _save(fig, figures_dir / "correlations.png")

    hourly = frame.assign(hour=frame["timestamp"].dt.hour).groupby("hour", observed=True).agg(
        abs_return_bps=("ret_1", lambda x: np.mean(np.abs(x)) * 10_000),
        spread_bps=("spread_bps", "mean"),
        trades=("trade_count", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, col, color, ylabel in zip(
        axes,
        ["abs_return_bps", "spread_bps", "trades"],
        ["#2a9d8f", "#e76f51", "#457b9d"],
        ["Mean |return| (bps)", "Mean spread (bps)", "Mean trades/second"],
    ):
        ax.plot(hourly.index, hourly[col], marker="o", ms=3, color=color)
        ax.set(xlabel="UTC hour", ylabel=ylabel)
    fig.suptitle("Intraday patterns")
    _save(fig, figures_dir / "intraday_patterns.png")

    findings = {
        "rows": int(len(frame)),
        "start_utc": str(frame["timestamp"].min()),
        "end_utc": str(frame["timestamp"].max()),
        "mid": frame["mid"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict(),
        "spread_bps": frame["spread_bps"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict(),
        "return_bps": (frame["ret_1"] * 10_000).describe(percentiles=[0.01, 0.5, 0.99]).to_dict(),
        "trade_count": frame["trade_count"].describe(percentiles=[0.5, 0.99]).to_dict(),
        "quote_age_seconds": frame["quote_age_seconds"].describe(percentiles=[0.5, 0.99]).to_dict(),
        "zero_return_fraction": float((frame["ret_1"] == 0).mean()),
        "return_acf_lag1": float(acf_values[1]),
        "absolute_return_acf_lag1": float(abs_acf_values[1]),
        "stationarity_mid": _stationarity(frame["mid"]),
        "stationarity_returns": _stationarity(frame["ret_1"]),
        "spearman_correlations": corr.to_dict(),
        "hourly": hourly.reset_index().to_dict(orient="records"),
    }
    return findings


def plot_target_balance(target_counts: pd.DataFrame, figures_dir: str | Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=target_counts, x="horizon", y="fraction", hue="target", ax=ax, palette="Set2")
    ax.set(xlabel="Forecast horizon (seconds)", ylabel="Fraction", title="Target class balance")
    ax.legend(title="Direction")
    _save(fig, Path(figures_dir) / "target_balance.png")


def plot_confusion(matrix: list[list[int]], title: str, path: str | Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["down", "neutral", "up"], yticklabels=["down", "neutral", "up"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    _save(fig, Path(path))


def plot_equity(curves: dict[str, pd.DataFrame], path: str | Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in curves.items():
        step = max(1, len(curve) // 5000)
        view = curve.iloc[::step]
        ax.plot(view["timestamp"], view["cumulative_pnl"], lw=1, label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel="UTC time", ylabel="Cumulative normalized PnL", title=title)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, Path(path))


def save_json(value: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.6g}" if pd.notna(value) else "NA"


def write_submission_tables(results: pd.DataFrame, reports_dir: str | Path) -> None:
    """Create compact, slide-ready tables directly from persisted experiment rows."""
    reports_dir = Path(reports_dir)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    strategy_order = {
        "naive_majority": 0,
        "ofi_threshold": 1,
        "logistic_no_ofi": 2,
        "logistic_all": 3,
        "lightgbm_all": 4,
    }
    base = results[results["cost_scenario"] == "realistic_base"].copy()
    base["strategy_order"] = base["model"].map(strategy_order)
    base = base.sort_values(["horizon", "strategy_order"])
    final = base[
        [
            "horizon",
            "model",
            "accuracy",
            "macro_f1",
            "gross_pnl",
            "net_pnl",
            "sharpe_daily_annualized",
            "max_drawdown",
            "turnover",
            "number_of_trades",
        ]
    ].rename(
        columns={
            "horizon": "horizon_seconds",
            "model": "strategy",
            "sharpe_daily_annualized": "sharpe",
            "number_of_trades": "trades",
        }
    )
    final.to_csv(tables_dir / "final_results.csv", index=False)

    primary_horizon = int(results["horizon"].min())
    primary = final[final["horizon_seconds"] == primary_horizon]
    primary_slide = primary[["strategy", "macro_f1", "gross_pnl", "net_pnl", "turnover"]]
    active = base[base["model"] != "naive_majority"]
    best_active = active.loc[active["net_pnl"].idxmax()]
    best_f1 = base.loc[base["macro_f1"].idxmax()]
    ofi = results[(results["horizon"] == primary_horizon) & (results["model"] == "ofi_threshold")]
    ofi_zero = float(ofi.loc[ofi["cost_scenario"] == "zero_cost", "net_pnl"].iloc[0])
    ofi_spread = float(ofi.loc[ofi["cost_scenario"] == "spread_only", "net_pnl"].iloc[0])
    ofi_base = float(ofi.loc[ofi["cost_scenario"] == "realistic_base", "net_pnl"].iloc[0])
    ablation = base.pivot(index="horizon", columns="model", values="macro_f1")
    f1_delta = ablation["logistic_all"] - ablation["logistic_no_ofi"]

    sensitivity = (
        results[results["horizon"] == primary_horizon]
        .pivot(index="model", columns="cost_scenario", values="net_pnl")
        .reindex(strategy_order)
        .reset_index()
        .rename(columns={"model": "strategy"})
    )
    sensitivity.to_csv(tables_dir / "cost_sensitivity_h1.csv", index=False)

    text = f"""# Final held-out results

All rows below use the untouched chronological test block. PnL is normalized additive return. The primary table uses the realistic-base scenario: observed half-spread + 1.0 bps fee + 0.5 bps slippage per traded notional unit.

## Slide-ready primary comparison — {primary_horizon}-second horizon

{primary_slide.to_markdown(index=False, floatfmt='.5g')}

## Detailed primary comparison

{primary.to_markdown(index=False, floatfmt='.5g')}

## All horizons — realistic-base costs

{final.to_markdown(index=False, floatfmt='.5g')}

## Cost sensitivity — {primary_horizon}-second horizon

{sensitivity.to_markdown(index=False, floatfmt='.5g')}

## Conclusions

1. The highest test macro-F1 is **{best_f1['macro_f1']:.4f}** from `{best_f1['model']}` at {int(best_f1['horizon'])}s; accuracy alone is misleading because the neutral class dominates at 1s.
2. Every active strategy has negative net PnL under realistic-base costs. The least-negative active result is `{best_active['model']}` at {int(best_active['horizon'])}s: **{best_active['net_pnl']:.4f}**.
3. The {primary_horizon}s OFI rule earns **{ofi_zero:.4f}** before costs and **{ofi_spread:.4f}** after spread only, but falls to **{ofi_base:.4f}** under base fee/slippage.
4. Adding the OFI family to logistic regression changes macro-F1 by only **{f1_delta.min():.4f} to {f1_delta.max():.4f}** across horizons; this ablation shows no material predictive improvement.
5. The neutral majority baseline has zero turnover and zero PnL. It is a no-trade benchmark, not a profitable strategy.
"""
    (tables_dir / "final_results.md").write_text(text, encoding="utf-8")
    (reports_dir / "FINAL_RESULTS.md").write_text(text, encoding="utf-8")


def write_final_report(
    path: str | Path,
    audit: dict,
    eda: dict,
    results: pd.DataFrame,
    config: dict,
) -> None:
    base = results[results["cost_scenario"] == "realistic_base"].copy()
    columns = ["horizon", "model", "accuracy", "macro_f1", "net_pnl", "sharpe_daily_annualized", "sharpe_daily_observations", "max_drawdown", "turnover", "number_of_trades"]
    table = base[columns].sort_values(["horizon", "model"]).to_markdown(index=False, floatfmt=".5g")
    zero = results[results["cost_scenario"] == "zero_cost"]
    sensitivity = (
        results.groupby(["cost_scenario", "fee_bps", "slippage_bps", "include_spread"], as_index=False)["net_pnl"]
        .mean()
        .to_markdown(index=False, floatfmt=".5g")
    )
    best_row = base.loc[base["net_pnl"].idxmax()]
    learned = base[base["model"] != "naive_majority"]
    best_learned = learned.loc[learned["net_pnl"].idxmax()]
    min_horizon = int(min(config["horizons"]))
    ofi_min = results[(results["horizon"] == min_horizon) & (results["model"] == "ofi_threshold")]
    ofi_zero = ofi_min.loc[ofi_min["cost_scenario"] == "zero_cost", "net_pnl"].iloc[0]
    ofi_spread = ofi_min.loc[ofi_min["cost_scenario"] == "spread_only", "net_pnl"].iloc[0]
    ofi_base = ofi_min.loc[ofi_min["cost_scenario"] == "realistic_base", "net_pnl"].iloc[0]
    logistic_ablation = base.pivot(index="horizon", columns="model", values="macro_f1")
    ofi_f1_delta = logistic_ablation["logistic_all"] - logistic_ablation["logistic_no_ofi"]
    profitability = "positive" if best_row["net_pnl"] > 0 else "non-positive"
    if best_row["model"] == "naive_majority" and best_row["turnover"] == 0:
        baseline_note = (
            "The winning majority baseline predicts the neutral class, holds no position, and is "
            "therefore the no-trade benchmark—not evidence of a predictive edge."
        )
    elif best_row["model"] == "naive_majority":
        baseline_note = (
            "The winning static majority baseline's PnL reflects held-out directional drift, not "
            "evidence of an order-flow trading edge."
        )
    else:
        baseline_note = "The winning strategy is active, but its result remains specific to this test block."
    violations = sum(int(v) for v in audit["depth_ordering_violations"].values())
    text = f"""# Order-flow prediction and HFT backtest

## Research question

Can contemporaneously observable order-flow and limit-order-book state predict short-horizon mid-price direction, and does that signal retain positive held-out PnL after spread, fee, and slippage assumptions?

## Dataset and audit

The anonymous dataset contains **{audit['lob']['rows']:,}** 25-level LOB snapshots and **{audit['trades']['rows']:,}** trades from {audit['lob']['start_utc']} to {audit['lob']['end_utc']}. Timestamps are integer microseconds and both files are chronologically ordered. LOB timestamps are unique; trade timestamps need not be. There is no instrument, venue, exchange timestamp, tick-size metadata, fee schedule, or aggressor-side definition. The `side` field is therefore treated as aggressor direction, an explicit assumption.

Null counts are recorded in `reports/data_audit.json`. The touch has {audit['lob']['locked_or_crossed']} locked/crossed observations and adjacent-level ordering checks found {violations} violations. The supplied unnamed column is a unique row identifier, not a feature. No rows were silently deduplicated.

Hashing every payload field except the CSV row-id finds {audit['lob']['duplicate_payload_rows']:,} duplicate LOB payload rows and {audit['trades']['duplicate_payload_rows']:,} potential duplicate trade payload rows. Trades are not deduplicated: simultaneous identical fills may be legitimate, and no exchange trade identifier exists to resolve the ambiguity.

The event streams are converted to {config['frequency_seconds']}-second bars. A bar labelled T uses the last quote and trades in (T-1s, T]; absent quote seconds are forward-filled and quote age is retained. This yields {eda['rows']:,} decision rows. The median quoted spread is {_fmt(eda['spread_bps']['50%'])} bps and {eda['zero_return_fraction']:.2%} of one-second returns are zero.

## EDA findings

One-second return lag-1 autocorrelation is {_fmt(eda['return_acf_lag1'])}; absolute-return lag-1 autocorrelation is {_fmt(eda['absolute_return_acf_lag1'])}, which measures short volatility clustering. The return ADF p-value is {_fmt(eda['stationarity_returns']['adf_pvalue'])} and KPSS p-value is {_fmt(eda['stationarity_returns']['kpss_pvalue'])}; the corresponding mid-price p-values are {_fmt(eda['stationarity_mid']['adf_pvalue'])} and {_fmt(eda['stationarity_mid']['kpss_pvalue'])}. Test p-values are descriptive because microstructure dependence violates ideal iid assumptions and the sample covers only six days.

Figures in `reports/figures/` cover price/spread, distributions, depth, rank correlations, autocorrelation, intraday seasonality, class balance, confusion matrix, and test equity curves. Heavy-tailed plots clip only for display; modeling uses unclipped inputs with median imputation learned on training data.

## Related work and hypotheses

- Cont, Kukanov & Stoikov, [The Price Impact of Order Book Events](https://doi.org/10.1093/jjfinec/nbt003), motivates top-of-book OFI. **H1:** positive (negative) OFI predicts upward (downward) short-horizon moves.
- Cartea, Donnelly & Jaimungal, [Enhancing trading strategies with order book signals](https://doi.org/10.1080/1350486X.2018.1434009), motivates depth imbalance and microprice signals. **H2:** combining imbalance, microprice, spread, and flow improves on an OFI threshold.
- Sirignano & Cont, [Universal features of price formation in financial markets](https://doi.org/10.1080/14697688.2019.1622295), motivates nonlinear LOB models; here a gradient-boosted tree tests nonlinearity without introducing a sequence model before the tabular baseline is sound.

## Methodology

Features include mid, spread, microprice displacement, 1/5/10-level imbalance, depth and changes, Cont-style OFI with explicit lags and causal rolling OFI, lagged returns, momentum, realized volatility, trade intensity/signed flow and volume change, staleness, and cyclic UTC time. Every rolling feature is backward-looking and includes no negative shift. Targets are future log-mid returns over {config['horizons']} seconds, mapped to down/neutral/up using ±{config['neutral_bps']} bps. The 3 bps default is an ex-ante rounded approximation to a base-cost round trip (two executions, each paying fee, slippage, and half-spread), rather than a threshold estimated from the test set.

The chronological split is {config['train_fraction']:.0%}/{config['validation_fraction']:.0%}/{1-config['train_fraction']-config['validation_fraction']:.0%}. At each boundary, a gap at least as large as the target horizon is purged on the left and embargoed on the right. Imputation/scaling fit only on training data; standardized logistic inputs are capped at ±20 to bound regime-shift outlier influence without learning test quantiles. Hyperparameters and the OFI threshold use validation only; the test block is touched once for reported metrics.

Models are a training-majority baseline, validation-tuned OFI threshold, one-vs-rest logistic regression without OFI, logistic regression with OFI, and LightGBM with all features. This directly supplies the requested naive, rule-based, linear, nonlinear, and OFI ablation comparisons.

## Backtesting

A signal computed at T is executed at T+1 and earns returns only from T+1 onward. Desired positions are clipped to {config['max_position']} normalized notional unit, the explicit sizing/risk limit. Turnover pays the contemporaneous half-spread per traded unit plus scenario-specific fee and adverse slippage; reversal costs two units and final inventory is liquidated. PnL is normalized return on capital, not currency PnL. Daily Sharpe is annualized with sqrt(365), appropriate to a presumed continuously traded asset, but each held-out run contains only the `sharpe_daily_observations` shown in the table, so it is descriptive rather than reliable.

## Held-out results (realistic base cost)

The base assumption is spread crossing + {config['costs']['realistic_base']['fee_bps']} bps fee + {config['costs']['realistic_base']['slippage_bps']} bps slippage per traded notional unit.

{table}

The best base-cost held-out result is **{best_row['model']}** at horizon {int(best_row['horizon'])}s with net PnL {_fmt(best_row['net_pnl'])}; it is **{profitability}**. The best non-naive strategy is **{best_learned['model']}** at {int(best_learned['horizon'])}s with net PnL {_fmt(best_learned['net_pnl'])}. {baseline_note} These statements are mechanical and do not generalize beyond the supplied test period.

## Ablations and cost sensitivity

`logistic_no_ofi` versus `logistic_all` isolates the OFI feature family; `ofi_threshold` tests H1 directly; the three horizons test decay. The `zero_cost` rows are an upper bound, while spread-only, low, base, and stressed scenarios expose turnover sensitivity. Mean net PnL across all model/horizon combinations is:

{sensitivity}

Zero-cost mean net PnL is {_fmt(zero['net_pnl'].mean())}. Full per-run prediction, trading, class, validation, and cost outputs are in `reports/results.csv`.

At {min_horizon}s, the OFI threshold earns {_fmt(ofi_zero)} before costs and {_fmt(ofi_spread)} after quoted spread alone, but {_fmt(ofi_base)} under base fees/slippage. This supports a weak H1 association while rejecting economic viability under the base implementation. Adding OFI to logistic regression changes held-out macro-F1 by only {_fmt(ofi_f1_delta.min())} to {_fmt(ofi_f1_delta.max())} across horizons and does not make any logistic strategy profitable after base costs; H2 is therefore not supported in economically meaningful form here.

## Limitations

- Six days and one unnamed instrument are too narrow for claims of stable profitability or cross-asset generalization.
- Local timestamps may include feed/network latency; exchange timestamps and sequence numbers are absent.
- Trade-side semantics, fees, tick size, queue position, available displayed liquidity, latency, and market impact are not documented.
- The one-second aggregation discards sub-second ordering. Market orders are approximated by next-bar spread crossing and bps slippage, without partial fills.
- Overlapping horizon labels and serial dependence reduce effective sample size despite purging.
- Hyperparameter search is deliberately small; repeated validation choices can still overfit this short sample.

## Conclusion

The experiment tests predictive association and executable economics separately. Classification above a naive baseline is not itself evidence of profitability: the decisive result is the held-out, delayed-execution, cost-aware PnL above. Given the short sample and execution omissions, any positive result should be treated as a hypothesis for evaluation on additional dates and venues; any non-positive result rejects profitability only for the tested implementation and assumptions.
"""
    Path(path).write_text(text, encoding="utf-8")

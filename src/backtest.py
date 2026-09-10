"""Causal vectorized backtester with delayed execution and explicit costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    frame: pd.DataFrame,
    signal: np.ndarray | pd.Series,
    fee_bps: float = 0.5,
    slippage_bps: float = 0.25,
    include_spread: bool = True,
    max_position: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Backtest target positions in normalized notional units.

    A signal formed on row t becomes a position on row t+1. That position earns
    the mid-price return from t+1 to t+2. Position changes pay the contemporaneous
    half-spread for each unit traded, plus fee and adverse slippage.
    """
    df = frame[["timestamp", "mid", "spread_bps"]].copy().reset_index(drop=True)
    desired = np.clip(np.asarray(signal, dtype=float), -max_position, max_position)
    position = pd.Series(desired).shift(1).fillna(0.0).to_numpy()
    forward_return = df["mid"].shift(-1) / df["mid"] - 1.0
    turnover = np.abs(np.diff(position, prepend=0.0))
    # Explicit final liquidation; it incurs costs without earning another return.
    if len(turnover):
        turnover[-1] += abs(position[-1])
    spread_cost = turnover * df["spread_bps"].to_numpy() / 20_000.0 if include_spread else 0.0
    explicit_cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    gross = position * forward_return.fillna(0.0).to_numpy()
    net = gross - spread_cost - explicit_cost
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    result = df.assign(
        signal=desired,
        position=position,
        turnover=turnover,
        gross_pnl=gross,
        cost=np.asarray(spread_cost) + explicit_cost,
        net_pnl=net,
        cumulative_pnl=equity,
        drawdown=drawdown,
    )
    daily = result.set_index("timestamp")["net_pnl"].resample("1D").sum()
    daily_std = daily.std(ddof=1)
    sharpe = np.sqrt(365.0) * daily.mean() / daily_std if daily_std and np.isfinite(daily_std) else 0.0
    executions = int(np.count_nonzero(turnover))
    active = position != 0
    interval_hit_rate = float(np.mean(gross[active] > 0)) if active.any() else 0.0
    metrics = {
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "sharpe_daily_annualized": float(sharpe),
        "sharpe_daily_observations": int(len(daily)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "turnover": float(turnover.sum()),
        "hit_rate": interval_hit_rate,
        "number_of_trades": executions,
        "average_pnl_per_trade": float(net.sum() / executions) if executions else 0.0,
        "active_fraction": float(active.mean()),
    }
    return metrics, result

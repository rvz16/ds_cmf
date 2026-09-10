"""Causal market-microstructure feature engineering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl


BASE_FEATURES = [
    "spread_bps",
    "microprice_delta_bps",
    "book_imbalance_1",
    "book_imbalance_5",
    "book_imbalance_10",
    "depth_ask_5_log",
    "depth_bid_5_log",
    "depth_ask_10_log",
    "depth_bid_10_log",
    "quote_age_seconds",
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_10",
    "momentum_5",
    "momentum_20",
    "volatility_10",
    "volatility_60",
    "depth_change_ask_5",
    "depth_change_bid_5",
    "trade_count_log",
    "trade_volume_log",
    "trade_volume_change_1",
    "trade_imbalance",
    "trade_intensity_10",
    "trade_flow_10",
    "tod_sin",
    "tod_cos",
]

OFI_FEATURES = [
    "ofi",
    "ofi_lag_1",
    "ofi_lag_5",
    "ofi_ema_5",
    "ofi_ema_20",
    "ofi_sum_10",
    "ofi_sum_60",
]
ALL_FEATURES = BASE_FEATURES + OFI_FEATURES


def _safe_imbalance(bid: pd.Series, ask: pd.Series) -> pd.Series:
    denominator = bid + ask
    return (bid - ask) / denominator.where(denominator != 0)


def make_features(bars: pl.DataFrame | pd.DataFrame, depth_levels: int = 10) -> pd.DataFrame:
    """Build features using only the current and preceding rows."""
    df = bars.to_pandas() if isinstance(bars, pl.DataFrame) else bars.copy()
    df = df.sort_values("timestamp_us", kind="stable").reset_index(drop=True)

    ask = df["asks[0].price"]
    bid = df["bids[0].price"]
    ask_size = df["asks[0].amount"]
    bid_size = df["bids[0].amount"]
    df["mid"] = (ask + bid) / 2.0
    df["spread"] = ask - bid
    df["spread_bps"] = 10_000.0 * df["spread"] / df["mid"]
    df["microprice"] = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    df["microprice_delta_bps"] = 10_000.0 * (df["microprice"] / df["mid"] - 1.0)

    for levels in (1, 5, min(10, depth_levels)):
        ask_depth = sum(df[f"asks[{i}].amount"] for i in range(levels))
        bid_depth = sum(df[f"bids[{i}].amount"] for i in range(levels))
        df[f"depth_ask_{levels}"] = ask_depth
        df[f"depth_bid_{levels}"] = bid_depth
        df[f"book_imbalance_{levels}"] = _safe_imbalance(bid_depth, ask_depth)
        if levels > 1:
            df[f"depth_ask_{levels}_log"] = np.log1p(ask_depth)
            df[f"depth_bid_{levels}_log"] = np.log1p(bid_depth)

    # Cont, Kukanov & Stoikov (2014) top-of-book order-flow imbalance.
    prev_bid, prev_ask = bid.shift(1), ask.shift(1)
    prev_bid_size, prev_ask_size = bid_size.shift(1), ask_size.shift(1)
    bid_flow = np.where(
        bid > prev_bid,
        bid_size,
        np.where(bid == prev_bid, bid_size - prev_bid_size, -prev_bid_size),
    )
    ask_flow = np.where(
        ask < prev_ask,
        ask_size,
        np.where(ask == prev_ask, ask_size - prev_ask_size, -prev_ask_size),
    )
    depth_scale = (bid_size + ask_size).replace(0, np.nan)
    df["ofi"] = (bid_flow - ask_flow) / depth_scale
    df["ofi_lag_1"] = df["ofi"].shift(1)
    df["ofi_lag_5"] = df["ofi"].shift(5)
    df["ofi_ema_5"] = df["ofi"].ewm(span=5, adjust=False, min_periods=5).mean()
    df["ofi_ema_20"] = df["ofi"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["ofi_sum_10"] = df["ofi"].rolling(10, min_periods=10).sum()
    df["ofi_sum_60"] = df["ofi"].rolling(60, min_periods=60).sum()

    log_mid = np.log(df["mid"])
    df["ret_1"] = log_mid.diff(1)
    for lag in (2, 5, 10):
        df[f"ret_{lag}"] = log_mid.diff(lag)
    df["momentum_5"] = df["ret_1"].rolling(5, min_periods=5).sum()
    df["momentum_20"] = df["ret_1"].rolling(20, min_periods=20).sum()
    df["volatility_10"] = df["ret_1"].rolling(10, min_periods=10).std()
    df["volatility_60"] = df["ret_1"].rolling(60, min_periods=60).std()
    df["depth_change_ask_5"] = np.log1p(df["depth_ask_5"]).diff()
    df["depth_change_bid_5"] = np.log1p(df["depth_bid_5"]).diff()

    volume = df["trade_volume"].astype(float)
    signed_volume = df["signed_trade_volume"].astype(float)
    df["trade_count_log"] = np.log1p(df["trade_count"])
    df["trade_volume_log"] = np.log1p(volume)
    df["trade_volume_change_1"] = df["trade_volume_log"].diff()
    df["trade_imbalance"] = signed_volume / volume.replace(0, np.nan)
    df["trade_imbalance"] = df["trade_imbalance"].fillna(0.0)
    df["trade_intensity_10"] = df["trade_count"].rolling(10, min_periods=10).sum()
    rolling_volume = volume.rolling(10, min_periods=10).sum()
    df["trade_flow_10"] = (
        signed_volume.rolling(10, min_periods=10).sum() / rolling_volume.replace(0, np.nan)
    ).fillna(0.0)

    timestamp = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    seconds = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    angle = 2 * np.pi * seconds / 86_400
    df["tod_sin"] = np.sin(angle)
    df["tod_cos"] = np.cos(angle)
    df["timestamp"] = timestamp

    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df


def feature_columns(include_ofi: bool = True) -> Sequence[str]:
    return ALL_FEATURES if include_ofi else BASE_FEATURES

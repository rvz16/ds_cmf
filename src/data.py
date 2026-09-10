"""Data audit and memory-conscious one-second bar construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def _iso_utc(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).isoformat()


def _scan(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(path, low_memory=True, rechunk=False)


def audit_data(data_dir: str | Path) -> dict[str, Any]:
    """Audit the complete raw files without loading their wide contents into pandas."""
    data_dir = Path(data_dir)
    lob = _scan(data_dir / "lob.csv")
    trades = _scan(data_dir / "trades.csv")

    lob_schema = lob.collect_schema()
    trade_schema = trades.collect_schema()
    lob_cols = lob_schema.names()
    trade_cols = trade_schema.names()
    lob_payload_cols = [c for c in lob_cols if c != ""]
    trade_payload_cols = [c for c in trade_cols if c != ""]
    lob_null_exprs = [pl.col(c).null_count().alias(c or "csv_index") for c in lob_cols]
    trade_null_exprs = [pl.col(c).null_count().alias(c or "csv_index") for c in trade_cols]

    lob_summary = lob.select(
        pl.len().alias("rows"),
        pl.col("local_timestamp").min().alias("ts_min"),
        pl.col("local_timestamp").max().alias("ts_max"),
        pl.col("local_timestamp").n_unique().alias("unique_timestamps"),
        pl.col("").n_unique().alias("unique_row_ids"),
        (pl.len() - pl.struct(lob_payload_cols).hash(seed=42).n_unique()).alias(
            "duplicate_payload_rows"
        ),
        (pl.col("local_timestamp").diff() < 0).sum().alias("out_of_order"),
        (pl.col("asks[0].price") <= pl.col("bids[0].price")).sum().alias("locked_or_crossed"),
        (pl.col("asks[0].price") <= 0).sum().alias("nonpositive_best_ask"),
        (pl.col("bids[0].price") <= 0).sum().alias("nonpositive_best_bid"),
        (pl.col("asks[0].amount") < 0).sum().alias("negative_best_ask_size"),
        (pl.col("bids[0].amount") < 0).sum().alias("negative_best_bid_size"),
        pl.col("local_timestamp").diff().quantile(0.5).alias("median_spacing_us"),
        pl.col("local_timestamp").diff().quantile(0.99).alias("p99_spacing_us"),
        *lob_null_exprs,
    ).collect(engine="streaming").row(0, named=True)

    # Check that prices move away from the touch at every adjacent depth level.
    ordering_exprs: list[pl.Expr] = []
    for level in range(24):
        ordering_exprs.extend(
            [
                (pl.col(f"asks[{level + 1}].price") <= pl.col(f"asks[{level}].price"))
                .sum()
                .alias(f"ask_{level}_{level + 1}"),
                (pl.col(f"bids[{level + 1}].price") >= pl.col(f"bids[{level}].price"))
                .sum()
                .alias(f"bid_{level}_{level + 1}"),
            ]
        )
    ordering = lob.select(ordering_exprs).collect(engine="streaming").row(0, named=True)

    trade_summary = trades.select(
        pl.len().alias("rows"),
        pl.col("local_timestamp").min().alias("ts_min"),
        pl.col("local_timestamp").max().alias("ts_max"),
        pl.col("local_timestamp").n_unique().alias("unique_timestamps"),
        pl.col("").n_unique().alias("unique_row_ids"),
        (pl.len() - pl.struct(trade_payload_cols).hash(seed=42).n_unique()).alias(
            "duplicate_payload_rows"
        ),
        (pl.col("local_timestamp").diff() < 0).sum().alias("out_of_order"),
        (pl.col("price") <= 0).sum().alias("nonpositive_price"),
        (pl.col("amount") <= 0).sum().alias("nonpositive_amount"),
        pl.col("price").min().alias("price_min"),
        pl.col("price").max().alias("price_max"),
        pl.col("amount").min().alias("amount_min"),
        pl.col("amount").max().alias("amount_max"),
        pl.col("local_timestamp").diff().quantile(0.5).alias("median_spacing_us"),
        pl.col("local_timestamp").diff().quantile(0.99).alias("p99_spacing_us"),
        *trade_null_exprs,
    ).collect(engine="streaming").row(0, named=True)
    side_counts = (
        trades.group_by("side").len().sort("side").collect(engine="streaming").to_dicts()
    )

    lob_nulls = {k: int(lob_summary.pop(k)) for k in [c or "csv_index" for c in lob_cols]}
    trade_nulls = {k: int(trade_summary.pop(k)) for k in [c or "csv_index" for c in trade_cols]}
    for summary in (lob_summary, trade_summary):
        summary["start_utc"] = _iso_utc(summary["ts_min"])
        summary["end_utc"] = _iso_utc(summary["ts_max"])

    return {
        "schema": {
            "lob_columns": lob_cols,
            "trade_columns": trade_cols,
            "lob_dtypes": {name or "csv_index": str(dtype) for name, dtype in lob_schema.items()},
            "trade_dtypes": {
                name or "csv_index": str(dtype) for name, dtype in trade_schema.items()
            },
            "timestamp_unit": "microseconds since Unix epoch (inferred from magnitude/calendar)",
            "instrument": "not provided; treated as one anonymous instrument",
        },
        "lob": {**lob_summary, "null_counts": lob_nulls},
        "trades": {**trade_summary, "side_counts": side_counts, "null_counts": trade_nulls},
        "depth_ordering_violations": ordering,
        "duplicate_definition": (
            "Duplicate payload count uses a seeded 64-bit hash of every field except the unnamed row-id. "
            "Repeated trade timestamps can be valid concurrent events. Potential duplicates are reported "
            "but not removed because no exchange trade identifier is available."
        ),
    }


def save_audit(audit: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")


def build_bars(
    data_dir: str | Path,
    cache_path: str | Path,
    frequency_seconds: int = 1,
    depth_levels: int = 10,
    force: bool = False,
) -> pl.DataFrame:
    """Create causal bars labelled by interval end.

    A row labelled T uses the last quote and all trades timestamped in (T-freq, T].
    Empty quote seconds are forward-filled and carry an explicit staleness feature.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    data_dir = Path(data_dir)
    frequency_us = int(frequency_seconds * 1_000_000)
    depth_levels = min(depth_levels, 25)
    level_cols = []
    for level in range(depth_levels):
        level_cols += [
            f"asks[{level}].price",
            f"asks[{level}].amount",
            f"bids[{level}].price",
            f"bids[{level}].amount",
        ]

    lob = (
        _scan(data_dir / "lob.csv")
        .select(["local_timestamp", *level_cols])
        .with_columns(
            (((pl.col("local_timestamp") + frequency_us - 1) // frequency_us) * frequency_us)
            .alias("timestamp_us")
        )
        .group_by("timestamp_us")
        .agg(
            pl.col("local_timestamp").last().alias("quote_timestamp_us"),
            *[pl.col(c).last().alias(c) for c in level_cols],
        )
        .sort("timestamp_us")
        .collect(engine="streaming")
    )

    trades = (
        _scan(data_dir / "trades.csv")
        .select(["local_timestamp", "side", "price", "amount"])
        .with_columns(
            (((pl.col("local_timestamp") + frequency_us - 1) // frequency_us) * frequency_us)
            .alias("timestamp_us"),
            pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0).alias("sign"),
        )
        .group_by("timestamp_us")
        .agg(
            pl.len().alias("trade_count"),
            pl.col("amount").sum().alias("trade_volume"),
            (pl.col("price") * pl.col("amount")).sum().alias("trade_notional"),
            (pl.col("amount") * pl.col("sign")).sum().alias("signed_trade_volume"),
            pl.when(pl.col("side") == "buy")
            .then(pl.col("amount"))
            .otherwise(0)
            .sum()
            .alias("buy_volume"),
            pl.when(pl.col("side") == "sell")
            .then(pl.col("amount"))
            .otherwise(0)
            .sum()
            .alias("sell_volume"),
        )
        .sort("timestamp_us")
        .collect(engine="streaming")
    )

    start = int(lob["timestamp_us"].min())
    end = int(lob["timestamp_us"].max())
    grid = pl.DataFrame({"timestamp_us": np.arange(start, end + frequency_us, frequency_us)})
    bars = grid.join(lob, on="timestamp_us", how="left").sort("timestamp_us")
    bars = bars.with_columns(
        pl.col("quote_timestamp_us").forward_fill(),
        *[pl.col(c).forward_fill() for c in level_cols],
    ).join(trades, on="timestamp_us", how="left")
    trade_fill = [
        "trade_count",
        "trade_volume",
        "trade_notional",
        "signed_trade_volume",
        "buy_volume",
        "sell_volume",
    ]
    bars = bars.with_columns(
        *[pl.col(c).fill_null(0) for c in trade_fill],
        pl.from_epoch("timestamp_us", time_unit="us").alias("timestamp"),
        ((pl.col("timestamp_us") - pl.col("quote_timestamp_us")) / 1_000_000).alias(
            "quote_age_seconds"
        ),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(cache_path, compression="zstd")
    return bars

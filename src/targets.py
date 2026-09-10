"""Leakage-safe labels and purged chronological splits."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_target(df: pd.DataFrame, horizon: int, neutral_bps: float) -> pd.DataFrame:
    result = df.copy()
    result["future_return"] = np.log(result["mid"].shift(-horizon) / result["mid"])
    threshold = neutral_bps / 10_000.0
    result["target"] = np.select(
        [result["future_return"] > threshold, result["future_return"] < -threshold],
        [1, -1],
        default=0,
    ).astype(np.int8)
    result.loc[result["future_return"].isna(), "target"] = np.nan
    return result


def purged_chronological_split(
    n_rows: int,
    horizon: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    embargo: int | None = None,
) -> np.ndarray:
    """Return train/validation/test labels with purge+embargo gaps.

    The purge is at least the forward label horizon. An equally sized embargo is
    placed after each boundary for a conservative separation.
    """
    boundary_1 = int(n_rows * train_fraction)
    boundary_2 = int(n_rows * (train_fraction + validation_fraction))
    gap = max(horizon, embargo or horizon)
    split = np.full(n_rows, "purged", dtype=object)
    split[: max(0, boundary_1 - gap)] = "train"
    split[min(n_rows, boundary_1 + gap) : max(boundary_1 + gap, boundary_2 - gap)] = "validation"
    split[min(n_rows, boundary_2 + gap) :] = "test"
    return split


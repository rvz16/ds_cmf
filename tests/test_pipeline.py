import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.features import ALL_FEATURES, make_features
from src.targets import add_target, purged_chronological_split


def _bars(n: int = 100) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    data = {
        "timestamp_us": (1_700_000_000 + np.arange(n)) * 1_000_000,
        "quote_age_seconds": np.zeros(n),
        "trade_count": np.ones(n),
        "trade_volume": np.full(n, 10.0),
        "signed_trade_volume": np.sin(x) * 5,
    }
    for level in range(10):
        data[f"asks[{level}].price"] = 100.01 + 0.01 * level + 0.001 * np.sin(x / 4)
        data[f"bids[{level}].price"] = 99.99 - 0.01 * level + 0.001 * np.sin(x / 4)
        data[f"asks[{level}].amount"] = 100 + level + x
        data[f"bids[{level}].amount"] = 120 + level + x
    return pd.DataFrame(data)


def test_features_do_not_change_when_future_is_modified():
    bars = _bars()
    original = make_features(bars)
    changed = bars.copy()
    changed.loc[80:, "asks[0].amount"] *= 100
    revised = make_features(changed)
    pd.testing.assert_frame_equal(original.loc[:79, ALL_FEATURES], revised.loc[:79, ALL_FEATURES])


def test_target_is_forward_and_split_has_gap():
    features = make_features(_bars())
    labelled = add_target(features, horizon=5, neutral_bps=0)
    expected = np.log(features.loc[5, "mid"] / features.loc[0, "mid"])
    assert np.isclose(labelled.loc[0, "future_return"], expected)
    split = purged_chronological_split(100, horizon=5)
    assert np.all(split[55:65] == "purged")
    assert np.all(split[75:85] == "purged")


def test_signal_executes_one_row_later():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="s", tz="UTC"),
            "mid": [100.0, 100.0, 110.0],
            "spread_bps": [0.0, 0.0, 0.0],
        }
    )
    metrics, path = run_backtest(
        frame, np.array([1, 0, 0]), fee_bps=0, slippage_bps=0, include_spread=False
    )
    assert np.allclose(path["position"], [0, 1, 0])
    assert np.isclose(metrics["gross_pnl"], 0.1)


def test_position_cap_and_cost_accounting():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC"),
            "mid": [100.0, 100.0, 100.0, 100.0],
            "spread_bps": [2.0, 2.0, 2.0, 2.0],
        }
    )
    metrics, path = run_backtest(
        frame,
        np.array([2, 2, 0, 0]),
        fee_bps=1.0,
        slippage_bps=0.5,
        include_spread=True,
        max_position=1.0,
    )
    assert np.allclose(path["position"], [0, 1, 1, 0])
    assert np.isclose(metrics["turnover"], 2.0)
    assert np.isclose(metrics["net_pnl"], -0.0005)

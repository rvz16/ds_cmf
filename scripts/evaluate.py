#!/usr/bin/env python3
"""Print a compact held-out comparison from an existing experiment."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
results = pd.read_csv(ROOT / "reports/results.csv")
view = results.loc[
    results["cost_scenario"] == "realistic_base",
    ["horizon", "model", "accuracy", "macro_f1", "net_pnl", "sharpe_daily_annualized", "turnover"],
].sort_values(["horizon", "net_pnl"], ascending=[True, False])
print(view.to_string(index=False))


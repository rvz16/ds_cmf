#!/usr/bin/env python3
"""Run the complete audit, EDA, model, backtest, and reporting workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ds_cmf_matplotlib")

import joblib
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data import audit_data, build_bars, save_audit
from src.features import ALL_FEATURES, BASE_FEATURES, make_features
from src.metrics import prediction_metrics
from src.models import fit_lightgbm, fit_logistic, fit_naive_majority, fit_ofi_threshold
from src.reporting import (
    create_eda,
    plot_confusion,
    plot_equity,
    plot_target_balance,
    save_json,
    write_final_report,
    write_submission_tables,
)
from src.targets import add_target, purged_chronological_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--force-bars", action="store_true", help="Rebuild cached one-second bars")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    np.random.seed(config["seed"])
    reports = ROOT / "reports"
    figures = reports / "figures"
    models_dir = ROOT / "artifacts" / "models"
    reports.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Auditing complete raw files", flush=True)
    audit = audit_data(ROOT / "data")
    save_audit(audit, reports / "data_audit.json")

    print("[2/6] Building/loading causal bars", flush=True)
    bars = build_bars(
        ROOT / "data",
        ROOT / "artifacts" / f"bars_{config['frequency_seconds']}s.parquet",
        frequency_seconds=config["frequency_seconds"],
        depth_levels=config["depth_levels"],
        force=args.force_bars,
    )

    print("[3/6] Engineering causal features and creating EDA", flush=True)
    features = make_features(bars, depth_levels=config["depth_levels"])
    eda = create_eda(features, figures)
    save_json(eda, reports / "eda_summary.json")

    all_results: list[dict] = []
    target_balance: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    equity_curves: dict[str, pd.DataFrame] = {}
    selected_confusion: list[list[int]] | None = None
    selected_confusion_title = ""

    print("[4/6] Training chronological experiments", flush=True)
    for horizon in config["horizons"]:
        labelled = add_target(features, horizon=horizon, neutral_bps=config["neutral_bps"])
        labelled = labelled.loc[labelled["target"].notna() & (labelled["spread"] > 0)].reset_index(drop=True)
        y = labelled["target"].astype(np.int8).to_numpy()
        split = purged_chronological_split(
            len(labelled),
            horizon=horizon,
            train_fraction=config["train_fraction"],
            validation_fraction=config["validation_fraction"],
        )
        for label, count in pd.Series(y).value_counts().sort_index().items():
            target_balance.append(
                {
                    "horizon": int(horizon),
                    "target": {-1: "down", 0: "neutral", 1: "up"}[int(label)],
                    "count": int(count),
                    "fraction": float(count / len(y)),
                }
            )

        fits = [
            fit_naive_majority(y, split),
            fit_ofi_threshold(labelled, y, split),
            fit_logistic(labelled, y, split, list(BASE_FEATURES), "logistic_no_ofi", config["seed"]),
            fit_logistic(labelled, y, split, list(ALL_FEATURES), "logistic_all", config["seed"]),
            fit_lightgbm(labelled, y, split, list(ALL_FEATURES), config["seed"]),
        ]
        test = split == "test"
        test_frame = labelled.loc[test].reset_index(drop=True)
        prediction_output = test_frame[["timestamp", "mid", "future_return", "target"]].copy()
        prediction_output["horizon"] = horizon

        for fit in fits:
            pred_test = fit.predictions[test]
            predictive = prediction_metrics(y[test], pred_test)
            prediction_output[fit.name] = pred_test
            if fit.estimator is not None:
                joblib.dump(fit.estimator, models_dir / f"{fit.name}_h{horizon}.joblib")
            save_json(fit.params, models_dir / f"{fit.name}_h{horizon}.json")

            for scenario_name, cost in config["costs"].items():
                trading, curve = run_backtest(
                    test_frame, pred_test, max_position=config["max_position"], **cost
                )
                row = {
                    "horizon": int(horizon),
                    "model": fit.name,
                    "cost_scenario": scenario_name,
                    **cost,
                    "validation_macro_f1": fit.validation_f1,
                    **predictive,
                    **trading,
                    "test_rows": int(test.sum()),
                }
                row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
                all_results.append(row)
                if horizon == config["plot_horizon"] and scenario_name == "realistic_base":
                    equity_curves[fit.name] = curve

            if horizon == config["plot_horizon"] and fit.name == "lightgbm_all":
                selected_confusion = predictive["confusion_matrix"]
                selected_confusion_title = f"LightGBM test confusion matrix ({horizon}s horizon)"
                importance = pd.DataFrame(
                    {"feature": ALL_FEATURES, "importance": fit.estimator.feature_importances_}
                ).sort_values("importance", ascending=False)
                importance.to_csv(reports / f"feature_importance_h{horizon}.csv", index=False)
        prediction_frames.append(prediction_output)

    print("[5/6] Saving results and diagnostic figures", flush=True)
    results = pd.DataFrame(all_results)
    results.to_csv(reports / "results.csv", index=False)
    write_submission_tables(results, reports)
    counts = pd.DataFrame(target_balance)
    counts.to_csv(reports / "target_balance.csv", index=False)
    plot_target_balance(counts, figures)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        ROOT / "artifacts" / "test_predictions.parquet", index=False
    )
    if selected_confusion is not None:
        plot_confusion(selected_confusion, selected_confusion_title, figures / "confusion_matrix.png")
    if equity_curves:
        plot_equity(
            equity_curves,
            figures / "test_equity_curves.png",
            f"Held-out test PnL, realistic base costs ({config['plot_horizon']}s targets)",
        )

    print("[6/6] Writing data-driven final report", flush=True)
    write_final_report(reports / "final_report.md", audit, eda, results, config)
    print(f"Done. Results: {reports / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()

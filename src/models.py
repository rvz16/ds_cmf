"""Baselines and reproducible tabular classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


@dataclass
class StrategyFit:
    name: str
    predictions: np.ndarray
    validation_f1: float
    params: dict[str, Any]
    estimator: Any = None


def _score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=[-1, 0, 1], average="macro", zero_division=0))


def fit_naive_majority(y: np.ndarray, split: np.ndarray) -> StrategyFit:
    train_y = y[split == "train"]
    values, counts = np.unique(train_y, return_counts=True)
    majority = int(values[np.argmax(counts)])
    pred = np.full(len(y), majority, dtype=np.int8)
    val = split == "validation"
    return StrategyFit("naive_majority", pred, _score(y[val], pred[val]), {"class": majority})


def fit_ofi_threshold(frame: pd.DataFrame, y: np.ndarray, split: np.ndarray) -> StrategyFit:
    score = frame["ofi_ema_20"].fillna(0.0).to_numpy()
    train, val = split == "train", split == "validation"
    candidates = np.unique(np.quantile(np.abs(score[train]), [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]))
    best: tuple[float, float, np.ndarray] | None = None
    for threshold in candidates:
        pred = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0)).astype(np.int8)
        metric = _score(y[val], pred[val])
        if best is None or metric > best[0]:
            best = (metric, float(threshold), pred)
    assert best is not None
    return StrategyFit("ofi_threshold", best[2], best[0], {"absolute_threshold": best[1]})


def fit_logistic(
    frame: pd.DataFrame,
    y: np.ndarray,
    split: np.ndarray,
    features: list[str],
    name: str,
    seed: int,
) -> StrategyFit:
    train, val = split == "train", split == "validation"
    best: tuple[float, float, Pipeline] | None = None
    for c_value in (0.05, 0.5):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                # Regime shifts can put test observations hundreds of training
                # standard deviations away. Bound influence without using any
                # validation/test quantile information.
                ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -20.0, "a_max": 20.0})),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=200,
                        # One-vs-rest liblinear is numerically stable for rare,
                        # extreme-but-valid microstructure observations.
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(frame.loc[train, features], y[train])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred_val = pipeline.predict(frame.loc[val, features])
        metric = _score(y[val], pred_val)
        if best is None or metric > best[0]:
            best = (metric, c_value, pipeline)
    assert best is not None
    # NumPy 2.x can emit spurious BLAS floating warnings here even when the
    # clipped design matrix, coefficients, and decision scores remain finite.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        decision = best[2].decision_function(frame[features])
        if not np.isfinite(decision).all():
            raise FloatingPointError(f"Non-finite logistic decision score in {name}")
        predictions = best[2].predict(frame[features]).astype(np.int8)
    return StrategyFit(name, predictions, best[0], {"C": best[1], "features": features}, best[2])


def fit_lightgbm(
    frame: pd.DataFrame,
    y: np.ndarray,
    split: np.ndarray,
    features: list[str],
    seed: int,
) -> StrategyFit:
    train, val = split == "train", split == "validation"
    # Map {-1, 0, 1} to LightGBM's required {0, 1, 2} labels.
    y_encoded = y + 1
    candidates = [
        {"num_leaves": 15, "min_child_samples": 100, "learning_rate": 0.05},
        {"num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.03},
    ]
    best: tuple[float, dict[str, Any], lgb.LGBMClassifier] | None = None
    for candidate in candidates:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **candidate,
        )
        model.fit(
            frame.loc[train, features],
            y_encoded[train],
            eval_set=[(frame.loc[val, features], y_encoded[val])],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        pred_val = model.predict(frame.loc[val, features]).astype(np.int8) - 1
        metric = _score(y[val], pred_val)
        if best is None or metric > best[0]:
            best = (metric, candidate, model)
    assert best is not None
    predictions = best[2].predict(frame[features]).astype(np.int8) - 1
    params = {**best[1], "best_iteration": int(best[2].best_iteration_ or 300), "features": features}
    return StrategyFit("lightgbm_all", predictions, best[0], params, best[2])

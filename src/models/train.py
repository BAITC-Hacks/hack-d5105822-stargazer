"""Train and evaluate the power-forecasting model for one turbine.

One LightGBM regressor per turbine, trained on BOTH lead-time groups
(lead_hours as an explicit feature) so it learns how forecast confidence
should degrade from 24h to 48h, rather than needing two separate models.
Compared against two baselines on the same held-out January-2026 slice:
persistence (last known actual power) and the empirical power curve.
"""

from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import schema as data_schema
from src.features import schema as feature_schema
from src.models import schema
from src.models.baseline import PowerCurveBaseline


def load_turbine_train_table(turbine_id: int) -> pd.DataFrame:
    path = data_schema.PROCESSED_DATA_DIR / feature_schema.TRAIN_PARQUET_TEMPLATE.format(turbine_id=turbine_id)
    return pd.read_parquet(path)


class TrainedModel(NamedTuple):
    """The two fitted artifacts needed to reproduce a prediction at
    inference time: the residual booster and the curve it corrects.
    """

    turbine_id: int
    booster: lgb.Booster
    curve_baseline: PowerCurveBaseline


def load_trained_model(turbine_id: int) -> TrainedModel:
    model_path = schema.MODELS_OUTPUT_DIR / schema.MODEL_FILENAME_TEMPLATE.format(turbine_id=turbine_id)
    curve_path = schema.MODELS_OUTPUT_DIR / schema.POWER_CURVE_FILENAME_TEMPLATE.format(turbine_id=turbine_id)
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    curve_baseline = PowerCurveBaseline.load(curve_path)
    return TrainedModel(turbine_id=turbine_id, booster=booster, curve_baseline=curve_baseline)


def predict(trained: TrainedModel, features: pd.DataFrame) -> np.ndarray:
    """features must already have the columns in feature_schema.FEATURE_COLUMNS,
    with `weather_source` as a `category` dtype matching training (see
    src.agent.tools for how the agent builds this consistently).
    """
    X = features[feature_schema.FEATURE_COLUMNS]
    curve_pred = trained.curve_baseline.predict(features[schema.POWER_CURVE_FEATURE])
    residual_pred = trained.booster.predict(X)
    return np.clip(curve_pred + residual_pred, 0, 1)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train_fit (early-stopping training) / train_earlystop (early-stopping
    dev set) / validation (January 2026, never touched during fitting).
    """
    train = df[df["timestamp"] < schema.VALID_START]
    validation = df[df["timestamp"] >= schema.VALID_START]
    train_fit = train[train["timestamp"] < schema.EARLY_STOP_START]
    train_earlystop = train[train["timestamp"] >= schema.EARLY_STOP_START]
    return train_fit, train_earlystop, validation


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    # weather_source must already be `category` dtype on the full table
    # (see train_and_evaluate_turbine) before splitting -- converting each
    # split independently would let train/validation disagree on which
    # category maps to which integer code (e.g. validation only ever sees
    # "previous_runs", since it's Jan 2026, past the proxy cutover).
    return df[feature_schema.FEATURE_COLUMNS].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_turbine(turbine_id: int) -> dict:
    df = load_turbine_train_table(turbine_id)
    for col in schema.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    train_fit, train_earlystop, validation = time_split(df)

    X_fit, y_fit = _prepare_X(train_fit), train_fit[feature_schema.TARGET_COLUMN]
    X_es, y_es = _prepare_X(train_earlystop), train_earlystop[feature_schema.TARGET_COLUMN]

    # LightGBM (plain, direct target) came out roughly tied with the plain
    # power-curve baseline on a first pass -- the curve is an unusually
    # strong prior for this problem, and a general-purpose L2 booster with
    # 17 features (several weak/noisy at this forecast horizon) doesn't
    # reliably beat it on MAE. Instead, follow the plan's residual-model
    # design (PLAN_AND_ARCHITECTURE.md section 5): fit the curve first,
    # then let LightGBM learn only the correction on top of it. If the
    # correction learns nothing useful the result is exactly the baseline,
    # so this can't do worse than the curve on the fitted distribution.
    curve_baseline = PowerCurveBaseline().fit(train_fit[schema.POWER_CURVE_FEATURE], y_fit)
    y_fit_resid = y_fit.values - curve_baseline.predict(train_fit[schema.POWER_CURVE_FEATURE])
    y_es_resid = y_es.values - curve_baseline.predict(train_earlystop[schema.POWER_CURVE_FEATURE])

    model = lgb.LGBMRegressor(**schema.LGBM_PARAMS)
    model.fit(
        X_fit,
        y_fit_resid,
        eval_X=X_es,
        eval_y=y_es_resid,
        callbacks=[lgb.early_stopping(schema.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    results = {"turbine_id": turbine_id, "n_train_fit": len(train_fit), "n_train_earlystop": len(train_earlystop), "by_lead": {}}

    for lead_hours, group in validation.groupby("lead_hours"):
        y_true = group[feature_schema.TARGET_COLUMN].values

        curve_pred = curve_baseline.predict(group[schema.POWER_CURVE_FEATURE])
        residual_pred = model.predict(_prepare_X(group))
        lgbm_pred = np.clip(curve_pred + residual_pred, 0, 1)
        persistence_pred = group[schema.PERSISTENCE_FEATURE].fillna(y_fit.mean()).clip(0, 1).values

        results["by_lead"][int(lead_hours)] = {
            "n_valid": len(group),
            "lgbm": compute_metrics(y_true, lgbm_pred),
            "power_curve_baseline": compute_metrics(y_true, curve_pred),
            "persistence_baseline": compute_metrics(y_true, persistence_pred),
        }

    # LightGBM's C++ save_model() chokes on non-ASCII characters anywhere in
    # the path (this repo lives under a Cyrillic Windows username), so the
    # model is serialized to a string in-process and written with Python's
    # own (Unicode-safe) file I/O instead.
    model_path = schema.MODELS_OUTPUT_DIR / schema.MODEL_FILENAME_TEMPLATE.format(turbine_id=turbine_id)
    model_path.write_text(model.booster_.model_to_string(), encoding="utf-8")
    results["model_path"] = str(model_path)

    curve_path = schema.MODELS_OUTPUT_DIR / schema.POWER_CURVE_FILENAME_TEMPLATE.format(turbine_id=turbine_id)
    curve_baseline.save(curve_path)
    results["power_curve_path"] = str(curve_path)

    # Gain, not split-count (the sklearn wrapper's default): split-count
    # over-ranks high-cardinality features like wind_direction purely
    # because they offer more distinct thresholds to split on, which made
    # an earlier pass look like the model was ignoring wind speed when gain
    # showed the opposite.
    importance = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=X_fit.columns).sort_values(
        ascending=False
    )
    results["feature_importance"] = importance.to_dict()

    return results

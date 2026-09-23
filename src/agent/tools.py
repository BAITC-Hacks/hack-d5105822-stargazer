"""Agent tools: each function wraps one existing pipeline capability and
does nothing "smart" on its own. The autonomous decision-making (retry,
low-confidence flagging, when to recompute) lives in the orchestrator,
which calls these in sequence -- see src/agent/orchestrator.py.
"""

import json

import numpy as np
import pandas as pd

from src import config
from src.agent import schema as agent_schema
from src.features import engineering, schema as feature_schema, weather_features
from src.models import schema as model_schema
from src.models.train import TrainedModel, predict as model_predict
from src.weather import client as weather_client


def fetch_weather(
    turbine: config.TurbineInfo,
    target_date: str,
    lead_hours: int,
    mode: str,
) -> dict:
    """Tool 1: get the weather forecast for `target_date`, as it would
    actually have been known `lead_hours` before (mode="backtest") or a
    live forward-looking forecast (mode="live").

    Returns {"success": bool, "weather": DataFrame | None, "error": str | None}.
    """
    try:
        if mode == "backtest":
            weather = weather_features.get_weather_wide(
                turbine.latitude,
                turbine.longitude,
                lead_hours,
                target_date,
                target_date,
                agent_schema.BACKTEST_FORCE_PREVIOUS_RUNS_COVERAGE_START,
            )
        elif mode == "live":
            long_df = weather_client.fetch_live_forecast(turbine.latitude, turbine.longitude, forecast_days=2)
            long_df = long_df[long_df["valid_time"].dt.date.astype(str) == target_date]
            weather = weather_features._wide_from_long(long_df) if len(long_df) else long_df
        else:
            raise ValueError(f"unknown mode: {mode}")
    except Exception as exc:  # noqa: BLE001 -- any failure here is a "retry" signal to the orchestrator
        return {"success": False, "weather": None, "error": str(exc)}

    if weather is None or weather.empty:
        return {"success": False, "weather": None, "error": "empty weather response"}
    return {"success": True, "weather": weather, "error": None}


def build_features(
    turbine: config.TurbineInfo,
    weather: pd.DataFrame,
    lead_hours: int,
    power_clean: pd.Series,
) -> pd.DataFrame:
    """Tool 2: turn raw weather + the turbine's own known history into the
    exact feature table the model was trained on (src/features/engineering.py
    -- the SAME code path training used, so inference sees identical inputs).
    """
    df = weather.copy()
    lags = engineering.lag_features(df["valid_time"], lead_hours, power_clean)
    df = df.merge(lags, on="valid_time", how="left")
    df = engineering.engineer_physical_features(df)
    df = engineering.engineer_time_features(df)
    df["turbine_id"] = turbine.turbine_id
    df["weather_source"] = df["weather_source"].astype("category")
    return df


def run_forecast_model(trained: TrainedModel, features: pd.DataFrame) -> pd.DataFrame:
    """Tool 3: run the trained (power curve + LightGBM residual) model."""
    out = features[["valid_time", "lead_hours", "weather_source"] + feature_schema.FEATURE_COLUMNS[1:-1]].copy()
    out["curve_pred"] = trained.curve_baseline.predict(features[model_schema.POWER_CURVE_FEATURE])
    out["residual_pred"] = trained.booster.predict(features[feature_schema.FEATURE_COLUMNS])
    out["predicted_power_raw"] = out["curve_pred"] + out["residual_pred"]
    out["predicted_power"] = out["predicted_power_raw"].clip(0, 1)
    return out


def qa_and_analyze(forecast: pd.DataFrame) -> dict:
    """Tool 4: sanity-check the forecast and its inputs, and DECIDE whether
    it is trustworthy enough to publish as-is. Never blocks the forecast --
    the case wants an hourly number every cycle -- but flags it so a
    downstream consumer (or a human) knows to discount it.
    """
    n = len(forecast)
    n_clipped = int((forecast["predicted_power_raw"] != forecast["predicted_power"]).sum())
    large_correction = forecast["residual_pred"].abs() > agent_schema.LARGE_CORRECTION_ABS_THRESHOLD
    frac_large_correction = float(large_correction.mean()) if n else 0.0

    comments = []
    verdict = "ok"

    if n_clipped:
        comments.append(
            f"{n_clipped}/{n} часов: сырой прогноз (кривая + ML-поправка) вышел за физическую границу "
            f"[0,1] и был обрезан -- обычно означает экстремальный входной ветер."
        )
    if frac_large_correction > agent_schema.LARGE_CORRECTION_FRAC_THRESHOLD:
        verdict = "low_confidence"
        comments.append(
            f"Необычно крупная ML-поправка к кривой мощности для {frac_large_correction:.0%} часов "
            f"(> {agent_schema.LARGE_CORRECTION_ABS_THRESHOLD}) -- возможно, погодные условия нетипичны "
            f"для обучающей выборки. Прогноз опубликован, но с пониженной уверенностью."
        )
    if not comments:
        comments.append("Все проверки пройдены: входные данные полны, поправка модели в норме.")

    return {"verdict": verdict, "comments": comments, "n_hours": n, "frac_large_correction": frac_large_correction}


def persist_and_report(
    turbine: config.TurbineInfo,
    issue_date: str,
    lead_hours: int,
    mode: str,
    forecast: pd.DataFrame,
    qa_result: dict,
    n_fetch_attempts: int,
) -> dict:
    """Tool 5: append this cycle's forecast to the turbine's running CSV and
    write a JSON run log for this (turbine, issue_date, lead_hours) cycle.
    """
    agent_schema.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    agent_schema.RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    out = forecast[["valid_time", "lead_hours", "predicted_power"]].copy()
    out.insert(0, "turbine_id", turbine.turbine_id)
    out.insert(1, "issue_date", issue_date)

    all_issues_path = agent_schema.FORECASTS_DIR / f"turbine_{turbine.turbine_id}_all_issues.csv"
    out.to_csv(all_issues_path, mode="a", header=not all_issues_path.exists(), index=False)

    log = {
        "turbine_id": turbine.turbine_id,
        "issue_date": issue_date,
        "lead_hours": lead_hours,
        "mode": mode,
        "n_weather_fetch_attempts": n_fetch_attempts,
        "n_hours_forecast": qa_result["n_hours"],
        "verdict": qa_result["verdict"],
        "comments": qa_result["comments"],
        "target_hours": [str(t) for t in forecast["valid_time"]],
        "predicted_power": [round(float(p), 4) for p in forecast["predicted_power"]],
    }
    log_path = agent_schema.RUN_LOGS_DIR / f"turbine_{turbine.turbine_id}_{issue_date}_lead{lead_hours}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"forecast_path": str(all_issues_path), "log_path": str(log_path)}

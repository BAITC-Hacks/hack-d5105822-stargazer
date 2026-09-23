"""The agentic control loop: for one turbine and one issue date, decide
what to do at each step rather than just running a fixed pipeline.

State machine per lead-time group (24h, then 48h):
    fetch_weather --(failed)--> retry (up to MAX_WEATHER_FETCH_RETRIES) --(still failed)--> give up, log it
                  --(ok)--> build_features -> run_forecast_model -> qa_and_analyze -> persist_and_report

"Повторный расчёт при обновлении входных данных" (recompute when input
data updates) is not a separate code path: it falls out of running this
cycle once per calendar day. Each day's lead=24h forecast for tomorrow
supersedes yesterday's lead=48h forecast for the same hours, made with
older weather -- see scripts/run_backtest.py's final consolidation, which
always keeps the freshest (smallest-lead) prediction per hour.
"""

import pandas as pd

from src import config
from src.agent import schema as agent_schema
from src.agent import tools
from src.features import schema as feature_schema
from src.models.train import TrainedModel


def run_forecast_cycle(
    turbine: config.TurbineInfo,
    issue_date: str,
    trained: TrainedModel,
    power_clean: pd.Series,
    mode: str = "backtest",
) -> dict:
    """Run the full agentic cycle for every lead-time group on one issue
    date. Returns {lead_hours: {verdict, comments, n_hours, forecast_df, ...}}.
    """
    results = {}

    for lead_hours in feature_schema.LEAD_HOURS_OPTIONS:
        target_date = (pd.Timestamp(issue_date) + pd.Timedelta(hours=lead_hours)).date().isoformat()

        attempt = 0
        weather_result = {"success": False, "error": "not attempted"}
        while attempt < agent_schema.MAX_WEATHER_FETCH_RETRIES and not weather_result["success"]:
            attempt += 1
            weather_result = tools.fetch_weather(turbine, target_date, lead_hours, mode)

        if not weather_result["success"]:
            results[lead_hours] = {
                "target_date": target_date,
                "verdict": "failed_weather_fetch",
                "comments": [f"Не удалось получить погоду после {attempt} попыток: {weather_result['error']}"],
                "n_hours": 0,
                "n_fetch_attempts": attempt,
                "forecast": None,
            }
            continue

        features = tools.build_features(turbine, weather_result["weather"], lead_hours, power_clean)
        forecast = tools.run_forecast_model(trained, features)
        qa_result = tools.qa_and_analyze(forecast)
        persisted = tools.persist_and_report(turbine, issue_date, lead_hours, mode, forecast, qa_result, attempt)

        results[lead_hours] = {
            "target_date": target_date,
            "verdict": qa_result["verdict"],
            "comments": qa_result["comments"],
            "n_hours": qa_result["n_hours"],
            "n_fetch_attempts": attempt,
            "forecast": forecast,
            **persisted,
        }

    return results

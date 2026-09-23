"""Constants for the agentic forecasting cycle and backtest runner."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECASTS_DIR = PROJECT_ROOT / "outputs" / "forecasts"
RUN_LOGS_DIR = PROJECT_ROOT / "outputs" / "run_logs"
BACKTEST_REPORT_PATH = FORECASTS_DIR / "backtest_report.md"

# The backtest window is entirely inside 2026, far past every variable's
# previous-runs coverage cutover (measured: 2024-01-19..2024-02-18, see
# PLAN_AND_ARCHITECTURE.md 4.1.1/6.4). Forcing an early coverage_start here
# means the shared get_weather_wide() helper always takes the previous_runs
# branch for backtest dates and never the reanalysis_proxy fallback --
# exactly the "no leakage, no cheating with actual weather" contract the
# case requires. This is a *fact already verified* for this date range
# (scripts/run_weather_client_check.py: 0 missing), not a new assumption.
BACKTEST_FORCE_PREVIOUS_RUNS_COVERAGE_START = "2020-01-01"

MAX_WEATHER_FETCH_RETRIES = 2

# QA thresholds for flagging a forecast as low-confidence rather than
# rejecting it outright -- the case wants a forecast every cycle, just an
# honest one.
LARGE_CORRECTION_ABS_THRESHOLD = 0.25  # |lgbm_residual| this big is an unusually large correction
LARGE_CORRECTION_FRAC_THRESHOLD = 0.30  # ...flag the cycle if more than this share of hours hit it

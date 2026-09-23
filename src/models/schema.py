"""Constants for the baseline + LightGBM forecasting models.

Time-based split (per PLAN_AND_ARCHITECTURE.md section 5): train on
everything before 2026-01-01, validate on January 2026 -- the last full
month of real history, held out exactly like the actual Feb-2026 backtest
will be. January 2026 also postdates the previous-runs coverage cutover
(2024-02-16/18), so validation is never contaminated by the
`reanalysis_proxy` fallback -- it reflects real deployment conditions.

A further split inside the training period is used only for LightGBM's
early stopping, so the January validation metric is never touched during
model fitting.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

VALID_START = "2026-01-01"
EARLY_STOP_START = "2025-11-01"  # carve-out inside the training period, for early stopping only

CATEGORICAL_FEATURES = ["weather_source"]
POWER_CURVE_FEATURE = "wind_speed_100m"
PERSISTENCE_FEATURE = "lag_power_t0"

LGBM_PARAMS = {
    # regression_l1 (MAE) beat the default L2 objective on MAE *and* still
    # beat the power-curve baseline on RMSE/R2 -- L2 alone traded a few
    # large-error corrections for many small ones, which improved RMSE but
    # left MAE slightly worse than just using the raw curve.
    "objective": "regression_l1",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 30,
    "random_state": 0,
}
EARLY_STOPPING_ROUNDS = 50

POWER_CURVE_BIN_WIDTH = 0.5

MODEL_FILENAME_TEMPLATE = "turbine_{turbine_id}_lgbm.txt"
POWER_CURVE_FILENAME_TEMPLATE = "turbine_{turbine_id}_power_curve.json"

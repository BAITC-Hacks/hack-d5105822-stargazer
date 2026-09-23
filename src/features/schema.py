"""Constants for assembling the supervised training dataset.

Each historical hour with a trustworthy actual-power reading becomes TWO
training rows -- one simulating a forecast issued ~24h earlier, one ~48h
earlier -- matching the case's required 24-48h horizon and letting a
single model learn how accuracy should degrade with `lead_hours`.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset"

LEAD_HOURS_OPTIONS = (24, 48)

# Rolling-lag window for the "recent operating state at issue time" feature.
LAG_ROLLING_WINDOW_HOURS = 24
LAG_ROLLING_MIN_PERIODS = 12  # require >=half the window to trust the mean

# Physical constants for air-density feature.
GAS_CONSTANT_DRY_AIR = 287.05  # J / (kg*K)

FEATURE_COLUMNS = [
    "lead_hours",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "temperature_2m",
    "surface_pressure",
    "air_density",
    "wind_speed_hub_cubed",
    "power_flux_proxy",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month",
    "lag_power_t0",
    "lag_power_mean_24h",
    "weather_source",
]

TARGET_COLUMN = "power"

TRAIN_PARQUET_TEMPLATE = "turbine_{turbine_id}_train.parquet"

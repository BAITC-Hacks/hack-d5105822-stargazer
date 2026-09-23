"""Feature engineering shared between offline training-set assembly
(src/features/build.py) and the online agent (src/agent/tools.py), so
both paths compute features identically -- a model must see the same
inputs at inference as it did in training.
"""

import numpy as np
import pandas as pd

from src.features import schema


def clean_power_series(hourly: pd.DataFrame) -> pd.Series:
    """Actual power indexed by hour, with untrustworthy (`is_gap`) hours
    set to NaN so lag features never silently use a bad reading.
    """
    indexed = hourly.set_index("timestamp")
    return indexed["power"].where(~indexed["is_gap"])


def lag_features(valid_times: pd.Series, lead_hours: int, power_clean: pd.Series) -> pd.DataFrame:
    """Lag features computed strictly at/before issue time
    t0 = valid_time - lead_hours -- the only turbine-history information
    that was actually available when a lead_hours-ahead forecast would
    have been issued.
    """
    t0 = valid_times - pd.to_timedelta(lead_hours, unit="h")
    rolling_mean = power_clean.rolling(
        f"{schema.LAG_ROLLING_WINDOW_HOURS}h", min_periods=schema.LAG_ROLLING_MIN_PERIODS
    ).mean()

    return pd.DataFrame(
        {
            "valid_time": valid_times.values,
            "lag_power_t0": power_clean.reindex(t0).values,
            "lag_power_mean_24h": rolling_mean.reindex(t0).values,
        }
    )


def engineer_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hub_wind = df["wind_speed_100m"].fillna(df["wind_speed_10m"])
    temp_kelvin = df["temperature_2m"] + 273.15
    pressure_pa = df["surface_pressure"] * 100.0
    df["air_density"] = pressure_pa / (schema.GAS_CONSTANT_DRY_AIR * temp_kelvin)
    df["wind_speed_hub_cubed"] = hub_wind**3
    df["power_flux_proxy"] = df["air_density"] * df["wind_speed_hub_cubed"]
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["valid_time"].dt.hour + df["valid_time"].dt.minute / 60.0
    doy = df["valid_time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["valid_time"].dt.month
    return df

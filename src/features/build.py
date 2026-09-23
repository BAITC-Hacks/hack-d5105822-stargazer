"""Assemble the supervised training table for one turbine.

Pipeline per turbine:
  1. load the resampled hourly table (data/processed/turbine_<id>_hourly.parquet);
  2. for each lead in (24, 48)h, fetch weather features for every historical
     hour as if a forecast had been issued that many hours earlier;
  3. attach a "recent operating state" lag feature computed from the
     turbine's OWN actual power up to the issue time t0 = valid_time - lead
     (never later -- this is the leakage boundary for lag features);
  4. engineer physical (air density, cubed hub-height wind) and cyclical
     time features;
  5. drop rows whose target is untrustworthy (`is_gap`) or looks like an
     unpredictable operational anomaly (`curtailment_suspected` -- this
     flag is itself derived from the target, so it must never be used as
     an input feature, only as a row filter);
  6. concatenate the two lead groups into one long training table.
"""

import numpy as np
import pandas as pd

from src import config
from src.data import schema as data_schema
from src.features import schema, weather_features
from src.weather import client


def load_turbine_hourly(turbine_id: int) -> pd.DataFrame:
    path = data_schema.PROCESSED_DATA_DIR / f"turbine_{turbine_id}_hourly.parquet"
    return pd.read_parquet(path)


def _clean_power_series(hourly: pd.DataFrame) -> pd.Series:
    """Actual power indexed by hour, with untrustworthy (`is_gap`) hours
    set to NaN so lag features never silently use a bad reading.
    """
    indexed = hourly.set_index("timestamp")
    return indexed["power"].where(~indexed["is_gap"])


def _lag_features(valid_times: pd.Series, lead_hours: int, power_clean: pd.Series) -> pd.DataFrame:
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


def _engineer_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hub_wind = df["wind_speed_100m"].fillna(df["wind_speed_10m"])
    temp_kelvin = df["temperature_2m"] + 273.15
    pressure_pa = df["surface_pressure"] * 100.0
    df["air_density"] = pressure_pa / (schema.GAS_CONSTANT_DRY_AIR * temp_kelvin)
    df["wind_speed_hub_cubed"] = hub_wind**3
    df["power_flux_proxy"] = df["air_density"] * df["wind_speed_hub_cubed"]
    return df


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["valid_time"].dt.hour + df["valid_time"].dt.minute / 60.0
    doy = df["valid_time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["valid_time"].dt.month
    return df


def assemble_turbine_dataset(turbine_id: int) -> tuple[pd.DataFrame, dict]:
    turbine = config.load_turbines()[turbine_id]
    hourly = load_turbine_hourly(turbine_id)

    start_date = hourly["timestamp"].min().date().isoformat()
    end_date = hourly["timestamp"].max().date().isoformat()

    power_clean = _clean_power_series(hourly)
    hourly_target = hourly[["timestamp", "power", "is_gap", "curtailment_suspected"]].rename(
        columns={"timestamp": "valid_time"}
    )

    stats = {"turbine_id": turbine_id, "by_lead": {}}
    lead_frames = []

    for lead_hours in schema.LEAD_HOURS_OPTIONS:
        # Coverage backfill is not uniform across variables (e.g.
        # wind_speed_100m can lag wind_speed_10m by weeks), so the cutover
        # must be the max across ALL requested variables for this lead --
        # otherwise some columns would be silently NULL past a
        # single-variable cutover. See src/weather/client.py.
        coverage_start = client.find_full_coverage_start(turbine.latitude, turbine.longitude, lead_days=lead_hours // 24)
        if coverage_start is None:
            raise RuntimeError(
                f"no full previous-runs coverage at all for turbine {turbine_id}, lead={lead_hours}h -- cannot build a leak-free dataset"
            )
        stats["by_lead"].setdefault(lead_hours, {})["coverage_start"] = coverage_start

        weather_wide = weather_features.get_weather_wide(
            turbine.latitude, turbine.longitude, lead_hours, start_date, end_date, coverage_start
        )
        merged = hourly_target.merge(weather_wide, on="valid_time", how="inner")

        lags = _lag_features(merged["valid_time"], lead_hours, power_clean)
        merged = merged.merge(lags, on="valid_time", how="left")

        merged = _engineer_physical_features(merged)
        merged = _engineer_time_features(merged)
        merged["turbine_id"] = turbine_id

        n_total = len(merged)
        n_gap = int(merged["is_gap"].sum())
        n_curtailed = int((merged["curtailment_suspected"] & ~merged["is_gap"]).sum())
        clean = merged.loc[~merged["is_gap"] & ~merged["curtailment_suspected"]].drop(
            columns=["is_gap", "curtailment_suspected"]
        )
        n_missing_lag = int(clean["lag_power_t0"].isna().sum())

        stats["by_lead"][lead_hours].update({
            "n_total_hours": n_total,
            "n_excluded_gap": n_gap,
            "n_excluded_curtailment": n_curtailed,
            "n_kept": len(clean),
            "n_missing_lag_power_t0": n_missing_lag,
            "weather_source_counts": clean["weather_source"].value_counts().to_dict(),
            # Sanity check: wind-power correlation should be strong and should
            # DEGRADE as lead_hours grows (forecast uncertainty compounds).
            # A value near 0 or flat-across-lead here is the signature of a
            # timestamp/timezone misalignment between turbine and weather data
            # (this caught exactly that bug once already -- see
            # PLAN_AND_ARCHITECTURE.md).
            "wind_power_correlation": float(clean["wind_speed_100m"].corr(clean["power"])),
        })
        lead_frames.append(clean)

    dataset = pd.concat(lead_frames, ignore_index=True).sort_values(["valid_time", "lead_hours"]).reset_index(drop=True)
    dataset = dataset.rename(columns={"valid_time": "timestamp"})
    return dataset, stats

"""Resampling raw 10-minute turbine data to hourly bins, with gap and
curtailment flagging.

The forecasting target is hourly, but the raw logger data is 10-minute.
An hourly bin is only trusted if enough of its six 10-minute samples are
actually present; otherwise it is marked `is_gap` and excluded from
training targets (it is not silently interpolated across multi-day
outages).
"""

import pandas as pd

from . import schema


def detect_raw_gaps(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return every place where consecutive raw timestamps are not exactly
    RAW_FREQ_MINUTES apart, one row per gap, sorted by duration descending.
    """
    ts = df_raw["timestamp"].reset_index(drop=True)
    delta_min = ts.diff().dt.total_seconds() / 60
    mask = delta_min != schema.RAW_FREQ_MINUTES
    mask.iloc[0] = False  # first row has no predecessor
    gaps = pd.DataFrame(
        {
            "gap_start": ts.shift(1)[mask],
            "gap_end": ts[mask],
            "duration_minutes": delta_min[mask],
        }
    )
    gaps["duration_days"] = gaps["duration_minutes"] / (60 * 24)
    return gaps.sort_values("duration_minutes", ascending=False).reset_index(drop=True)


def resample_to_hourly(
    df_raw: pd.DataFrame,
    coverage_threshold: float = schema.HOURLY_COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate one turbine's 10-minute data to hourly mean values.

    Adds `n_obs` (how many 10-min samples fell in the hour), `coverage`
    (n_obs / 6) and `is_gap` (coverage below threshold -> untrustworthy bin).
    """
    turbine_id = df_raw["turbine_id"].iloc[0]
    indexed = df_raw.set_index("timestamp")

    hourly = indexed.resample("1h").agg(
        wind_speed=("wind_speed", "mean"),
        power=("power", "mean"),
        temperature=("temperature", "mean"),
        n_obs=("power", "count"),
    )
    hourly["coverage"] = hourly["n_obs"] / schema.SAMPLES_PER_HOUR
    hourly["is_gap"] = hourly["coverage"] < coverage_threshold
    hourly["turbine_id"] = turbine_id
    hourly = hourly.reset_index()
    return hourly[
        ["timestamp", "turbine_id", "wind_speed", "power", "temperature", "n_obs", "coverage", "is_gap"]
    ]


def flag_curtailment(
    df_hourly: pd.DataFrame,
    bin_width: float = 0.5,
    underperformance_ratio: float = 0.3,
    min_bin_count: int = 20,
) -> pd.DataFrame:
    """Flag hours where power is far below what the turbine's own empirical
    power curve would predict for that wind speed (curtailment, an outage
    that still logs data, or a sensor fault) -- kept, not dropped.
    """
    df = df_hourly.copy()
    valid = df.loc[~df["is_gap"]]

    wind_bin = (valid["wind_speed"] // bin_width) * bin_width
    bin_counts = wind_bin.value_counts()
    reliable_bins = bin_counts[bin_counts >= min_bin_count].index
    curve = valid.assign(wind_bin=wind_bin).loc[wind_bin.isin(reliable_bins)]
    power_curve = curve.groupby("wind_bin")["power"].median()

    df["wind_bin"] = (df["wind_speed"] // bin_width) * bin_width
    df["expected_power"] = df["wind_bin"].map(power_curve)
    df["curtailment_suspected"] = (
        (~df["is_gap"])
        & df["expected_power"].notna()
        & (df["expected_power"] > 0.1)
        & (df["power"] < underperformance_ratio * df["expected_power"])
    )
    return df.drop(columns=["wind_bin"])


def build_processed_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full per-turbine pipeline: resample to hourly, then flag curtailment."""
    hourly = resample_to_hourly(df_raw)
    return flag_curtailment(hourly)

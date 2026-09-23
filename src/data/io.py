"""Loading raw 10-minute turbine CSVs into a clean, uniform DataFrame."""

import pandas as pd

from . import schema


def load_turbine_raw(turbine_id: int) -> pd.DataFrame:
    """Load one turbine's raw CSV, rename columns, parse timestamps, sort and dedupe."""
    path = schema.RAW_FILES[turbine_id]
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns=schema.RAW_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["turbine_id"] = turbine_id
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    return df[schema.CLEAN_COLUMNS]


def load_all_raw() -> pd.DataFrame:
    """Load and concatenate all turbines' raw data."""
    frames = [load_turbine_raw(tid) for tid in schema.RAW_FILES]
    return pd.concat(frames, ignore_index=True)

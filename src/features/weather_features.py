"""Per-turbine weather feature retrieval for training-set assembly.

Combines the two zero-ambiguity sources from src/weather/client.py into a
single wide table per lead-time group:
  - previous_runs (fixed lead-time archived forecast) wherever it has
    coverage for the requested location;
  - the ERA5 reanalysis proxy for any earlier dates, explicitly tagged
    `weather_source="reanalysis_proxy"` so training code / README can
    single these rows out.
"""

from datetime import timedelta

import pandas as pd

from src.weather import client


def _wide_from_long(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a client long-format frame (valid_time, variable, value, source, lead_hours)
    into one row per valid_time with one column per variable, plus source/lead_hours.
    """
    wide = long_df.pivot(index="valid_time", columns="variable", values="value")
    meta = long_df.drop_duplicates("valid_time").set_index("valid_time")[["source", "lead_hours"]]
    wide = wide.join(meta).rename(columns={"source": "weather_source"})
    return wide.reset_index()


def get_weather_wide(
    latitude: float,
    longitude: float,
    lead_hours: int,
    start_date: str,
    end_date: str,
    coverage_start: str,
) -> pd.DataFrame:
    """Weather features for every hour in [start_date, end_date], as if a
    forecast for that hour had been issued `lead_hours` earlier.

    `coverage_start` (measured once via
    client.find_previous_runs_coverage_start) decides, per date, which of
    the two sources is used -- never mixed within the same date.
    """
    lead_days = lead_hours // 24
    frames = []

    if end_date >= coverage_start:
        pr_start = max(start_date, coverage_start)
        pr_long = client.fetch_previous_runs(latitude, longitude, pr_start, end_date, lead_days=(lead_days,))
        frames.append(pr_long)

    if start_date < coverage_start:
        proxy_end_date = (pd.Timestamp(coverage_start) - timedelta(days=1)).date().isoformat()
        proxy_end_date = min(proxy_end_date, end_date)
        proxy_long = client.fetch_historical_weather(latitude, longitude, start_date, proxy_end_date)
        proxy_long = proxy_long.copy()
        proxy_long["lead_hours"] = lead_hours  # relabel: this proxy is standing in for this lead group
        frames.append(proxy_long)

    long_df = pd.concat(frames, ignore_index=True)
    return _wide_from_long(long_df)

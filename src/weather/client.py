"""Open-Meteo weather client: previous-runs (fixed lead-time archived
forecasts), historical (ERA5 reanalysis, pre-archive proxy) and live
forecast, all going through the same cached, retrying HTTP call.

Every function returns a tidy "long" DataFrame:
    valid_time | variable | lead_hours | value | source | model | latitude | longitude
so the three sources can be concatenated and told apart downstream purely
by the `source` / `lead_hours` columns.
"""

import re
import time
from datetime import timedelta

import pandas as pd
import requests

from . import cache, schema


def _daterange_chunks(start_date: str, end_date: str, max_days: int = schema.MAX_REQUEST_DAYS):
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        yield chunk_start.isoformat(), chunk_end.isoformat()
        chunk_start = chunk_end + timedelta(days=1)


def _request(endpoint: str, params: dict) -> dict:
    cached = cache.load(schema.CACHE_DIR, endpoint, params)
    if cached is not None:
        return cached

    last_exc: Exception | None = None
    for attempt in range(1, schema.MAX_RETRIES + 1):
        try:
            resp = requests.get(endpoint, params=params, timeout=schema.REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            cache.save(schema.CACHE_DIR, endpoint, params, data)
            return data
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < schema.MAX_RETRIES:
                time.sleep(schema.RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"weather request failed after {schema.MAX_RETRIES} attempts: {endpoint} {params}") from last_exc


def _hourly_frame(response: dict) -> pd.DataFrame:
    hourly = response.get("hourly")
    if not hourly:
        raise ValueError(f"response has no 'hourly' block: {response}")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "valid_time"})


_LONG_COLUMNS = ["valid_time", "variable", "lead_hours", "value", "source", "model", "latitude", "longitude"]


def fetch_previous_runs(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    lead_days: tuple[int, ...] = schema.PREVIOUS_RUNS_LEAD_DAYS,
    model: str = schema.DEFAULT_MODEL,
) -> pd.DataFrame:
    """Fixed lead-time archived forecasts.

    For each valid hour, returns the value that was actually forecast
    `lead_days * 24` hours earlier -- the zero-leakage source for the
    Feb-2026 backtest and for training features from 2024-01-19 onward
    (empirically the first date with non-null coverage at these
    coordinates; older dates come back as NaN, not an error, so callers
    should check for gaps rather than assume a hardcoded cutover).
    """
    variables = variables or schema.BASE_VARIABLES
    hourly_vars = [f"{var}_previous_day{d}" for var in variables for d in lead_days]

    frames = []
    for chunk_start, chunk_end in _daterange_chunks(start_date, end_date):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(hourly_vars),
            "wind_speed_unit": "ms",
            "models": model,
        }
        data = _request(schema.PREVIOUS_RUNS_URL, params)
        frames.append(_hourly_frame(data))
    wide = pd.concat(frames, ignore_index=True).drop_duplicates(subset="valid_time")

    pattern = re.compile(r"^(?P<variable>.+)_previous_day(?P<lead_day>\d+)$")
    long_rows = []
    for col in wide.columns:
        if col == "valid_time":
            continue
        m = pattern.match(col)
        if not m:
            continue
        long_rows.append(
            wide[["valid_time", col]]
            .rename(columns={col: "value"})
            .assign(variable=m.group("variable"), lead_hours=int(m.group("lead_day")) * 24)
        )
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["source"] = "previous_runs"
    long_df["model"] = model
    long_df["latitude"] = latitude
    long_df["longitude"] = longitude
    return long_df[_LONG_COLUMNS]


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """ERA5 reanalysis (actual observed weather, not a forecast).

    Used only as a training-feature proxy for the slice of history before
    previous-runs coverage begins (2023-03-11 .. 2024-01-18). Rows built
    from this must be tagged downstream and never used for the backtest.
    """
    variables = variables or schema.BASE_VARIABLES

    frames = []
    for chunk_start, chunk_end in _daterange_chunks(start_date, end_date):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(variables),
            "wind_speed_unit": "ms",
        }
        data = _request(schema.ARCHIVE_URL, params)
        frames.append(_hourly_frame(data))
    wide = pd.concat(frames, ignore_index=True).drop_duplicates(subset="valid_time")

    long_df = wide.melt(id_vars="valid_time", var_name="variable", value_name="value")
    long_df["lead_hours"] = 0
    long_df["source"] = "reanalysis_proxy"
    long_df["model"] = "era5"
    long_df["latitude"] = latitude
    long_df["longitude"] = longitude
    return long_df[_LONG_COLUMNS]


def find_previous_runs_coverage_start(
    latitude: float,
    longitude: float,
    variable: str = "wind_speed_10m",
    lead_days: int = 1,
    search_start: str = "2015-01-01",
    search_end: str = "2026-01-01",
    model: str = schema.DEFAULT_MODEL,
) -> str | None:
    """Binary-search the earliest date with non-null previous-runs data for
    one variable/lead at these coordinates.

    Coverage start varies by location and model (docs only say "most
    models: from January 2024"), so this measures it directly instead of
    trusting a hardcoded date. Returns an ISO date string, or None if even
    `search_end` has no coverage yet.
    """

    def has_data(day: str) -> bool:
        df = fetch_previous_runs(latitude, longitude, day, day, variables=[variable], lead_days=(lead_days,), model=model)
        return df["value"].notna().any()

    if not has_data(search_end):
        return None

    lo = pd.Timestamp(search_start)
    hi = pd.Timestamp(search_end)
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) / 2
        mid_str = mid.normalize().date().isoformat()
        if has_data(mid_str):
            hi = mid
        else:
            lo = mid
    return hi.normalize().date().isoformat()


def find_full_coverage_start(
    latitude: float,
    longitude: float,
    lead_days: int,
    variables: list[str] | None = None,
    search_start: str = "2015-01-01",
    search_end: str = "2026-01-01",
    model: str = schema.DEFAULT_MODEL,
) -> str | None:
    """The first date from which EVERY requested variable is simultaneously
    populated for one lead, i.e. max(per-variable coverage start).

    Archive backfill is not uniform across variables -- e.g. at 43.65N
    78.54E, wind_speed_100m becomes available a full month after
    wind_speed_10m/temperature_2m/surface_pressure for the same lead.
    Using a single variable's cutover (as `find_previous_runs_coverage_start`
    does) would silently leave the other variables NULL for that gap; this
    is the version that should be used to pick the previous_runs/proxy
    split point in training-set assembly. Returns None if any variable has
    no coverage at all by `search_end`.
    """
    variables = variables or schema.BASE_VARIABLES
    starts = [
        find_previous_runs_coverage_start(
            latitude, longitude, variable=var, lead_days=lead_days, search_start=search_start, search_end=search_end, model=model
        )
        for var in variables
    ]
    if any(s is None for s in starts):
        return None
    return max(starts)


def fetch_live_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 2,
    variables: list[str] | None = None,
    model: str = schema.DEFAULT_MODEL,
) -> pd.DataFrame:
    """Live forward-looking forecast for production/real-time agent runs
    after the hackathon (not used for the backtest, which needs a fixed
    historical issue time rather than "now").
    """
    variables = variables or schema.BASE_VARIABLES
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "hourly": ",".join(variables),
        "wind_speed_unit": "ms",
        "models": model,
    }
    data = _request(schema.FORECAST_URL, params)
    wide = _hourly_frame(data)
    issue_time = pd.Timestamp.utcnow().tz_localize(None)

    long_df = wide.melt(id_vars="valid_time", var_name="variable", value_name="value")
    long_df["lead_hours"] = (long_df["valid_time"] - issue_time).dt.total_seconds() / 3600
    long_df["source"] = "live_forecast"
    long_df["model"] = model
    long_df["latitude"] = latitude
    long_df["longitude"] = longitude
    return long_df[_LONG_COLUMNS]

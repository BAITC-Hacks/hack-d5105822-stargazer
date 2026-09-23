"""Constants for the Open-Meteo-based weather client.

Three Open-Meteo endpoints are used, deliberately kept separate so the
"what did we actually know at time t" boundary in the agentic pipeline is
never ambiguous:

- PREVIOUS_RUNS_URL (`previous-runs-api`): for a given valid hour, exposes
  the value that was forecast a *fixed lead time* earlier
  (`<var>_previous_day1` = forecast issued ~24h before that hour,
  `_previous_day2` = ~48h before). This is the only source used for the
  mandatory backtest (issue a forecast on day d for d+1..d+2, using nothing
  that wasn't known on day d) and is the primary source for training
  features. Empirically verified for our coordinates (43.645N, 78.536E):
  data is NULL before 2024-01-19 and populated from 2024-01-19 onward
  (matches the docs' "most models archived from January 2024"). This date
  is a *measured fact about this location*, not a hardcoded assumption --
  `client.fetch_previous_runs` always checks for and reports NULLs rather
  than silently trusting any cutover date.

- ARCHIVE_URL (`archive-api`, ERA5 reanalysis): actual observed weather,
  available back to 1940. Used ONLY as a proxy feature source for the
  training slice before previous-runs coverage begins (2023-03-11 ->
  2024-01-18). This is a documented limitation (reanalysis is smoother and
  more accurate than a real 24-48h NWP forecast would have been) -- rows
  built from it are tagged `weather_source="reanalysis_proxy"` downstream
  and must never be used for the Feb-2026 backtest itself.

- FORECAST_URL (`api.open-meteo.com` live forecast): for real-time /
  production use of the agent after the hackathon, to get an actual
  forward-looking 24-48h forecast "as of now".
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "weather_cache"

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Base hourly variables requested from every source (Open-Meteo names).
BASE_VARIABLES = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "temperature_2m",
    "surface_pressure",
]

# previous-runs lead days to pull: day1 ~ 24h lead, day2 ~ 48h lead --
# matches the case's required 24-48h forecast horizon exactly.
PREVIOUS_RUNS_LEAD_DAYS = (1, 2)

DEFAULT_MODEL = "best_match"

# Chunk long date ranges into requests of at most this many days, to stay
# well under any undocumented per-request size limits and to keep cache
# entries small and independently reusable.
MAX_REQUEST_DAYS = 31

REQUEST_TIMEOUT_S = 20
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

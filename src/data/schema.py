"""Paths and column-mapping constants for the HackAlem VES turbine dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

RAW_FILES = {
    1: RAW_DATA_DIR / "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 1.csv",
    2: RAW_DATA_DIR / "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv",
}

# Raw CSV column name (Russian) -> internal name
RAW_COLUMNS = {
    "ID": "row_id",
    "Статистическое время": "timestamp",
    "Средняя скорость ветра(m/s)": "wind_speed",
    "Нормализованная активная мощность": "power",
    "Средняя температура окружающей среды(°C)": "temperature",
}

CLEAN_COLUMNS = ["timestamp", "turbine_id", "wind_speed", "power", "temperature"]

RAW_FREQ_MINUTES = 10
SAMPLES_PER_HOUR = 60 // RAW_FREQ_MINUTES

# An hourly bin needs at least this share of its 6 ten-minute samples present
# to be trusted as a real observation instead of a sensor/logger gap.
HOURLY_COVERAGE_THRESHOLD = 0.5

# The raw CSV timestamps are NOT UTC. The case gives no timezone, so this
# was determined empirically: cross-correlating the turbine's own
# wind_speed against ERA5 actual wind_speed_10m (src.weather.client,
# same coordinates) across the FULL 2023-03..2026-01 history shows the
# correlation peaking sharply at a +6h shift (r=0.717 vs r=0.712 at +5h,
# and <=0.65 at every other integer shift) -- i.e. raw_timestamp - 6h =
# true UTC instant. This matches Asia/Almaty standard time (UTC+6, the
# zone this site sat in before Kazakhstan's March-2024 unification to
# UTC+5 nationwide) and is stable before AND after that reform date, so
# it looks like a fixed logger convention rather than a real clock
# change. Everything is normalized to UTC at load time (src/data/io.py)
# so this constant is the ONLY place the assumption lives; revisit it
# immediately if the organizers ever state the true source timezone.
RAW_TIMESTAMP_UTC_OFFSET_HOURS = 6

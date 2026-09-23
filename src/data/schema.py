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

# History, per the case statement, ends 2026-01-31 23:50 inclusive.
HISTORY_END = "2026-01-31 23:50:00"

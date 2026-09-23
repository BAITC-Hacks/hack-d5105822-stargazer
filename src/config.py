"""Loading turbine metadata (coordinates) from config/turbines.yaml."""

from pathlib import Path
from typing import NamedTuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURBINES_CONFIG_PATH = PROJECT_ROOT / "config" / "turbines.yaml"


class TurbineInfo(NamedTuple):
    turbine_id: int
    name: str
    latitude: float
    longitude: float


def load_turbines() -> dict[int, TurbineInfo]:
    raw = yaml.safe_load(TURBINES_CONFIG_PATH.read_text(encoding="utf-8"))
    turbines = {}
    for tid, entry in raw["turbines"].items():
        tid = int(tid)
        turbines[tid] = TurbineInfo(
            turbine_id=tid,
            name=entry["name"],
            latitude=float(entry["latitude"]),
            longitude=float(entry["longitude"]),
        )
    return turbines

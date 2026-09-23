"""Disk cache for weather API responses, keyed by request parameters.

Every real HTTP call is cached to JSON on first fetch; reruns of any
pipeline (EDA, training-set assembly, backtest) hit the cache instead of
the network. This is required for reproducibility (README criterion) and
also keeps the agent well under Open-Meteo's free-tier rate limits.
"""

import hashlib
import json
from pathlib import Path


def _cache_key(endpoint: str, params: dict) -> str:
    blob = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _cache_path(cache_dir: Path, endpoint: str, params: dict) -> Path:
    source = endpoint.rstrip("/").split("/")[2].split(".")[0]  # e.g. "previous-runs-api"
    return cache_dir / source / f"{_cache_key(endpoint, params)}.json"


def load(cache_dir: Path, endpoint: str, params: dict) -> dict | None:
    path = _cache_path(cache_dir, endpoint, params)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save(cache_dir: Path, endpoint: str, params: dict, response: dict) -> Path:
    path = _cache_path(cache_dir, endpoint, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response), encoding="utf-8")
    return path

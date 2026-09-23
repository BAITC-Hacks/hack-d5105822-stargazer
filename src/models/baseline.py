"""Physical power-curve baseline.

Fits a smooth median power-vs-wind-speed curve from training data and uses
it as a sanity-check floor for the ML model: any model that cannot beat
"just look up the wind speed on our own historical curve" is not worth
deploying. Deliberately fit on the FORECAST wind speed column (not the
turbine's true simultaneous wind), since that is the only input actually
available at inference time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import schema


class PowerCurveBaseline:
    def __init__(self, bin_width: float = schema.POWER_CURVE_BIN_WIDTH):
        self.bin_width = bin_width
        self.curve_: pd.Series | None = None

    def fit(self, wind_speed: pd.Series, power: pd.Series) -> "PowerCurveBaseline":
        valid = pd.DataFrame({"wind_speed": wind_speed, "power": power}).dropna()
        binned = (valid["wind_speed"] // self.bin_width) * self.bin_width
        self.curve_ = valid.groupby(binned)["power"].median().sort_index()
        return self

    def predict(self, wind_speed: pd.Series) -> np.ndarray:
        if self.curve_ is None:
            raise RuntimeError("call fit() before predict()")
        # np.interp linearly interpolates between known bins and clamps to
        # the first/last known value outside the fitted range -- no NaNs,
        # no need to handle unseen bins separately.
        return np.interp(wind_speed.fillna(self.curve_.index.to_series().median()), self.curve_.index, self.curve_.values).clip(0, 1)

    def save(self, path: Path) -> None:
        payload = {"bin_width": self.bin_width, "curve": {str(k): v for k, v in self.curve_.items()}}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PowerCurveBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(bin_width=payload["bin_width"])
        curve = pd.Series({float(k): v for k, v in payload["curve"].items()}).sort_index()
        obj.curve_ = curve
        return obj

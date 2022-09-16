from dataclasses import dataclass
from typing import List


@dataclass
class TechnicalIndicators:
    ema200: float
    ema100: float
    atr: float
    atr_sma: float
    rsi: float
    rsi_sma: float
    adx: float
    macd: float
    macdsignal: float
    slowk_rsi: float
    slowd_rsi: float
    vo: float
    willy: float
    willy_ema: float

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]


class Assumptions:
    def __init__(self, n_candles_since_open: int, ti_vals: TechnicalIndicators, prediction: float) -> None:
        self.n_candles_since_open = float(n_candles_since_open)
        self.ti_vals = ti_vals
        self.prediction = prediction

    def create_aat_tuple(self) -> List[float]:
        return [self.n_candles_since_open] + self.ti_vals.get_values() + [self.prediction, self.prediction]



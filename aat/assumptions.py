from collections import deque
from dataclasses import dataclass
import pandas as pd
from typing import List
from utils.utils import USE_BOOL_VALS


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

    def bool_values(self) -> List[int]:
        up_trend = self.ema100 > self.ema200
        down_trend = self.ema100 < self.ema200
        atr_up_trend = self.atr > self.atr_sma
        atr_down_trend = self.atr < self.atr_sma
        rsi_up_trend = self.rsi > self.rsi_sma
        rsi_down_trend = self.rsi < self.rsi_sma
        rsi_buy = self.rsi > 50
        rsi_sell = self.rsi < 50
        adx_small = self.adx > 20
        adx_medium = self.adx > 25
        adx_large = self.adx > 30
        macd_up = min([self.macd, self.macdsignal, 0]) == 0
        macd_down = max([self.macd, self.macdsignal, 0]) == 0
        volume_small = self.vo > 0
        volume_medium = self.vo > 0.10
        volume_large = self.vo > 0.20
        willy_up_trend = self.willy > self.willy_ema
        willy_down_trend = self.willy < self.willy_ema

        return [up_trend, down_trend, atr_up_trend, atr_down_trend, rsi_up_trend, rsi_down_trend, rsi_buy, rsi_sell,
                adx_small, adx_medium, adx_large, macd_up, macd_down, volume_small, volume_medium, volume_large,
                willy_up_trend, willy_down_trend]


class Assumptions:
    def __init__(self, ti_vals: TechnicalIndicators, bid_open: float, ask_open: float, key_level: float,
                 prediction: float) -> None:
        self.ti_vals = ti_vals

        mid_open = (ask_open + bid_open) / 2

        self.level_distance = mid_open - key_level
        self.up_trend = ti_vals.ema100 - ti_vals.ema200
        self.spread = abs(ask_open - bid_open)
        self.spread_atr_percentage = self.spread / self.ti_vals.atr
        self.prediction = prediction

    def create_aat_tuple(self) -> List[float]:
        if USE_BOOL_VALS:
            return self.create_aat_bool_tuple()

        return self.ti_vals.get_values() + [self.level_distance, self.up_trend, self.spread,
                                            self.spread_atr_percentage, self.prediction]

    def create_aat_bool_tuple(self) -> List[float]:
        return self.ti_vals.bool_values() + [self.level_distance, self.up_trend, self.spread,
                                             self.spread_atr_percentage, self.prediction]

    def create_aat_tuple_from_names(self, names: List[str]) -> List[float]:
        tup = []
        ti_names = list(self.ti_vals.__annotations__.keys())
        other_names = [var_name for var_name in self.__dict__.keys() if var_name not in ['ti_vals', 'prediction']]

        for name in names:
            if name in ti_names:
                tup.append(self.ti_vals.__getattribute__(name))

            elif name in other_names:
                tup.append(self.__getattribute__(name))

            else:
                raise Exception(f'Unknown feature name: {name}')

        return tup

    def assumption_names(self) -> List[str]:
        ti_names = list(self.ti_vals.__annotations__.keys())
        other_names = [var_name for var_name in self.__dict__.keys() if var_name not in ['ti_vals', 'prediction']]

        return ti_names + other_names


class AssumptionsCollection:
    def __init__(self, lookback: int) -> None:
        self.collections = []
        self.lookback = lookback

    def update(self, new_assumptions: Assumptions) -> None:
        tup = new_assumptions.create_aat_bool_tuple() if USE_BOOL_VALS else new_assumptions.create_aat_tuple()
        tup = tup[:-1]

        for i, val in enumerate(tup):
            if i >= len(self.collections):
                self.collections.append(deque(maxlen=self.lookback))

            self.collections[i].append(val)

    def generate_moving_averages(self) -> List[float]:
        moving_averages = []

        for collection in self.collections:
            moving_average = list(pd.Series.ewm(pd.Series(collection), span=self.lookback).mean())[-1]
            moving_averages.append(moving_average)

        return moving_averages

from aat.assumptions import Assumptions, TechnicalIndicators
from aat.aat_market_trainer import CNN_LOOKBACK, grab_image_data
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.market_calculations import AMOUNT_TO_RISK
import numpy as np
from pandas import DataFrame
import pickle
from tensorflow.keras.models import load_model
from typing import Optional


class AatMarketTester:
    def __init__(self, risk_reward_ratio: float, testing_data_percentage: float) -> None:
        self.risk_reward_ratio = risk_reward_ratio
        if not 0.0 < testing_data_percentage < 1.0:
            raise Exception(f'Testing data percentage is not between 0 and 1: {testing_data_percentage}')
        self.testing_data_percentage = testing_data_percentage
        self.baseline = self.risk_reward_ratio * AMOUNT_TO_RISK

    def make_prediction(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> float:
        pass


class KnnAatMarketTester(AatMarketTester):
    def __init__(self, risk_reward_ratio: float, currency_pair: Optional[CurrencyPairs] = None,
                 testing_data_percentage: float = 0.3) -> None:
        AatMarketTester.__init__(self, risk_reward_ratio, testing_data_percentage)
        self.currency_pair = currency_pair
        self.scaler = None
        self.knn_model = None
        self.training_data = None

    def make_prediction(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> float:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, vo, willy, \
            willy_ema = market_data.loc[market_data.index[curr_idx], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi',
                                                                      'rsi_sma', 'adx', 'macd', 'macdsignal',
                                                                      'slowk_rsi', 'slowd_rsi', 'vo', 'willy',
                                                                      'willy_ema']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)
        prediction = self.baseline
        new_assumptions = Assumptions(n_candles, ti_vals, prediction)
        new_tup = new_assumptions.create_aat_tuple()

        if self.scaler is None:
            scaler_path = f'../aat/training_data/{self.currency_pair.value}_trained_knn_scaler_aat.pickle' if \
                self.currency_pair is not None else '../aat/training_data/trained_knn_scaler_aat.pickle'
            knn_path = f'../aat/training_data/{self.currency_pair.value}_trained_knn_aat.pickle' if \
                self.currency_pair is not None else '../aat/training_data/trained_knn_aat.pickle'
            data_path = f'../aat/training_data/{self.currency_pair.value}_training_data.pickle' if \
                self.currency_pair is not None else '../aat/training_data/training_data.pickle'

            self.scaler = pickle.load(open(scaler_path, 'rb'))
            self.knn_model = pickle.load(open(knn_path, 'rb'))
            self.training_data = np.array(pickle.load(open(data_path, 'rb')))

        x = np.array(new_tup[0:-2]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        neighbor_distances, neighbor_indices = self.knn_model.kneighbors(x_scaled, 15)

        corrections, distances = [], []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            corrections.append(self.training_data[neighbor_idx, -2])
            distances.append(neighbor_dist)

        trade_amount_pred, inverse_distance_sum = 0, 0

        for dist in distances:
            inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

        for i in range(len(corrections)):
            distance_i, cor = distances[i], corrections[i]
            inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
            distance_weight = inverse_distance_i / inverse_distance_sum

            trade_amount_pred += (self.baseline * cor * distance_weight)

        return trade_amount_pred


class CnnAatMarketTester(AatMarketTester):
    def __init__(self, risk_reward_ratio: float, currency_pair: Optional[CurrencyPairs] = None,
                 testing_data_percentage: float = 0.3) -> None:
        AatMarketTester.__init__(self, risk_reward_ratio, testing_data_percentage)
        self.currency_pair = currency_pair
        path = f'../aat/training_data/{currency_pair.value}_trained_cnn' if currency_pair is not None \
            else '../aat/training_data/trained_cnn'
        self.cnn_model = load_model(path)

    def make_prediction(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> float:
        if curr_idx < CNN_LOOKBACK:
            first_n = CNN_LOOKBACK - curr_idx
            curr_x = [market_data.iloc[0, :-2]] * first_n
            curr_x += list(market_data.iloc[0:curr_idx, 0:-2])

        else:
            curr_x = list(market_data.iloc[curr_idx - CNN_LOOKBACK:curr_idx, :-2])

        curr_x = grab_image_data(curr_x)

        trade_amount_pred = self.cnn_model.predict(curr_x.reshape(1, curr_x.shape[0], curr_x.shape[1],
                                                                  curr_x.shape[2]))[0]

        return trade_amount_pred

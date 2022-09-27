from aat.aat_market_trainer import AatMarketTrainer
from aat.assumptions import Assumptions, AssumptionsCollection, TechnicalIndicators
from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_calculations import AMOUNT_TO_RISK
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade
from nn.learner import Learner
import numpy as np
import pickle
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from typing import List, Optional, Tuple
from utils.utils import ESTIMATES_LOOKBACK, USE_BOOL_VALS


# TODO: Try AlegAATr without moving average assumptions
# TODO: Try just returning None if the best prediction is negative
# TODO: Try using ensemble of strategies
    # Todo: Return None if 1 or fewer predictions are positive (i.e. we need at least 2 positive predictions to execute a strategy)


class Alegaatr(Strategy):
    def __init__(self, starting_idx: int, experts: List[Strategy]):
        description = f'AlegAATr with {len(experts)} different strategies/experts'
        Strategy.__init__(self, 'AlegAATr', description, starting_idx)
        self.experts = experts
        self.models, self.scalers, self.training_datas = {}, {}, {}
        self.assumptions_collection = AssumptionsCollection(ESTIMATES_LOOKBACK)

    def _load_models(self, currency_pair: CurrencyPairs) -> None:
        for expert in self.experts:
            expert_name = expert.name
            data_dir = f'../aat/training_data'

            self.models[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{currency_pair.value}_{USE_BOOL_VALS}_trained_knn_aat.pickle', 'rb'))
            self.scalers[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{currency_pair.value}_{USE_BOOL_VALS}_trained_knn_scaler_aat.pickle', 'rb'))
            self.training_datas[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{currency_pair.value}_{USE_BOOL_VALS}_training_data.pickle', 'rb'))

    def _knn_prediction(self, x: List[float], expert_name: str) -> Tuple[List[float], List[float]]:
        model, scaler, training_data = \
            self.models[expert_name], self.scalers[expert_name], self.training_datas[expert_name]

        x = np.array(x[:-2]).reshape(1, -1)
        x_scaled = scaler.transform(x)
        neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, 15)
        corrections, distances = [], []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            corrections.append(training_data[neighbor_idx, -2])
            distances.append(neighbor_dist)

        return corrections, distances

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
        vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]
        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, 0)
        self.assumptions_collection.update(new_assumptions)
        tup = self.assumptions_collection.generate_moving_averages()

        predictions = {}

        for expert in self.experts:
            expert_name = expert.name
            baseline = AMOUNT_TO_RISK * expert.risk_reward_ratio
            corrections, distances = self._knn_prediction(tup, expert_name)

            total_pred, inverse_distance_sum = 0, 0

            for dist in distances:
                inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

            for i in range(len(corrections)):
                distance_i = distances[i]
                cor = corrections[i]
                inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
                distance_weight = inverse_distance_i / inverse_distance_sum
                total_pred += (baseline * cor * distance_weight)

            predictions[expert_name] = total_pred

        expert_key = max(predictions, key=lambda key: predictions[key])

        return self.experts[expert_key].place_trade(curr_idx, market_data)

    def run_strategy(self, currency_pair: CurrencyPairs, aat_trainer: Optional[AatMarketTrainer] = None,
                     learner: Optional[Learner] = None, date_range: str = '2018-2021') -> StrategyResults:
        self.currency_pair = currency_pair
        self._load_models(currency_pair)
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range)

        return MarketSimulator.run_simulation(self, market_data, aat_trainer, learner)

from aat.aat_market_trainer import AatMarketTrainer
from aat.assumptions import Assumptions, AssumptionsCollection, TechnicalIndicators
from collections import deque
from copy import deepcopy
from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_calculations import AMOUNT_TO_RISK
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade, TradeCalculations, TradeType
from nn.learner import Learner
import numpy as np
import pickle
import random
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from typing import List, Optional, Tuple
from utils.utils import USE_ALEGAATR_EMPIRICAL_REWARDS, EMPIRICAL_REWARDS_LOOKBACK, ESTIMATES_LOOKBACK, USE_BOOL_VALS


class Alegaatr(Strategy):
    def __init__(self, starting_idx: int, experts: List[Strategy], lmbda: float = 0.99, at_least_n_positive: int = 3):
        description = f'AlegAATr with {len(experts)} different strategies/experts'
        Strategy.__init__(self, 'AlegAATr', description, starting_idx)
        self.experts = {}

        for expert in experts:
            self.experts[expert.name] = deepcopy(expert)

        self.lmbda = lmbda
        self.at_least_n_positive = at_least_n_positive
        self.models, self.scalers, self.training_datas = {}, {}, {}
        self.assumptions_collection = AssumptionsCollection(ESTIMATES_LOOKBACK)
        self.expert_name = None
        self.empirical_rewards, self.n_rounds_since_played = {}, {}

        for expert_name in self.experts.keys():
            self.empirical_rewards[expert_name] = deque(maxlen=EMPIRICAL_REWARDS_LOOKBACK)
            self.n_rounds_since_played[expert_name] = 0

    def load_models(self) -> None:
        for expert_name in self.experts.keys():
            data_dir = f'../aat/training_data'

            self.models[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_trained_knn_aat.pickle', 'rb'))
            self.scalers[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_trained_knn_scaler_aat.pickle', 'rb'))
            self.training_datas[expert_name] = pickle.load(open(
                f'{data_dir}/{expert_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_training_data.pickle', 'rb'))

    def _knn_prediction(self, x: List[float], expert_name: str) -> Tuple[List[float], List[float]]:
        model, scaler, training_data = \
            self.models[expert_name], self.scalers[expert_name], self.training_datas[expert_name]

        x = np.array(x).reshape(1, -1)
        x_scaled = scaler.transform(x)
        neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, 15)
        corrections, distances = [], []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            corrections.append(training_data[neighbor_idx][-1])
            distances.append(neighbor_dist)

        return corrections, distances

    def trade_finished(self, net_profit: float) -> None:
        self.empirical_rewards[self.expert_name].append(net_profit)

    def stop_early(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Tuple[bool, float, float]:
        curr_bid_open, curr_bid_high, curr_bid_low, curr_ask_open, curr_ask_high, curr_ask_low, \
        curr_mid_open, curr_date = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Bid_High',
                                                                                 'Bid_Low', 'Ask_Open',
                                                                                 'Ask_High', 'Ask_Low',
                                                                                 'Mid_Open', 'Date']]
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
        vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, curr_bid_open, curr_ask_open, key_level, 0)
        tup = new_assumptions.create_aat_tuple()[:-1]

        expert = self.experts[self.expert_name]
        baseline = AMOUNT_TO_RISK * expert.risk_reward_ratio
        corrections, distances = self._knn_prediction(tup, self.expert_name)

        total_pred, inverse_distance_sum = 0, 0

        for dist in distances:
            inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

        for i in range(len(corrections)):
            distance_i = distances[i]
            cor = corrections[i]
            inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
            distance_weight = inverse_distance_i / inverse_distance_sum
            total_pred += (baseline * cor * distance_weight)

        trade_copy = deepcopy(trade)
        trade_copy.end_date = curr_date
        reward = ((curr_bid_open - trade.open_price) * trade.n_units) if trade.trade_type == TradeType.BUY else ((trade.open_price - curr_ask_open) * trade.n_units)
        day_fees = TradeCalculations.calculate_day_fees(trade_copy)

        if total_pred < reward + day_fees and reward + day_fees > 0:
            return True, reward, day_fees

        else:
            return False, 0, 0

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
        # self.assumptions_collection.update(new_assumptions)
        # tup = self.assumptions_collection.generate_moving_averages()
        tup = new_assumptions.create_aat_tuple()[:-1]

        predictions = {}

        for expert_name, expert in self.experts.items():
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

            if USE_ALEGAATR_EMPIRICAL_REWARDS and len(self.empirical_rewards[expert_name]) > 0:
                self.n_rounds_since_played[expert_name] += 1 if expert_name != self.expert_name else 0
                prob = self.lmbda ** self.n_rounds_since_played[expert_name]
                use_empricial_avgs = np.random.choice([1, 0], p=[prob, 1 - prob])

            else:
                use_empricial_avgs = False

            if use_empricial_avgs:
                avg_reward = np.array(self.empirical_rewards[expert_name]).mean()
                predictions[expert_name] = avg_reward

            else:
                predictions[expert_name] = total_pred

        # sorted_preds = list(predictions.items())
        # sorted_preds.sort(key=lambda x: x[1])
        # rewards = [pred[1] for pred in sorted_preds[:self.at_least_n_positive]]
        #
        # if not all([reward > 0 for reward in rewards]):
        #     return None
        #
        # names = [pred[0] for pred in sorted_preds[:self.at_least_n_positive]]
        # reward_sum = sum(rewards)
        # weights = [reward / reward_sum for reward in rewards]
        #
        # expert_key = random.choices(names, weights=weights, k=1)[0]

        expert_key = max(predictions, key=lambda key: predictions[key])
        sorted_preds = sorted(predictions.values(), reverse=True)

        if not all([pred > 0 for pred in sorted_preds[:self.at_least_n_positive]]):
            return None

        self.expert_name = expert_key
        self.n_rounds_since_played[self.expert_name] = 0

        return self.experts[self.expert_name].place_trade(curr_idx, market_data)

    def run_strategy(self, currency_pair: CurrencyPairs, aat_trainer: Optional[AatMarketTrainer] = None,
                     learner: Optional[Learner] = None, date_range: str = '2018-2021') -> StrategyResults:
        self.currency_pair = currency_pair

        for expert in self.experts.values():
            expert.currency_pair = self.currency_pair
            expert.load_models()

        self.load_models()
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range)

        return MarketSimulator.run_simulation(self, market_data, aat_trainer, learner)

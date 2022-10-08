from aat.aat_market_trainer import AatMarketTrainer
from copy import deepcopy
from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade
from nn.learner import Learner
import numpy as np
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from typing import List, Optional


class UCB(Strategy):
    def __init__(self, starting_idx: int, experts: List[Strategy], delta: float = 0.99):
        description = f'UCB with {len(experts)} different strategies/experts'
        Strategy.__init__(self, 'UCB', description, starting_idx)
        self.experts, self.empirical_rewards, self.n_samples = {}, {}, {}

        for expert in experts:
            self.experts[expert.name] = deepcopy(expert)
            self.empirical_rewards[expert.name] = 0
            self.n_samples[expert.name] = 0

        self.delta = delta
        self.expert_name = None

    def trade_finished(self, net_profit: float) -> None:
        self.empirical_rewards[self.expert_name] += net_profit

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        predictions = {}

        for expert_name in self.experts.keys():
            n_samples = self.n_samples[expert_name]

            if n_samples == 0:
                predictions[expert_name] = np.inf

            else:
                empirical_avg = self.empirical_rewards[expert_name] / n_samples
                upper_bound = ((2 * np.log(1 / self.delta)) / n_samples) ** 0.5
                predictions[expert_name] = empirical_avg + upper_bound

        self.expert_name = max(predictions, key=lambda key: predictions[key])
        self.n_samples[self.expert_name] += 1

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


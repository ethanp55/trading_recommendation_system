from aat.aat_market_trainer import AatMarketTrainer
from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import Trade
from nn.learner import Learner
from strategy.strategy_results import StrategyResults
from typing import Optional, Tuple


class Strategy:
    def __init__(self, name: str, description: str, starting_idx: int) -> None:
        self.name = name
        self.description = description
        self.starting_idx = starting_idx
        self.currency_pair = None

    def stop_early(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Tuple[bool, float, float]:
        return False, 0, 0

    def load_models(self) -> None:
        pass

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        pass

    def run_strategy(self, currency_pair: CurrencyPairs, aat_trainer: Optional[AatMarketTrainer] = None,
                     learner: Optional[Learner] = None, date_range: str = '2018-2021') -> StrategyResults:
        pass

    def trade_finished(self, net_profit: float) -> None:
        pass

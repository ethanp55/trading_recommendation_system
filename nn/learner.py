from datetime import datetime
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.trades import TradeType


class Learner:
    def __init__(self, training_data_percentage: float, currency_pair: CurrencyPairs) -> None:
        if not 0.0 < training_data_percentage < 1.0:
            raise Exception(f'Training data percentage is not between 0 and 1: {training_data_percentage}')

        self.training_data_percentage = training_data_percentage
        self.currency_pair = currency_pair
        self.buy_dates, self.sell_dates, self.no_trade_dates = [], [], []
        self.market_data = None

    def trade_finished(self, net_profit: float, start_date: datetime, trade_type: TradeType) -> None:
        pass

    def save_data(self) -> None:
        pass

    def train(self) -> None:
        pass

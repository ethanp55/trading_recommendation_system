from aat.aat_market_trainer import AatMarketTrainer
from nn.cnn_utilities import CNN_LOOKBACK, grab_image_data
from datetime import datetime
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade, TradeType
from nn.learner import Learner
import numpy as np
from pandas import DataFrame
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from tensorflow.keras.models import load_model
from typing import Optional


class CnnStrategy(Strategy):
    def __init__(self, starting_idx: int, risk_reward_ratio: float, proba_threshold: float, lookback: int,
                 spread_cutoff: float) -> None:
        description = f'CNN strategy with {risk_reward_ratio} risk/reward, {spread_cutoff} spread ' \
                      f'ratio, stop loss lookback of {lookback}, proba threshold of {proba_threshold}'
        Strategy.__init__(self, 'cnn_strategy', description, starting_idx)

        if not 0.0 <= proba_threshold <= 1.0:
            raise Exception(f'Probability threshold for predictions is not between 0 and 1: {proba_threshold}')

        self.risk_reward_ratio = risk_reward_ratio
        self.proba_threshold = proba_threshold
        self.lookback = lookback
        self.spread_cutoff = spread_cutoff
        self.currency_pair = CurrencyPairs.EUR_USD
        self.model = None

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        # Use CNN to make a prediction
        curr_slice = market_data.drop(['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High',
                                       'Ask_Low', 'Ask_Close', 'Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close',
                                       'Volume', 'Date'], axis=1, inplace=False)
        curr_slice = curr_slice.iloc[curr_idx - CNN_LOOKBACK:curr_idx, :]
        curr_slice = grab_image_data(curr_slice)

        pred = self.model(curr_slice.reshape(1, curr_slice.shape[0], curr_slice.shape[1], curr_slice.shape[2]))
        argmax = np.argmax(pred)
        proba = pred[0][argmax].numpy()

        cnn_buy_signal = argmax == TradeType.BUY.value and proba >= self.proba_threshold
        cnn_sell_signal = argmax == TradeType.SELL.value and proba >= self.proba_threshold

        trade = None

        if cnn_buy_signal or cnn_sell_signal:
            # Get needed prices for placing a trade
            curr_bid_open, curr_ask_open, curr_mid_open, curr_date = \
                market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open', 'Mid_Open', 'Date']]

            spread = abs(curr_ask_open - curr_bid_open)

            mid_highs = list(
                market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_High'])
            mid_lows = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Low'])

            highest_high, lowest_low = max(mid_highs), min(mid_lows)

            # Try to place a trade (if the levels are set properly and the spread is small enough)
            if cnn_buy_signal:
                open_price = curr_ask_open
                stop_loss = lowest_low

                if stop_loss < open_price:
                    curr_pips_to_risk = open_price - stop_loss

                    if spread <= curr_pips_to_risk * self.spread_cutoff:
                        stop_gain = open_price + (self.risk_reward_ratio * curr_pips_to_risk)
                        trade_type = TradeType.BUY
                        n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ask_open, curr_bid_open,
                                                                 curr_mid_open, self.currency_pair)

                        trade = Trade(trade_type, open_price, stop_loss, stop_gain, n_units, n_units, curr_pips_to_risk,
                                      curr_date, None)

            elif cnn_sell_signal:
                open_price = float(curr_bid_open)
                stop_loss = highest_high

                if stop_loss > open_price:
                    curr_pips_to_risk = stop_loss - open_price

                    if spread <= curr_pips_to_risk * self.spread_cutoff:
                        stop_gain = open_price - (self.risk_reward_ratio * curr_pips_to_risk)
                        trade_type = TradeType.SELL
                        n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ask_open, curr_bid_open,
                                                                 curr_mid_open, self.currency_pair)

                        trade = Trade(trade_type, open_price, stop_loss, stop_gain, n_units, n_units, curr_pips_to_risk,
                                      curr_date, None)

        return trade

    def load_models(self) -> None:
        self.model = load_model(f'../nn/training_data/{self.currency_pair.value}_trained_cnn')

    def run_strategy(self, currency_pair: CurrencyPairs, aat_trainer: Optional[AatMarketTrainer] = None,
                     learner: Optional[Learner] = None, date_range: str = '2018-2021') -> StrategyResults:
        self.currency_pair = currency_pair
        self.load_models()
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range)

        return MarketSimulator.run_simulation(self, market_data, aat_trainer, learner)

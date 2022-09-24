from aat.aat_market_trainer import AatMarketTrainer
from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade, TradeType
from nn.learner import Learner
from typing import Optional
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults


class StochMACDCrossStrategy(Strategy):
    def __init__(self, starting_idx: int, risk_reward_ratio: float, spread_cutoff: float, lookback: int):
        description = f'Stochastic Crossover w/MACD strategy with {risk_reward_ratio} risk/reward, {spread_cutoff} ' \
                      f'spread ratio, {lookback} lookback'
        Strategy.__init__(self, 'stoch_macd_strategy', description, starting_idx)
        self.risk_reward_ratio = risk_reward_ratio
        self.spread_cutoff = spread_cutoff
        self.lookback = lookback
        self.starting_idx = self.lookback if self.starting_idx < self.lookback else self.starting_idx

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        slowk2, slowd2 = market_data.loc[market_data.index[curr_idx - 2], ['slowk_rsi', 'slowd_rsi']]
        slowk1, slowd1, ema200_1, ema100_1 = market_data.loc[market_data.index[curr_idx - 1], ['slowk_rsi', 'slowd_rsi', 'ema200', 'ema100']]
        stoch_vals = [slowk2, slowd2, slowk1, slowd1]
        macd_vals = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'macd'])
        macdsignal_vals = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'macdsignal'])
        curr_bid_open, curr_ask_open, curr_mid_open, curr_date = market_data.loc[market_data.index[curr_idx],
                                                                                 ['Bid_Open', 'Ask_Open', 'Mid_Open',
                                                                                  'Date']]
        spread = abs(curr_ask_open - curr_bid_open)

        macd_buy_cross, macd_sell_cross = False, False

        assert len(macd_vals) == len(macdsignal_vals)

        for j in range(1, len(macd_vals)):
            macd2, macdsignal2 = macd_vals[j - 1], macdsignal_vals[j - 1]
            macd1, macdsignal1 = macd_vals[j], macdsignal_vals[j]

            if macd2 < macdsignal2 and macd1 > macdsignal1:
                macd_buy_cross = True

            elif macd2 > macdsignal2 and macd1 < macdsignal1:
                macd_sell_cross = True

        buy_signal = slowk2 < slowd2 and slowk1 > slowd1 and max(stoch_vals + [20]) == 20 and macd_buy_cross
        sell_signal = slowk2 > slowd2 and slowk1 < slowd1 and min(stoch_vals + [80]) == 80 and macd_sell_cross

        mid_highs = list(
            market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_High'])
        mid_lows = list(
            market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Low'])

        highest_high, lowest_low = max(mid_highs), min(mid_lows)

        trade = None

        if buy_signal:
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

        elif sell_signal:
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

    def run_strategy(self, currency_pair: CurrencyPairs, aat_trainer: Optional[AatMarketTrainer] = None,
                     learner: Optional[Learner] = None, date_range: str = '2018-2021') -> StrategyResults:
        self.currency_pair = currency_pair
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range)

        return MarketSimulator.run_simulation(self, market_data, aat_trainer, learner)

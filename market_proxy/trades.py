from dataclasses import dataclass
from datetime import datetime
import enum
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
from typing import Optional


AMOUNTS_PER_DAY = [-0.00008, -0.0001, -0.00012]
AMOUNT_TO_RISK = 50.0


class TradeType(enum.Enum):
    NONE = 0
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    trade_type: TradeType
    open_price: float
    stop_loss: float
    stop_gain: float
    n_units: int
    original_units: int
    pips_risked: float
    start_date: datetime
    end_date: Optional[datetime] = None
    reward: Optional[float] = None
    day_fees: Optional[float] = None
    net_profit: Optional[float] = None

    def calculate_trade(self, curr_bid_low: float, curr_bid_high: float, curr_ask_low: float,
                        curr_ask_high: float, curr_date: datetime) -> None:
        if self.end_date is not None:
            raise Exception('Trade is already closed out')

        # Condition 1 - trade is a buy and the stop loss is hit
        if self.trade_type == TradeType.BUY and curr_bid_low <= self.stop_loss:
            self.end_date = curr_date
            trade_amount = (self.stop_loss - self.open_price) * self.n_units
            self.reward = trade_amount
            fees = TradeCalculations.calculate_day_fees(self)
            self.day_fees = fees
            self.net_profit = trade_amount + fees

        # Condition 2 - Trade is a buy and the take profit/stop gain is hit
        elif self.trade_type == TradeType.BUY and curr_bid_high >= self.stop_gain:
            self.end_date = curr_date
            trade_amount = (self.stop_gain - self.open_price) * self.n_units
            self.reward = trade_amount
            fees = TradeCalculations.calculate_day_fees(self)
            self.day_fees = fees
            self.net_profit = trade_amount + fees

        # Condition 3 - trade is a sell and the stop loss is hit
        elif self.trade_type == TradeType.SELL and curr_ask_high >= self.stop_loss:
            self.end_date = curr_date
            trade_amount = (self.open_price - self.stop_loss) * self.n_units
            self.reward = trade_amount
            fees = TradeCalculations.calculate_day_fees(self)
            self.day_fees = fees
            self.net_profit = trade_amount + fees

        # Condition 4 - Trade is a sell and the take profit/stop gain is hit
        elif self.trade_type == TradeType.SELL and curr_ask_low <= self.stop_gain:
            self.end_date = curr_date
            trade_amount = (self.open_price - self.stop_gain) * self.n_units
            self.reward = trade_amount
            fees = TradeCalculations.calculate_day_fees(self)
            self.day_fees = fees
            self.net_profit = trade_amount + fees


class TradeCalculations(object):
    @staticmethod
    def calculate_day_fees(trade: Trade) -> float:
        start_date, end_date, n_units = trade.start_date, trade.end_date, trade.original_units
        curr_fee = np.random.choice(AMOUNTS_PER_DAY, p=[0.25, 0.50, 0.25]) * n_units
        num_days = np.busday_count(start_date.date(), end_date.date())

        return num_days * curr_fee

    @staticmethod
    def get_n_units(trade_type: TradeType, stop_loss: float, ask_open: float, bid_open: float, mid_open: float,
                    currency_pair: CurrencyPairs) -> int:
        _, second = currency_pair.value.split('_')

        pips_to_risk = ask_open - stop_loss if trade_type == TradeType.BUY else stop_loss - bid_open
        pips_to_risk_calc = pips_to_risk * 10000 if second != 'Jpy' else pips_to_risk * 100

        if second == 'Usd':
            per_pip = 0.0001

        else:
            per_pip = 0.0001 / mid_open if second != 'Jpy' else 0.01 / mid_open

        n_units = int(AMOUNT_TO_RISK / (pips_to_risk_calc * per_pip))

        if second == 'Jpy':
            n_units /= 100

        return n_units

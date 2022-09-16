from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType, Trade
import numpy as np

AMOUNTS_PER_DAY = [-0.00008, -0.0001, -0.00012]
AMOUNT_TO_RISK = 50.0


class MarketCalculations(object):
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


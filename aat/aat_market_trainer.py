from aat.assumptions import Assumptions, TechnicalIndicators
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.market_calculations import AMOUNT_TO_RISK
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from utils.utils import USE_BOOL_VALS


class AatMarketTrainer:
    def __init__(self, currency_pair: CurrencyPairs, risk_reward_ratio: float, strategy_name: str) -> None:
        self.currency_pair = currency_pair
        self.risk_reward_ratio = risk_reward_ratio
        self.strategy_name = strategy_name
        self.curr_trade_data = []
        self.training_data = []

    def record_tuple(self, curr_idx: int, market_data: DataFrame) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
        vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]
        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, AMOUNT_TO_RISK * self.risk_reward_ratio)
        new_tup = new_assumptions.create_aat_tuple()

        self.curr_trade_data.append(new_tup)

    def trade_finished(self, trade_amount: float) -> None:
        for tup in self.curr_trade_data:
            tup[-1] = trade_amount / tup[-1]

        self.training_data.extend(self.curr_trade_data)
        self.curr_trade_data.clear()

    def save_data(self) -> None:
        self._data_dir = '../aat/training_data'

        file_path = f'{self._data_dir}/{self.strategy_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_training_data.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(self.training_data, f)


class AatKnnMarketTrainer(AatMarketTrainer):
    def __init__(self, currency_pair: CurrencyPairs, risk_reward_ratio: float, strategy_name: str) -> None:
        AatMarketTrainer.__init__(self, currency_pair, risk_reward_ratio, strategy_name)

    def save_data(self) -> None:
        AatMarketTrainer.save_data(self)

        print(len(self.training_data))

        x = np.array(self.training_data)[:, 0:-1]
        y = np.array(self.training_data)[:, -1]

        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = NearestNeighbors(n_neighbors=15)
        model.fit(x_scaled)

        trained_knn_file = f'{self.strategy_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_trained_knn_aat.pickle'
        trained_knn_scaler_file = f'{self.strategy_name}_{self.currency_pair.value}_{USE_BOOL_VALS}_trained_knn_scaler_aat.pickle'

        with open(f'{self._data_dir}/{trained_knn_file}', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{self._data_dir}/{trained_knn_scaler_file}', 'wb') as f:
            pickle.dump(scaler, f)
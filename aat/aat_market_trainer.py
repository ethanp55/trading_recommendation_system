from aat.assumptions import Assumptions, TechnicalIndicators
from market_proxy.market_calculations import AMOUNT_TO_RISK
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
from pandas import DataFrame
import pickle
from pyts.image import GramianAngularField
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from typing import Optional


CNN_LOOKBACK = 12


class AatMarketTrainer:
    def __init__(self, risk_reward_ratio: float, training_data_percentage: float) -> None:
        self.risk_reward_ratio = risk_reward_ratio
        if not 0.0 < training_data_percentage < 1.0:
            raise Exception(f'Training data percentage is not between 0 and 1: {training_data_percentage}')
        self.training_data_percentage = training_data_percentage
        self.baseline = self.risk_reward_ratio * AMOUNT_TO_RISK

    def record_tuple(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> None:
        pass

    def trade_finished(self, net_profit: float) -> None:
        pass

    def save_data(self, currency_pair: Optional[CurrencyPairs] = None) -> None:
        pass


class KnnAatMarketTrainer(AatMarketTrainer):
    def __init__(self, risk_reward_ratio: float, training_data_percentage: float = 0.7) -> None:
        AatMarketTrainer.__init__(self, risk_reward_ratio, training_data_percentage)
        self.training_data = []
        self.curr_trade_data = []

    def record_tuple(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, vo, willy, \
            willy_ema = market_data.loc[market_data.index[curr_idx], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi',
                                                                      'rsi_sma', 'adx', 'macd', 'macdsignal',
                                                                      'slowk_rsi', 'slowd_rsi', 'vo', 'willy',
                                                                      'willy_ema']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)
        prediction = self.baseline
        new_assumptions = Assumptions(n_candles, ti_vals, prediction)
        new_tup = new_assumptions.create_aat_tuple()

        self.curr_trade_data.append(new_tup)

    def trade_finished(self, net_profit: float) -> None:
        for tup in self.curr_trade_data:
            tup[-1] = net_profit
            tup[-2] = net_profit / tup[-2]

        self.training_data.extend(self.curr_trade_data)
        self.curr_trade_data.clear()

    def save_data(self, currency_pair: Optional[CurrencyPairs] = None) -> None:
        data_dir = '../aat/training_data'

        file_path = f'{data_dir}/{currency_pair.value}_training_data.pickle' if currency_pair is not None else \
            f'{data_dir}/training_data.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(self.training_data, f)

        x = np.array(self.training_data)[:, 0:-2]
        y = np.array(self.training_data)[:, -2]

        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = NearestNeighbors(n_neighbors=15)
        model.fit(x_scaled)

        trained_knn_file = f'{currency_pair.value}_trained_knn_aat.pickle' if currency_pair is not None else \
            'trained_knn_aat.pickle'
        trained_knn_scaler_file = f'{currency_pair.value}_trained_knn_scaler_aat.pickle' if currency_pair is not None \
            else 'trained_knn_scaler_aat.pickle'

        with open(f'{data_dir}/{trained_knn_file}', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{data_dir}/{trained_knn_scaler_file}', 'wb') as f:
            pickle.dump(scaler, f)


def grab_image_data(subset):
    gasf_transformer = GramianAngularField(method='summation')
    gasf_subset = gasf_transformer.transform(subset)

    return gasf_subset


class CnnAatMarketTrainer(AatMarketTrainer):
    def __init__(self, risk_reward_ratio: float, training_data_percentage: float = 0.7) -> None:
        AatMarketTrainer.__init__(self, risk_reward_ratio, training_data_percentage)
        self.training_data = []
        self.curr_trade_data = []

    def record_tuple(self, curr_idx: int, n_candles: int, market_data: DataFrame) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, vo, willy, \
        willy_ema = market_data.loc[market_data.index[curr_idx], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi',
                                                                  'rsi_sma', 'adx', 'macd', 'macdsignal',
                                                                  'slowk_rsi', 'slowd_rsi', 'vo', 'willy',
                                                                  'willy_ema']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)
        prediction = self.baseline
        new_assumptions = Assumptions(n_candles, ti_vals, prediction)
        new_tup = new_assumptions.create_aat_tuple()[1:]

        self.curr_trade_data.append(new_tup)

    def trade_finished(self, net_profit: float) -> None:
        for i in range(len(self.curr_trade_data)):
            self.curr_trade_data[i][-1] = net_profit
            self.curr_trade_data[i][-2] = net_profit / self.curr_trade_data[i][-2]

            if i < CNN_LOOKBACK - 1:
                first_n = CNN_LOOKBACK - 1 - i
                curr_slice = [self.curr_trade_data[0] for _ in range(first_n)]
                curr_slice.extend([self.curr_trade_data[j] for j in range(i + 1)])

            else:
                curr_slice = [self.curr_trade_data[j] for j in range(i - (CNN_LOOKBACK - 1), i + 1)]

            self.training_data.append(curr_slice)

        self.curr_trade_data.clear()

    def save_data(self, currency_pair: Optional[CurrencyPairs] = None) -> None:
        data_dir = '../aat/training_data'

        file_path = f'{data_dir}/{currency_pair.value}_training_data_cnn.pickle' if currency_pair is not None else \
            f'{data_dir}/training_data_cnn.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(self.training_data, f)

        x = np.array(self.training_data)

        print(x.shape)

    def fit_cnn(self, currency_pair: Optional[CurrencyPairs] = None) -> None:
        print('Reading in data...')

        data_path = f'../aat/training_data/{currency_pair.value}_training_data_cnn.pickle' if \
            currency_pair is not None else '../aat/training_data/training_data_cnn.pickle'

        training_data = np.array(pickle.load(open(data_path, 'rb')))

        print(training_data.shape)

        print('Generating labelled data...')

        data = []

        for interaction in training_data:
            curr_x = list(interaction[:, :-2])

            data.append([grab_image_data(interaction), interaction[0][-2]])

        np.random.shuffle(data)

        training_proportion = 0.70
        train_test_cutoff_index = int(len(data) * training_proportion)

        train_set = data[0:train_test_cutoff_index]
        test_set = data[train_test_cutoff_index:]

        x_train, y_train, x_test, y_test = [], [], [], []

        for seq, target in train_set:
            x_train.append(seq)
            y_train.append(target)

        for seq, target in test_set:
            x_test.append(seq)
            y_test.append(target)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        input_data_shape = x_train.shape[1:]

        model = Sequential()

        model.add(
            Conv2D(filters=16, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=input_data_shape))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        n_epochs = 1000
        batch_size = 64
        patience_percentage = 0.20

        path = f'../aat/training_data/{currency_pair.value}_trained_cnn' if currency_pair is not None \
            else '../aat/training_data/trained_cnn'

        early_stop = EarlyStopping(monitor='val_mean_squared_error', verbose=1,
                                   patience=int(patience_percentage * n_epochs))
        model_checkpoint = ModelCheckpoint(path, monitor='val_mean_squared_error', save_best_only=True, verbose=1)

        optimizer = Adam()

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, model_checkpoint]
        )

from nn.rnn_utilities import RNN_LOOKBACK
from copy import deepcopy
from datetime import datetime
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
from nn.learner import Learner
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential


class RnnMarketTrainer(Learner):
    def __init__(self, training_data_percentage: float, currency_pair: CurrencyPairs) -> None:
        Learner.__init__(self, training_data_percentage, currency_pair)

    def trade_finished(self, net_profit: float, start_date: datetime, trade_type: TradeType) -> None:
        if net_profit <= 0:
            self.no_trade_dates.append(start_date)

        elif trade_type == TradeType.BUY:
            self.buy_dates.append(start_date)

        else:
            self.sell_dates.append(start_date)

    def save_data(self) -> None:
        df = deepcopy(self.market_data)

        buy_indices = [df.index[df['Date'] == curr_date] - 1 for curr_date in self.buy_dates]
        sell_indices = [df.index[df['Date'] == curr_date] - 1 for curr_date in self.sell_dates]
        nones_indices = [df.index[df['Date'] == curr_date] - 1 for curr_date in self.no_trade_dates]

        df.drop(['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High', 'Ask_Low', 'Ask_Close',
                 'Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume', 'Date'], axis=1, inplace=True)

        scaler = StandardScaler()
        df = scaler.fit_transform(df)

        print(len(buy_indices))
        print(len(sell_indices))
        print(len(nones_indices))

        # This should be the correct shape of the data being passed to the RNN
        i = len(df) - 2
        seq = df[i - RNN_LOOKBACK + 1:i + 1, :]
        correct_shape = seq.shape

        print(correct_shape)

        no_actions, buys, sells = [], [], []

        for i in buy_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i - RNN_LOOKBACK + 1:i + 1, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    buys.append([seq, np.array([0, 1, 0])])

        for i in sell_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i - RNN_LOOKBACK + 1:i + 1, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    sells.append([seq, np.array([0, 0, 1])])

        for i in nones_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i - RNN_LOOKBACK + 1:i + 1, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    no_actions.append([seq, np.array([1, 0, 0])])

        np.random.shuffle(no_actions)
        np.random.shuffle(buys)
        np.random.shuffle(sells)

        # Make sure the classes are balanced
        lower = min([len(no_actions), len(buys), len(sells)])
        buys, sells, no_actions = buys[:lower], sells[:lower], no_actions[:lower]

        training_data = no_actions + buys + sells
        np.random.shuffle(training_data)

        data_dir = '../nn/training_data'

        file_path = f'{data_dir}/{self.currency_pair.value}_training_data_rnn.pickle'
        scaler_file_path = f'{data_dir}/{self.currency_pair.value}_trained_rnn_scaler.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(training_data, f)

        with open(scaler_file_path, 'wb') as f:
            pickle.dump(scaler, f)

    def train(self) -> None:
        data_path = f'../nn/training_data/{self.currency_pair.value}_training_data_rnn.pickle'

        training_data = np.array(pickle.load(open(data_path, 'rb')))

        train_test_cutoff_index = int(len(training_data) * self.training_data_percentage)

        train_set = training_data[0:train_test_cutoff_index]
        test_set = training_data[train_test_cutoff_index:]

        print('Dataset shapes:')
        print(len(train_set))
        print(len(test_set))

        x_train = []
        y_train = []

        for seq, target in train_set:
            x_train.append(seq)
            y_train.append(target)

        x_test = []
        y_test = []

        for seq, target in test_set:
            x_test.append(seq)
            y_test.append(target)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print(x_train.shape)
        print(y_train.shape)

        input_data_shape = x_train.shape[1:]
        n_actions = 3  # Buy, Sell, Do nothing

        model = Sequential()

        model.add(GRU(256, return_sequences=True, input_shape=input_data_shape))
        model.add(BatchNormalization())

        model.add(GRU(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(GRU(256, return_sequences=True))
        model.add(BatchNormalization())

        model.add(GRU(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(GRU(256))
        model.add(BatchNormalization())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(n_actions, activation='softmax'))

        n_epochs = 100
        batch_size = 32
        patience_percentage = 0.2

        path = f'../nn/training_data/{self.currency_pair.value}_trained_rnn'

        early_stop = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=int(patience_percentage * n_epochs))
        model_checkpoint = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, verbose=1)

        optimizer = Adam()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, model_checkpoint]
        )

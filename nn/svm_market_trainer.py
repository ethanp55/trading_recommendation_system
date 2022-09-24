from copy import deepcopy
from datetime import datetime
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
from nn.learner import Learner
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMMarketTrainer(Learner):
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
        print('Saving SVM data')
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

        # This should be the correct shape of the data being passed to the SVM
        correct_shape = df[0, :].shape

        print(correct_shape)

        no_actions, buys, sells = [], [], []

        for i in buy_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    buys.append([seq, 'buy'])

        for i in sell_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    sells.append([seq, 'sell'])

        for i in nones_indices:
            if len(i) == 1:
                i = i[0]
                seq = df[i, :]

                if seq.shape == correct_shape and not np.isnan(seq).any():
                    no_actions.append([seq, 'none'])

        np.random.shuffle(no_actions)
        np.random.shuffle(buys)
        np.random.shuffle(sells)

        # Make sure the classes are balanced
        lower = min([len(no_actions), len(buys), len(sells)])
        buys, sells, no_actions = buys[:lower], sells[:lower], no_actions[:lower]

        training_data = no_actions + buys + sells
        np.random.shuffle(training_data)

        data_dir = '../nn/training_data'

        file_path = f'{data_dir}/{self.currency_pair.value}_training_data_svm.pickle'
        scaler_file_path = f'{data_dir}/{self.currency_pair.value}_trained_svm_scaler.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(training_data, f)

        with open(scaler_file_path, 'wb') as f:
            pickle.dump(scaler, f)

    def train(self) -> None:
        print('Training SVM')
        data_path = f'../nn/training_data/{self.currency_pair.value}_training_data_svm.pickle'

        training_data = pickle.load(open(data_path, 'rb'))

        x_train = []
        y_train = []

        for seq, target in training_data:
            x_train.append(seq)
            y_train.append(target)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print(x_train.shape)
        print(y_train.shape)

        param_grid = {'C': [1, 5, 10, 50],
                      'gamma': [1, 5, 10, 50],
                      'kernel': ['rbf', 'linear']}

        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, return_train_score=True)
        grid_search.fit(x_train, y_train)

        print(f'Best SVM parameters:\n{grid_search.best_params_}\n{grid_search.best_score_}')

        model = grid_search.best_estimator_

        data_dir = '../nn/training_data'

        trained_svm_file = f'{self.currency_pair.value}_trained_svm.pickle'

        with open(f'{data_dir}/{trained_svm_file}', 'wb') as f:
            pickle.dump(model, f)



from nn.rnn_market_trainer import RnnMarketTrainer
from nn.rnn_utilities import RNN_LOOKBACK
from market_proxy.currency_pairs import CurrencyPairs
from strategy.learner_training_strategy import LearnerTrainingStrategy
from utils.utils import ML_TRAINING_YEARS


strategy = LearnerTrainingStrategy(RNN_LOOKBACK)
all_pairs = CurrencyPairs.all_pairs()

for i in range(len(all_pairs)):
    currency_pair = all_pairs[i]
    cnn_trainer = RnnMarketTrainer(0.70, currency_pair)
    results = strategy.run_strategy(currency_pair, learner=cnn_trainer, date_range=ML_TRAINING_YEARS)
    print(f'Results for {currency_pair.value}:\n{results}')
    cnn_trainer.train()

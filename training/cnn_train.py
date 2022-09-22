from nn.cnn_market_trainer import CnnMarketTrainer
from nn.cnn_utilities import CNN_LOOKBACK
from market_proxy.currency_pairs import CurrencyPairs
from strategy.learner_training_strategy import LearnerTrainingStrategy
from utils.utils import ML_TRAINING_YEARS


strategy = LearnerTrainingStrategy(CNN_LOOKBACK)
all_pairs = CurrencyPairs.all_pairs()

for i in range(len(all_pairs)):
    currency_pair = all_pairs[i]
    cnn_trainer = CnnMarketTrainer(0.70, currency_pair)
    results = strategy.run_strategy(currency_pair, learner=cnn_trainer, date_range=ML_TRAINING_YEARS)
    print(f'Results for {currency_pair.value}:\n{results}')
    cnn_trainer.train()

from nn.svm_market_trainer import SVMMarketTrainer
from market_proxy.currency_pairs import CurrencyPairs
from strategy.learner_training_strategy import LearnerTrainingStrategy
from utils.utils import ML_TRAINING_YEARS


strategy = LearnerTrainingStrategy()
all_pairs = CurrencyPairs.all_pairs()

for i in range(len(all_pairs)):
    currency_pair = all_pairs[i]
    svm_trainer = SVMMarketTrainer(0.70, currency_pair)
    results = strategy.run_strategy(currency_pair, learner=svm_trainer, date_range=ML_TRAINING_YEARS)
    print(f'Results for {currency_pair.value}:\n{results}')
    svm_trainer.train()

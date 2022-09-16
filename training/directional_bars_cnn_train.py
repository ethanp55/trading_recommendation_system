from aat.aat_market_trainer import CnnAatMarketTrainer
from market_proxy.currency_pairs import CurrencyPairs
from strategy.directional_bars_strategy import DirectionalBarsStrategy

RISK_REWARD_RATIO = 2.0

bar_strategy = DirectionalBarsStrategy(0, RISK_REWARD_RATIO, 0.1, False, 3, 20, True)
cnn_trainer = CnnAatMarketTrainer(RISK_REWARD_RATIO, training_data_percentage=0.7)

all_pairs = CurrencyPairs.all_pairs()

for i in range(len(all_pairs)):
    currency_pair = all_pairs[i]
    results = bar_strategy.run_strategy(currency_pair, cnn_trainer)
    print(f'Results for {currency_pair.value}:\n{results}')

cnn_trainer.save_data()

cnn_trainer.fit_cnn()

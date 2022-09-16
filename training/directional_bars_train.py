from aat.aat_market_trainer import KnnAatMarketTrainer
from market_proxy.currency_pairs import CurrencyPairs
from strategy.directional_bars_strategy import DirectionalBarsStrategy

RISK_REWARD_RATIO = 2.0

bar_strategy = DirectionalBarsStrategy(0, RISK_REWARD_RATIO, 0.1, False, 3, 20, True)
knn_trainer = KnnAatMarketTrainer(RISK_REWARD_RATIO)

all_pairs = CurrencyPairs.all_pairs()

for i in range(len(all_pairs) - 2):
    currency_pair = all_pairs[i]
    results = bar_strategy.run_strategy(currency_pair, knn_trainer)
    print(f'Results for {currency_pair.value}:\n{results}')

knn_trainer.save_data()

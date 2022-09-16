from aat.aat_market_tester import KnnAatMarketTester
from market_proxy.currency_pairs import CurrencyPairs
from strategy.directional_bars_strategy import DirectionalBarsStrategy

RISK_REWARD_RATIO = 2.0

bar_strategy = DirectionalBarsStrategy(0, RISK_REWARD_RATIO, 0.1, False, 3, 20, True)
aat_tester = KnnAatMarketTester(RISK_REWARD_RATIO, CurrencyPairs.EUR_USD)

all_pairs = CurrencyPairs.all_pairs()

# Test without AAT
print('RESULTS WITHOUT AAT')
for i in range(len(all_pairs) - 2, len(all_pairs)):
    currency_pair = all_pairs[i]
    results = bar_strategy.run_strategy(currency_pair)
    print(f'Results for {currency_pair.value}:\n{results}')

# Test with AAT
print('RESULTS WITH AAT')
for i in range(len(all_pairs) - 2, len(all_pairs)):
    currency_pair = all_pairs[i]
    results = bar_strategy.run_strategy(currency_pair, aat_tester=aat_tester)
    print(f'Results for {currency_pair.value}:\n{results}')



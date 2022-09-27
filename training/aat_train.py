from aat.aat_market_trainer import AatKnnMarketTrainer
from market_proxy.currency_pairs import CurrencyPairs
from strategy.cnn_strategy import CnnStrategy, CNN_LOOKBACK
from strategy.directional_bars_strategy import DirectionalBarsStrategy
from strategy.ema_crossover_strategy import EMACrossStrategy
from strategy.macd_strategy import MACDStrategy
from strategy.news_strategy import NewsStrategy
from strategy.rnn_strategy import RnnStrategy, RNN_LOOKBACK
from strategy.rsi_crossover_strategy import RSICrossStrategy
from strategy.scalp_strategy import ScalpStrategy
from strategy.stoch_macd_crossover_strategy import StochMACDCrossStrategy
from strategy.svm_strategy import SVMStrategy
from utils.utils import AAT_TRAINING_YEARS, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF


# strategies = [CnnStrategy(CNN_LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF),
#               RnnStrategy(RNN_LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF),
#               DirectionalBarsStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF),
#               EMACrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
#               MACDStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
#               NewsStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
#               RSICrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
#               ScalpStrategy(LOOKBACK, 0.5, SPREAD_CUTOFF, LOOKBACK),
#               StochMACDCrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
#               SVMStrategy(LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF)]

strategies = [RnnStrategy(RNN_LOOKBACK, RISK_REWARD_RATIO, 0.5, LOOKBACK, SPREAD_CUTOFF)]

all_pairs = CurrencyPairs.all_pairs()

for strategy in strategies:
    for currency_pair in all_pairs:
        trainer = AatKnnMarketTrainer(currency_pair, strategy.risk_reward_ratio, strategy.name)
        results = strategy.run_strategy(currency_pair, trainer, date_range=AAT_TRAINING_YEARS)
        print(f'Results for {currency_pair.value}:\n{results}')

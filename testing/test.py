from market_proxy.currency_pairs import CurrencyPairs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from utils.utils import AAT_TESTING_YEARS, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF


strategies = [CnnStrategy(CNN_LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF),
              RnnStrategy(RNN_LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF),
              DirectionalBarsStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF),
              EMACrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
              MACDStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
              NewsStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
              RSICrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
              ScalpStrategy(LOOKBACK, 0.5, SPREAD_CUTOFF, LOOKBACK),
              StochMACDCrossStrategy(LOOKBACK, RISK_REWARD_RATIO, SPREAD_CUTOFF, LOOKBACK),
              SVMStrategy(LOOKBACK, RISK_REWARD_RATIO, PROBA_CUTOFF, LOOKBACK, SPREAD_CUTOFF)]

all_pairs = CurrencyPairs.all_pairs()
reward_results, day_fee_results, profit_results, n_trades_results, win_rate_results, names = {}, {}, {}, {}, {}, []

for strategy in strategies:
    names.append(strategy.name)

    for currency_pair in all_pairs:
        results = strategy.run_strategy(currency_pair, date_range=AAT_TESTING_YEARS)
        print(f'Results for {currency_pair.value}:\n{results}')

        reward_results[currency_pair.value] = reward_results.get(currency_pair.value, []) + [results.reward]
        day_fee_results[currency_pair.value] = day_fee_results.get(currency_pair.value, []) + [results.day_fees]
        profit_results[currency_pair.value] = profit_results.get(currency_pair.value, []) + [results.reward +
                                                                                             results.day_fees]
        n_trades_results[currency_pair.value] = n_trades_results.get(currency_pair.value, []) + [results.n_buys +
                                                                                                 results.n_sells]
        win_rate_results[currency_pair.value] = win_rate_results.get(currency_pair.value, []) + \
            [results.n_wins / (results.n_wins + results.n_losses)]

for currency_pair in all_pairs:
    key = currency_pair.value
    rewards, day_fees, profits, n_trades, win_rates = reward_results[key], day_fee_results[key], profit_results[key], \
                                                n_trades_results[key], win_rate_results[key]

    # Save the results to csv files
    rewards_df = pd.DataFrame({'Rewards': rewards, 'Strategy': names})
    rewards_df.to_csv(f'./results/{key}_rewards.csv')

    day_fees_df = pd.DataFrame({'Day_Fees': day_fees, 'Strategy': names})
    day_fees_df.to_csv(f'./results/{key}_day_fees.csv')

    profits_df = pd.DataFrame({'Profits': profits, 'Strategy': names})
    profits_df.to_csv(f'./results/{key}_profits.csv')

    n_trades_df = pd.DataFrame({'N_Trades': n_trades, 'Strategy': names})
    n_trades_df.to_csv(f'./results/{key}_n_trades.csv')

    wls_df = pd.DataFrame({'Win_Rates': win_rates, 'Strategy': names})
    wls_df.to_csv(f'./results/{key}_wins_losses.csv')

    # Generate plots
    x_pos = np.arange(len(names))
    plt.bar(x_pos, rewards, align='center', alpha=0.5,
            color=['green', 'red', 'blue', 'orange', 'purple', 'teal', 'yellow', 'cyan', 'black', 'brown'])
    plt.xticks(x_pos, names, fontsize=6)
    plt.xlabel('Strategy')
    plt.ylabel('Reward ($)')
    plt.title(f'Trade Rewards on {key.upper()} (Excluding Day Fees)')
    plt.savefig(f'./results/images/{key}_rewards.png', bbox_inches='tight')
    plt.clf()

    x_pos = np.arange(len(names))
    day_fees_converted = [abs(fee) for fee in day_fees]
    plt.bar(x_pos, day_fees_converted, align='center', alpha=0.5,
            color=['green', 'red', 'blue', 'orange', 'purple', 'teal', 'yellow', 'cyan', 'black', 'brown'])
    plt.xticks(x_pos, names, fontsize=6)
    plt.xlabel('Strategy')
    plt.ylabel('Day Fees ($)')
    plt.title(f'Day Fees on {key.upper()}')
    plt.savefig(f'./results/images/{key}_day_fees.png', bbox_inches='tight')
    plt.clf()

    x_pos = np.arange(len(names))
    plt.bar(x_pos, profits, align='center', alpha=0.5,
            color=['green', 'red', 'blue', 'orange', 'purple', 'teal', 'yellow', 'cyan', 'black', 'brown'])
    plt.xticks(x_pos, names, fontsize=6)
    plt.xlabel('Strategy')
    plt.ylabel('Net Profit ($)')
    plt.title(f'Net Profits on {key.upper()}')
    plt.savefig(f'./results/images/{key}_profits.png', bbox_inches='tight')
    plt.clf()

    x_pos = np.arange(len(names))
    plt.bar(x_pos, n_trades, align='center', alpha=0.5,
            color=['green', 'red', 'blue', 'orange', 'purple', 'teal', 'yellow', 'cyan', 'black', 'brown'])
    plt.xticks(x_pos, names, fontsize=6)
    plt.xlabel('Strategy')
    plt.ylabel('# Trades')
    plt.title(f'Number of Trades on {key.upper()}')
    plt.savefig(f'./results/images/{key}_n_trades.png', bbox_inches='tight')
    plt.clf()

    x_pos = np.arange(len(names))
    plt.bar(x_pos, win_rates, align='center', alpha=0.5,
            color=['green', 'red', 'blue', 'orange', 'purple', 'teal', 'yellow', 'cyan', 'black', 'brown'])
    plt.xticks(x_pos, names, fontsize=6)
    plt.xlabel('Strategy')
    plt.ylabel('Win Rate')
    plt.title(f'Win Rate on {key.upper()}')
    plt.savefig(f'./results/images/{key}_win_rate.png', bbox_inches='tight')
    plt.clf()

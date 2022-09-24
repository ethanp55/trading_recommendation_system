from market_proxy.currency_pairs import CurrencyPairs
import pandas as pd
from ti.technical_indicator_generator import TechnicalIndicatorGenerator


class DataRetriever(object):
    @staticmethod
    def get_data_for_pair(currency_pair: CurrencyPairs, date_range: str) -> pd.DataFrame:
        print(f'Retrieving and formatting data for {currency_pair.value} from {date_range}')

        df = pd.read_csv(f'../market_proxy/data/Oanda_{currency_pair.value}_M5_{date_range}.csv')
        df.Date = pd.to_datetime(df.Date, utc=True)
        df = TechnicalIndicatorGenerator.add_indicators(df)

        return df

    @staticmethod
    def get_news_data(currency_pair: CurrencyPairs) -> pd.DataFrame:
        def _apply_impact(val, impact):
            return float(val) * float(impact)

        print(f'Retrieving and formatting news data for {currency_pair.value}')

        currency1, currency2 = currency_pair.value.split('_')
        currency1, currency2 = currency1.upper(), currency2.upper()

        news = pd.read_csv(f'../market_proxy/data/events.csv')

        news.Date = pd.to_datetime(news.Date, utc=True)
        news.drop(news[(news['Impact'] != 'low') & (news['Impact'] != 'med') & (news['Impact'] != 'high')].index,
                  inplace=True)
        news.loc[news['Impact'] == 'low', 'Impact'] = 1
        news.loc[news['Impact'] == 'med', 'Impact'] = 2
        news.loc[news['Impact'] == 'high', 'Impact'] = 3
        news['Impact'] = pd.to_numeric(news['Impact'])
        news['Actual_Class'] = news.apply(lambda row: _apply_impact(row['Actual'], row['Impact']), axis=1)
        news['Previous_Class'] = news.apply(lambda row: _apply_impact(row['Previous'], row['Impact']), axis=1)
        news_base = news.loc[news['Currency_Code'] == currency1]
        news_counter = news.loc[news['Currency_Code'] == currency2]
        news_base.drop(['Currency_Code', 'Actual', 'Previous', 'Actual_Val', 'Forecast_Val', 'Previous_Val'], axis=1,
                       inplace=True)
        news_counter.drop(['Currency_Code', 'Actual', 'Previous', 'Actual_Val', 'Forecast_Val', 'Previous_Val'], axis=1,
                          inplace=True)
        by_date1 = news_base.groupby('Date')
        impact1, actual1, previous1 = by_date1['Impact'].mean().reset_index(), by_date1[
            'Actual_Class'].mean().reset_index(), by_date1['Previous_Class'].mean().reset_index()
        news_base = news_base.iloc[0:0]
        news_base['Date'], news_base['Impact'], news_base['Actual_Class'], news_base['Previous_Class'] = \
            impact1['Date'], impact1['Impact'], actual1['Actual_Class'], previous1['Previous_Class']
        by_date2 = news_counter.groupby('Date')
        impact2, actual2, previous2 = by_date2['Impact'].mean().reset_index(), by_date2[
            'Actual_Class'].mean().reset_index(), by_date2['Previous_Class'].mean().reset_index()
        news_counter = news_counter.iloc[0:0]
        news_counter['Date'], news_counter['Impact'], news_counter['Actual_Class'], news_counter['Previous_Class'] = \
        impact2['Date'], impact2['Impact'], actual2['Actual_Class'], previous2['Previous_Class']

        df = pd.merge(news_base, news_counter, how='left', on='Date')
        df.reset_index(drop=True, inplace=True)
        df = df.fillna(method='ffill')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

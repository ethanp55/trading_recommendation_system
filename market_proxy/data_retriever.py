from market_proxy.currency_pairs import CurrencyPairs
import pandas as pd
from ti.technical_indicator_generator import TechnicalIndicatorGenerator


class DataRetriever(object):
    @staticmethod
    def get_data_for_pair(currency_pair: CurrencyPairs, date_range: str) -> pd.DataFrame:
        print(f'Retrieving and formatting data for {currency_pair.value} from {date_range}')

        df = pd.read_csv(f'../market_proxy/data/Oanda_{currency_pair.value}_M5_{date_range}.csv')
        df.Date = pd.to_datetime(df.Date)
        df = TechnicalIndicatorGenerator.add_indicators(df)

        return df

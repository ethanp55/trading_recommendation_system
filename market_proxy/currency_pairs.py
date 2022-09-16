import enum
from typing import List


class CurrencyPairs(enum.Enum):
    # The major pairs
    AUD_USD = 'Aud_Usd'
    EUR_USD = 'Eur_Usd'
    GBP_USD = 'Gbp_Usd'
    NZD_USD = 'Nzd_Usd'
    USD_CAD = 'Usd_Cad'
    USD_CHF = 'Usd_Chf'
    USD_JPY = 'Usd_Jpy'

    @staticmethod
    def all_pairs() -> List[enum.Enum]:
        return [pair for pair in CurrencyPairs]

    @staticmethod
    def usd_counter_pairs() -> List[enum.Enum]:
        return [CurrencyPairs.AUD_USD, CurrencyPairs.EUR_USD, CurrencyPairs.GBP_USD, CurrencyPairs.NZD_USD]

    @staticmethod
    def usd_base_pairs() -> List[enum.Enum]:
        return [CurrencyPairs.USD_CAD, CurrencyPairs.USD_CHF, CurrencyPairs.USD_JPY]

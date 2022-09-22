import enum
from typing import List


class CurrencyPairs(enum.Enum):
    EUR_USD = 'Eur_Usd'
    GBP_USD = 'Gbp_Usd'
    USD_CAD = 'Usd_Cad'

    @staticmethod
    def all_pairs() -> List[enum.Enum]:
        return [pair for pair in CurrencyPairs]

    @staticmethod
    def usd_counter_pairs() -> List[enum.Enum]:
        return [CurrencyPairs.EUR_USD, CurrencyPairs.GBP_USD]

    @staticmethod
    def usd_base_pairs() -> List[enum.Enum]:
        return [CurrencyPairs.USD_CAD]

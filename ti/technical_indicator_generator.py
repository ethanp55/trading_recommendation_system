import numpy as np
import pandas as pd


class TechnicalIndicatorGenerator(object):
    @staticmethod
    def add_indicators(df):
        df['ema200'] = pd.Series.ewm(df['Mid_Close'], span=200).mean()
        df['ema100'] = pd.Series.ewm(df['Mid_Close'], span=100).mean()
        df['atr'] = TechnicalIndicatorGenerator.atr(df['Mid_High'], df['Mid_Low'], df['Mid_Close'])
        df['atr_sma'] = df['atr'].rolling(window=20).mean()
        df['rsi'] = TechnicalIndicatorGenerator.rsi(df['Mid_Close'])
        df['rsi_sma'] = df['rsi'].rolling(50).mean()
        df['adx'] = TechnicalIndicatorGenerator.adx(df['Mid_High'], df['Mid_Low'], df['Mid_Close'])
        df['macd'] = pd.Series.ewm(df['Mid_Close'], span=12).mean() - pd.Series.ewm(df['Mid_Close'], span=26).mean()
        df['macdsignal'] = pd.Series.ewm(df['macd'], span=9).mean()
        df['slowk_rsi'], df['slowd_rsi'] = TechnicalIndicatorGenerator.stoch_rsi(df['rsi'])
        df['vo'] = TechnicalIndicatorGenerator.vo(df['Volume'])
        df['willy'], df['willy_ema'] = TechnicalIndicatorGenerator.williams_r(df['Mid_High'], df['Mid_Low'],
                                                                              df['Mid_Close'])

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def psar(barsdata, iaf=0.02, maxaf=0.2):
        length = len(barsdata)
        high = list(barsdata['Mid_High'])
        low = list(barsdata['Mid_Low'])
        close = list(barsdata['Mid_Close'])
        psar = close[0:len(close)]
        bull = True
        af = iaf
        hp = high[0]
        lp = low[0]
        for i in range(2, length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            reverse = False
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + iaf, maxaf)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + iaf, maxaf)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]
        return psar

    @staticmethod
    def atr(high, low, close, lookback=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        return true_range.rolling(lookback).sum() / lookback

    @staticmethod
    def rsi(closes, periods=14):
        close_delta = closes.diff()

        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))

        return rsi

    @staticmethod
    def adx(high, low, close, lookback=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha=1 / lookback).mean()

        return adx_smooth

    @staticmethod
    def stoch(high, low, close, lookback=14):
        high_lookback = high.rolling(lookback).max()
        low_lookback = low.rolling(lookback).min()
        slow_k = (close - low_lookback) * 100 / (high_lookback - low_lookback)
        slow_d = slow_k.rolling(3).mean()

        return slow_k, slow_d

    @staticmethod
    def stoch_rsi(data, k_window=3, d_window=3, window=14):
        min_val = data.rolling(window=window, center=False).min()
        max_val = data.rolling(window=window, center=False).max()

        stoch = ((data - min_val) / (max_val - min_val)) * 100

        slow_k = stoch.rolling(window=k_window, center=False).mean()

        slow_d = slow_k.rolling(window=d_window, center=False).mean()

        return slow_k, slow_d

    @staticmethod
    def n_macd(macd, macdsignal, lookback=50):
        n_macd = 2 * (
            ((macd - macd.rolling(lookback).min()) / (macd.rolling(lookback).max() - macd.rolling(lookback).min()))) - 1
        n_macdsignal = 2 * (((macdsignal - macdsignal.rolling(lookback).min()) / (
                macdsignal.rolling(lookback).max() - macdsignal.rolling(lookback).min()))) - 1

        return n_macd, n_macdsignal

    @staticmethod
    def vo(volume, short_lookback=5, long_lookback=10):
        short_ema = pd.Series.ewm(volume, span=short_lookback).mean()
        long_ema = pd.Series.ewm(volume, span=long_lookback).mean()

        volume_oscillator = (short_ema - long_ema) / long_ema

        return volume_oscillator

    @staticmethod
    def bar_lengths(bar_lens, window=36):
        return bar_lens.rolling(window=window).mean(), bar_lens.rolling(window=window).std()

    @staticmethod
    def sar_lengths(opens, sars, window=36):
        diffs = abs(opens - sars.shift(1))

        return diffs.rolling(window=window).mean(), diffs.rolling(window=window).std()

    @staticmethod
    def supertrend(barsdata, atr_len=3, mult=3):
        curr_atr = TechnicalIndicatorGenerator.atr(barsdata['Mid_High'], barsdata['Mid_Low'], barsdata['Mid_Close'],
                                                   lookback=atr_len)
        highs, lows = barsdata['Mid_High'], barsdata['Mid_Low']
        hl2 = (highs + lows) / 2
        final_upperband = hl2 + mult * curr_atr
        final_lowerband = hl2 - mult * curr_atr

        # initialize Supertrend column to True
        supertrend = [True] * len(barsdata)

        close = barsdata['Mid_Close']

        for i in range(1, len(barsdata.index)):
            curr, prev = i, i - 1

            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True

            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False

            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if supertrend[curr] is True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]

                if supertrend[curr] is False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

        return supertrend, final_upperband, final_lowerband

    @staticmethod
    def heikin_ashi(opens, highs, lows, closes):
        ha_close = list((opens + highs + lows + closes) / 4)
        ha_opens = []

        opens_list, closes_list = list(opens), list(closes)

        for i in range(len(ha_close)):
            if i == 0:
                ha_opens.append((opens_list[i] + closes_list[i]) / 2)

            else:
                ha_opens.append((ha_opens[i - 1] + ha_close[i - 1]) / 2)

        ha_highs = list(pd.DataFrame({'ha_open': ha_opens, 'ha_close': ha_close, 'high': list(highs)}).max(axis=1))
        ha_lows = list(pd.DataFrame({'ha_open': ha_opens, 'ha_close': ha_close, 'low': list(lows)}).min(axis=1))

        return ha_opens, ha_highs, ha_lows, ha_close

    @staticmethod
    def trend_indicator(opens, highs, lows, closes, ema_period=50, smoothing_period=10):
        ha_open, ha_high, ha_low, ha_close = TechnicalIndicatorGenerator.heikin_ashi(opens, highs, lows, closes)

        ha_o_ema = pd.Series.ewm(pd.DataFrame({'ha_open': ha_open}), span=ema_period).mean()
        ha_h_ema = pd.Series.ewm(pd.DataFrame({'ha_high': ha_high}), span=ema_period).mean()
        ha_l_ema = pd.Series.ewm(pd.DataFrame({'ha_low': ha_low}), span=ema_period).mean()
        ha_c_ema = pd.Series.ewm(pd.DataFrame({'ha_close': ha_close}), span=ema_period).mean()

        return pd.Series.ewm(ha_o_ema, span=smoothing_period).mean(), \
               pd.Series.ewm(ha_h_ema, span=smoothing_period).mean(), \
               pd.Series.ewm(ha_l_ema, span=smoothing_period).mean(), \
               pd.Series.ewm(ha_c_ema, span=smoothing_period).mean()

    @staticmethod
    def williams_r(highs, lows, closes, length=21, ema_length=15):
        highest_highs = highs.rolling(window=length).max()
        lowest_lows = lows.rolling(window=length).min()

        willy = 100 * (closes - highest_highs) / (highest_highs - lowest_lows)
        willy_ema = pd.Series.ewm(willy, span=ema_length).mean()

        return willy, willy_ema

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import yfinance as yf
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta.trend import *
from ta.others import *

warnings.filterwarnings("ignore")


class Crypto:
    def __init__(self, pair='BTC-USD', period='max'):
        self.pair = pair
        self.period = period
        self.frame = self.fetch()

    @property
    def open(self):
        return self.frame['Open']

    @property
    def high(self):
        return self.frame['High']

    @property
    def low(self):
        return self.frame['Low']

    @property
    def close(self):
        return self.frame['Close']

    @property
    def volume(self):
        return self.frame['Volume']

    @property
    def absolute_growth(self, tau=1):
        series = self.frame['Close'].copy()
        return pd.Series(data=[series[i] - series[i - 1] for i in range(tau, len(series))], name='absolute_growth')

    @property
    def relative_growth(self, tau=1):
        series = self.frame['Close'].copy()
        return pd.Series(data=[(series[i] - series[i - tau]) / series[i - tau] for i in range(tau, len(series))],
                         name='relative_growth')

    @property
    def logarithmic_growth(self, tau=1):
        series = self.frame['Close'].copy()
        return pd.Series(data=[np.log(series[i] / series[i - tau]) for i in range(tau, len(series))],
                         name='logarithmic_growth')

    @staticmethod
    def basic_info(*args):
        info = pd.DataFrame(columns=[i.name for i in args])
        series = pd.DataFrame(data=args).T

        info.loc['min'] = series.min()
        info.loc['max'] = series.max()
        info.loc['mean'] = series.mean()
        info.loc['median'] = series.median()
        info.loc['range'] = info.loc['max'] - info.loc['min']
        info.loc['std'] = series.std()
        info.loc['var_coef'] = info.loc['std'] / info.loc['mean']
        info.loc['asymmetry'] = series.skew()
        info.loc['kurtosis'] = series.kurt()

        return info.round(2)

    def fetch(self):
        df = yf.Ticker(self.pair).history(period=self.period).filter(['Open', 'High', 'Low', 'Close', 'Volume'])
        _open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
        df['awesome_oscillator'] = awesome_oscillator(high, low)
        df['kama'] = kama(close)
        df['ppo'] = ppo(close)
        df['ppo_signal'] = ppo_signal(close)
        df['pvo'] = pvo(volume)
        df['pvo_signal'] = pvo_signal(volume)
        df['roc'] = roc(close)
        df['rsi'] = rsi(close)
        df['stochrsi'] = stochrsi(close)
        df['stoch'] = stoch(close, high, low)
        df['stoch_signal'] = stoch_signal(close, high, low)
        df['tsi'] = tsi(close)
        df['ultimate_oscillator'] = ultimate_oscillator(high, low, close)
        df['williams_r'] = williams_r(high, low, close)
        df['acc_dist_index'] = acc_dist_index(high, low, close, volume)
        df['chaikin_money_flow'] = chaikin_money_flow(high, low, close, volume)
        df['ease_of_movement'] = ease_of_movement(high, low, volume)
        df['sma_ease_of_movement'] = sma_ease_of_movement(high, low, volume)
        df['force_index'] = force_index(close, volume)
        df['money_flow_index'] = money_flow_index(high, low, close, volume)
        df['negative_volume_index'] = negative_volume_index(close, volume)
        df['on_balance_volume'] = on_balance_volume(close, volume)
        df['volume_price_trend'] = volume_price_trend(close, volume)
        df['volume_weighted_average_price'] = volume_weighted_average_price(high, low, close, volume)
        df['average_true_range'] = average_true_range(high, low, close)
        df['bollinger_hband'] = bollinger_hband(close)
        df['bollinger_lband'] = bollinger_lband(close)
        df['bollinger_mavg'] = bollinger_mavg(close)
        df['bollinger_pband'] = bollinger_pband(close)
        df['bollinger_wband'] = bollinger_wband(close)
        df['donchian_channel_hband'] = donchian_channel_hband(high, low, close)
        df['donchian_channel_lband'] = donchian_channel_lband(high, low, close)
        df['donchian_channel_mband'] = donchian_channel_mband(high, low, close)
        df['donchian_channel_pband'] = donchian_channel_pband(high, low, close)
        df['donchian_channel_wband'] = donchian_channel_wband(high, low, close)
        df['keltner_channel_hband'] = keltner_channel_hband(high, low, close)
        df['keltner_channel_lband'] = keltner_channel_lband(high, low, close)
        df['keltner_channel_mband'] = keltner_channel_mband(high, low, close)
        df['keltner_channel_pband'] = keltner_channel_pband(high, low, close)
        df['keltner_channel_wband'] = keltner_channel_wband(high, low, close)
        df['ulcer_index'] = ulcer_index(close)
        df['aroon_down'] = aroon_down(close)
        df['aroon_up'] = aroon_up(close)
        df['cci'] = cci(high, low, close)
        df['dpo'] = dpo(close)
        df['ema_indicator'] = ema_indicator(close)
        df['ichimoku_a'] = ichimoku_a(high, low)
        df['ichimoku_b'] = ichimoku_b(high, low)
        df['ichimoku_base_line'] = ichimoku_base_line(high, low)
        df['ichimoku_conversion_line'] = ichimoku_conversion_line(high, low)
        df['kst'] = kst(close)
        df['kst_sig'] = kst_sig(close)
        df['macd'] = macd(close)
        df['macd_signal'] = macd_signal(close)
        df['mass_index'] = mass_index(high, low)
        df['sma'] = sma_indicator(close)
        df['stc'] = stc(close)
        df['trix'] = trix(close)
        df['vortex_indicator_neg'] = vortex_indicator_neg(high, low, close)
        df['vortex_indicator_pos'] = vortex_indicator_pos(high, low, close)
        df['cumulative_return'] = cumulative_return(close)
        df['daily_return'] = daily_return(close)

        return df

    def growth_plot(self):
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
        fig.suptitle(f"{self.pair} Rates of Return")

        axs[0, 0].plot(self.close)
        axs[0, 0].set_title(self.close.name)

        axs[0, 1].plot(self.absolute_growth)
        axs[0, 1].set_title(self.absolute_growth.name)

        axs[1, 0].plot(self.relative_growth)
        axs[1, 0].set_title(self.relative_growth.name)

        axs[1, 1].plot(self.logarithmic_growth)
        axs[1, 1].set_title(self.logarithmic_growth.name)

        plt.tight_layout()
        plt.savefig('plots/Rates of Return.svg', format='svg')

    def test_adf(self, description=False):
        test, p_value, _, _, crit, _ = adfuller(self.close)
        text = f"AUGMENTED DICKEY-FULLER TEST\n" \
               f"Test's statistics: {round(test, 4)}\n" \
               f"p-value: {round(p_value, 4)}\n" \
               f"Critical Values:\n" \
               f"1%: {round(crit['1%'], 4)}\n" \
               f"5%: {round(crit['5%'], 4)}\n" \
               f"10%: {round(crit['10%'], 4)}\n"

        if p_value > 0.05 and all(i < test for i in list(crit.values())):
            text += "\nNo reason to reject the null hypothesis. So, the time series is in fact non-stationary\n"
            result = False
        else:
            text += "\nWe can reject the null hypothesis and take that the series is stationary\n"
            result = True

        if description:
            print(text)

        return result

    def test_kpps(self, description=False):
        test, p_value, _, crit = kpss(self.close)

        text = f"KPPS TEST\nTest's statistics: {round(test, 4)}\n" \
               f"p-value: {round(p_value, 4)}\n" \
               f"Critical Values:\n" \
               f"10%: {round(crit['10%'], 4)}\n" \
               f"5%: {round(crit['5%'], 4)}\n" \
               f"2.5%: {round(crit['2.5%'], 4)}\n" \
               f"1%: {round(crit['1%'], 4)}\n"

        if all(i < test for i in list(crit.values())):
            text += "\nThere is evidence for rejecting the null hypothesis in favor of the alternative." \
                    "Hence, the series is non-stationary\n"
            result = False
        else:
            text += "\nNo evidence to reject null hypothesis. Hence, the series is stationary\n"
            result = True

        if description:
            print(text)

        return result

    def day(self, day="Monday"):
        df = self.frame.copy()
        df['day'] = [i.day_name() for i in df.index]

        return df[df['day'] == day].drop(columns=['day'])

    # TODO zrobic analize dnia tygodnia
    def day_analysis(self):
        pass

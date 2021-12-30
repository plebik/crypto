import warnings

import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
from ta.volatility import *

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
        return yf.Ticker(self.pair).history(period=self.period).filter(['Open', 'High', 'Low', 'Close', 'Volume'])

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

    def test_adf(self, info=False):
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

        if info:
            print(text)

        return result

    def test_kpps(self, info=False):
        test, p_value, _, crit = kpss(self.close)

        text = f"KPPS TEST\nTest's statistics: {round(test, 4)}\n" \
               f"p-value: {round(p_value, 4)}\n" \
               f"Critical Values:\n" \
               f"10%: {round(crit['10%'], 4)}\n" \
               f"5%: {round(crit['5%'], 4)}\n" \
               f"2.5%: {round(crit['2.5%'], 4)}\n" \
               f"1%: {round(crit['1%'], 4)}\n"

        if all(i < test for i in list(crit.values())):
            text += "\nThere is an evidence for rejecting the null hypothesis in favor of the alternative." \
                    "Hence, the series is non-stationary\n"
            result = False
        else:
            text += "\nNo evidence to reject null hypothesis. Hence, the series is stationary\n"
            result = True

        if info:
            print(text)

        return result

    def day(self, day="Monday"):
        df = self.frame.copy()
        df['day'] = [i.day_name() for i in df.index]

        return df[df['day'] == day].drop(columns=['day'])

    # TODO zrobic analize dnia tygodnia
    def day_analysis(self):
        pass


class Environment(Crypto):
    def __init__(self, balance):
        super().__init__()
        self.balance = balance
        self.units = 0.0
        self.fee = 0.999

    def buy(self, price, amount=10.0):
        self.units = (amount * self.fee) / price
        self.balance -= amount
        print(f"BUY: {round(price, 2)} Balance: {round(self.balance, 2)}")

    def sell(self, price):
        self.balance += self.units * price * self.fee
        self.units = 0.0
        print(f"SELL: {round(price, 2)} Balance: {round(self.balance, 2)}")

    def backtest(self, strategy='bollinger'):
        if strategy == 'bollinger':
            frame = self.frame.copy().filter(['Close'])
            upper = bollinger_hband(frame['Close'])
            lower = bollinger_lband(frame['Close'])
            price = frame['Close']
            balance = []
            for i in frame.index:
                if price[i] >= upper[i] and self.units > 0.0:
                    self.sell(price[i])
                if price[i] <= lower[i]:
                    self.buy(price[i])

                balance.append(self.balance)

            plt.plot(price[-100:])
            plt.plot(upper[-100:])
            plt.plot(lower[-100:])
            plt.show()

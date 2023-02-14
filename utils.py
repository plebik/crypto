import os
import pandas as pd
import numpy as np
from binance.client import Client
import warnings
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("display.max_columns", 10)

pd.options.mode.chained_assignment = None


class Crypto:
    def __init__(self, name):
        self.data = self.fetch_data(name)
        self.data['r'] = 100 * np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.name = name

    def __str__(self):
        return self.name

    @staticmethod
    def fetch_data(symbol, timeframe="1d", start='2018-01-01', stop='2022-12-31'):
        with open('vars.txt') as f:
            lines = f.readlines()

        api_key, secret_key = lines[0][:-2], lines[1]

        client = Client(api_key, secret_key)
        tmp = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", timeframe, start, stop),
                           columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                    'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                    'Taker buy quote asset volume', 'Ignore'])

        tmp = tmp[['Close time', 'Close', 'Volume']]
        tmp.columns = ['Time', 'Close', 'Volume']
        tmp['Time'] = pd.to_datetime(tmp['Time'], unit='ms')
        tmp['Time'] = tmp['Time'].dt.date
        tmp.index = tmp['Time']
        tmp.drop(columns=['Time'], inplace=True)
        tmp['Close'] = pd.to_numeric(tmp['Close'])
        tmp['Volume'] = pd.to_numeric(tmp['Volume'])

        return tmp

    @staticmethod
    def daily_subsets(data):
        if isinstance(data, pd.DataFrame):
            data['day'] = list(map(lambda x: date.weekday(x), data.index))
            dictionary = {
                'all_days': data,
                'monday': data[data['day'] == 0],
                'tuesday': data[data['day'] == 1],
                'wednesday': data[data['day'] == 2],
                'thursday': data[data['day'] == 3],
                'friday': data[data['day'] == 4],
                'saturday': data[data['day'] == 5],
                'sunday': data[data['day'] == 6]
            }
            for i in dictionary.values():
                i.drop(columns=['day'], inplace=True)

            return dictionary
        else:
            raise Exception("The object passed to the function has to be a dataframe")

    @staticmethod
    def annual_subsets(data):
        if isinstance(data, pd.DataFrame):
            data['year'] = list(map(lambda x: x.year, data.index))
            dictionary = {}
            for i in np.unique(data['year']):
                dictionary[i] = data[data['year'] == i]

            for i in dictionary.values():
                i.drop(columns=['year'], inplace=True)

            return dictionary
        else:
            raise Exception("The object passed to the function has to be a dataframe")

    @staticmethod
    def basic_statistics(data):
        frame = pd.DataFrame([data.mean().values[0],
                              data.std().values[0],
                              data.std().values[0] / data.mean().values[0],
                              data.skew().values[0],
                              data.kurtosis().values[0]],
                             index=['mean', 'std', 'volatility', 'skew', 'kurtosis'],
                             columns=['value'])
        return round(frame, 2)

    @staticmethod
    def basic_statistics_for_each_day(data):
        daily_subsets = Crypto.daily_subsets(data)
        statistics = [Crypto.basic_statistics(i) for i in daily_subsets.values()]
        frame = pd.concat(statistics, axis=1)
        frame.columns = ["all_days", 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

        return frame

    @staticmethod
    def average_daily_returns_indices_by_annual_sub_periods(data):
        annual_subsets = Crypto.annual_subsets(data)
        daily_subsets = {}
        for i, j in annual_subsets.items():
            daily_subsets[i] = Crypto.daily_subsets(j)
            for k in daily_subsets[i].keys():
                daily_subsets[i][k] = round(daily_subsets[i][k].mean().values[0], 2)

        return pd.DataFrame(daily_subsets.values(), index=list(daily_subsets.keys()))

    @staticmethod
    def adf(data, regression='c'):
        output = adfuller(data.dropna(), regression=regression)
        text = "\n------------------------------------------------------\n" \
               "H0: Time series is non-stationary\n" \
               "H1: Time series is stationary\n\n" \
               f"Test statistic:\t{round(output[0], 2)}\n" \
               f"P-value:\t{output[1]}\n" \
               f"Number of lags:\t{output[2]}\n" \
               f"Critical values:\n" \
               f"1%:\t\t{round(output[4]['1%'], 2)}\n" \
               f"5%:\t\t{round(output[4]['5%'], 2)}\n1" \
               f"0%:\t{round(output[4]['10%'], 2)}" \
               "\n------------------------------------------------------\n"
        if output[1] < 0.05:
            text += "We reject null hypothesis. Time series is stationary"
        else:
            text += "We fail to reject the null hypothesis. Time series is  non-stationary"
        return text

    @staticmethod
    def box_pierce(data):
        output = acorr_ljungbox(data.dropna(), boxpierce=True)
        output.columns = ['Ljung-Box_stat', 'Ljung-Box_pvalue', 'Box-Pierce_stat', 'Box-Pierce_pvaue']

        return output

    @staticmethod
    def arch(data):
        output = het_arch(data.dropna())
        text = f"--------------------------------\n" \
               f"Lagrange test statistic:\t{round(output[0], 3)}\n" \
               f"Lagrange p-value:\t{round(output[1], 3)}\n" \
               f"F statistics of F test:\t{round(output[2], 3)}\n" \
               f"F test p-value:\t{round(output[3], 3)}\n" \
               f"--------------------------------\n"

        return text

    def day_of_the_week_analysis(self):
        print(Crypto.adf(self.data['r']))

        return ''

    def event_analysis(self):
        pass

    def volume_analysis(self):
        pass

    def plot(self):
        pass


def get_plot(data, symbol='BTC'):
    tmp = data.copy()
    tmp['Time'] = pd.to_datetime(tmp['Time'])
    plt.plot(tmp['Time'], tmp[f'Close_{symbol}'], lw=1)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{symbol}")

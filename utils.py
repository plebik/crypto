import os
import pandas as pd
import numpy as np
from binance.client import Client
import warnings
import matplotlib.pyplot as plt
from datetime import date, datetime

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
                daily_subsets[i][k] = daily_subsets[i][k].mean().values[0]

        return pd.DataFrame(daily_subsets.values(), index=list(daily_subsets.keys()))

    def day_of_the_week_analysis(self):
        frame = self.data.copy()

        return self.data

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

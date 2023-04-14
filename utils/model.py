import warnings

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None


class Crypto:
    def __init__(self, symbol, interval="1d", start='2018-01-01', stop='2023-01-01'):
        self.data = yf.download(f'{symbol}-USD', interval=interval, start=start, end=stop)[['Close', 'Volume']]
        self.name = symbol

    def __str__(self):
        return self.name


class Index:
    def __init__(self, ticker):
        self.data = pd.read_csv(f'data/{ticker}')
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d')
        self.data.index = self.data['Date']
        self.data.drop(columns=['Date'], inplace=True)
        self.name = ticker[:-4]
    def __str__(self):
        return self.name
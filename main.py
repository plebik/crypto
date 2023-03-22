import os

import pandas as pd

from utils import *


def analysis(symbols=None):
    os.makedirs("csv", exist_ok=True)
    if symbols is None:
        symbols = ['BTC', 'BNB', 'XMR', 'BAT']
    cryptos = [Crypto(i) for i in symbols]
    for i in cryptos:
        frame = i.basic_statistics_for_each_day(i.data[['r']])
        frame.to_csv('csv/statistics.csv', mode='a', header=False)

        frame = i.average_daily_returns_indices_by_annual_sub_periods(i.data[['r']])
        frame.to_csv('csv/returns.csv', mode='a', header=False)

    tmp = pd.concat([i.data[['Close', 'Volume']] for i in cryptos], axis=1)
    tmp.columns = ['BTC', 'BTC_vol', 'BNB', 'BNB_vol', 'XMR', 'XMR_vol', 'BAT', 'BAT_vol']
    tmp.to_csv('csv/data.csv')


if __name__ == '__main__':

    # analysis()
    btc = Crypto('BTC')
    bnb = Crypto('BNB')

    cryptos = [btc, bnb]

    print(Crypto.volume_analysis(cryptos, plot=False))


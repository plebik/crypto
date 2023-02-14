from utils import Crypto
import os

if __name__ == '__main__':
    btc = Crypto('BTC')
    print(btc.arch(btc.data['r']))


def analysis(symbols=None):
    os.makedirs("csv", exist_ok=True)
    if symbols is None:
        symbols = ['BTC', 'BNB', 'XMR', 'BAT']
    cryptos = [Crypto(i) for i in symbols]
    for i in cryptos:
        frame = i.basic_statistics_for_each_day(i.data[['r']])
        frame.to_csv(f'csv/statistics.csv', mode='a', header=False)

    for i in cryptos:
        frame = i.average_daily_returns_indices_by_annual_sub_periods(i.data[['r']])
        frame.to_csv(f'csv/returns.csv', mode='a', header=False)

# analysis()

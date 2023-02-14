from utils import Crypto

if __name__ == '__main__':
    btc = Crypto('BTC')


def analysis(symbols=None):
    if symbols is None:
        symbols = ['BTC', 'BNB', 'XMR', 'BAT']
    cryptos = [Crypto(i) for i in symbols]
    with open('basic_statistics.txt', 'w') as file:
        for i in cryptos:
            file.write(i.name + str(i.basic_statistics_for_each_day(i.data[['r']])))
            file.write("\n\n")

    with open('daily_returns_by_annual_sub-periods.txt', 'w') as file:
        for i in cryptos:
            file.write(i.name + str(i.average_daily_returns_indices_by_annual_sub_periods(i.data[['r']])))
            file.write("\n\n")


analysis()

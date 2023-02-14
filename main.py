from utils import Crypto

if __name__ == '__main__':
    btc = Crypto('BTC')
    print(btc.average_daily_returns_indices_by_annual_sub_periods(btc.data[['Close']]))

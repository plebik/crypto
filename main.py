from utils import *


if __name__ == '__main__':
    btc = Crypto('BTC-USD', 'max')
    env = Environment(1000)
    env.backtest()

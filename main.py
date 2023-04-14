# import os
# from utils.event_analysis import event_analysis
from utils.volume_analysis import volume_analysis
from utils.dayoftheweek_analysis import dayoftheweek_analysis, ols
import pandas as pd

from utils.model import Crypto, Index

pd.set_option('display.max_columns', 15)

if __name__ == '__main__':
    cryptos = [Crypto(i) for i in ['BTC', 'BNB', 'XMR', 'BAT']]
    # indexes = [Index(i) for i in os.listdir("data")]

    #
    # volume_anal = volume_analysis(cryptos, plot=False)
    # volume_anal

    # event_anal = event_analysis(cryptos, indexes, '2022-02-24', 2, 5)
    # event_anal

    dayoftheweek_anal = dayoftheweek_analysis(cryptos, verbose=False)
    dayoftheweek_anal


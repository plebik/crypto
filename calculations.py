import os
import pandas as pd
from binance.client import Client
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

client = Client(os.environ["api_key"], os.environ["secret_key"])


def get_data(symbol, timeframe):
    tmp = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", timeframe, '2018-01-01', '2021-12-31'),
                       columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                'Taker buy quote asset volume', 'Ignore'])

    tmp = tmp[['Close time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    tmp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    tmp['Time'] = pd.to_datetime(tmp['Time'], unit='ms')

    return tmp


daily_btc = get_data("BTC", "1d")[['Time', 'Close']]  # payment token
daily_bnb = get_data("BNB", "1d")[['Time', 'Close']]  # exchange token
daily_xmr = get_data("XMR", "1d")[['Time', 'Close']]  # privacy token
daily_bat = get_data("BAT", "1d")[['Time', 'Close']]  # utility token

tmp1 = pd.merge(daily_btc, daily_bnb, on='Time', how="outer", suffixes=('_btc', '_bnb'))
tmp2 = pd.merge(tmp1, daily_xmr, on='Time', how="outer")
joint = pd.merge(tmp2, daily_bat, on='Time', how="outer", suffixes=('_xmr', '_bat'))
joint.dropna(inplace=True)

for i in['btc', 'bnb', 'xmr', 'bat']:
    joint[f'Close_{i}'] = pd.to_numeric(joint[f'Close_{i}'])


model = ols('Close_btc ~ Close_bnb + Close_xmr + Close_bat', data=joint).fit()

# view model summary
# print(model.summary())

print(durbin_watson(model.resid))

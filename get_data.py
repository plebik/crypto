import pandas as pd
from binance.client import Client

with open('vars.txt') as f:
    lines = f.readlines()

api_key, secret_key = lines[0][:-2], lines[1]
client = Client(api_key, secret_key)


def get_data(symbol, timeframe):
    tmp = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", timeframe, '2018-01-01', '2022-12-31'),
                       columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                'Taker buy quote asset volume', 'Ignore'])

    tmp = tmp[['Close time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    tmp.columns = ['Time', f'Open_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}', f'Volume_{symbol}']
    tmp['Time'] = pd.to_datetime(tmp['Time'], unit='ms')

    return tmp


data = [get_data(i, "1d")[['Time', f'Close_{i}']] for i in ['BTC', 'BNB', 'XMR', 'BAT']]

frame = data[0]
for i in range(1, len(data)):
    frame = frame.merge(data[i], on='Time', how='outer')

frame[:-1].to_csv('data.csv', index=False)

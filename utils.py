import os
import pandas as pd
from binance.client import Client
import warnings
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("display.max_columns", 10)

pd.options.mode.chained_assignment = None


def download_data(client, symbol, timeframe="1d", start='2018-01-01', stop='2022-12-31'):
    tmp = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", timeframe, start, stop),
                       columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                'Taker buy quote asset volume', 'Ignore'])

    tmp = tmp[['Close time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    tmp.columns = ['Time', f'Open_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}', f'Volume_{symbol}']
    tmp['Time'] = pd.to_datetime(tmp['Time'], unit='ms')

    return tmp


def get_data(symbols=None, timeframe=None):
    if symbols is None:
        symbols = ['BTC', 'BNB', 'XMR', 'BAT']
    if timeframe is None:
        timeframe = "1d"
    with open('vars.txt') as f:
        lines = f.readlines()

    api_key, secret_key = lines[0][:-2], lines[1]
    client = Client(api_key, secret_key)

    data = [download_data(client, i, timeframe=timeframe)[['Time', f'Close_{i}', f'Volume_{i}']] for i in symbols]

    frame = data[0]
    for i in range(1, len(data)):
        frame = frame.merge(data[i], on='Time', how='outer')

    frame[:-1].to_csv('data.csv', index=False)


def get_plot(data, symbol='BTC'):
    tmp = data.copy()
    tmp['Time'] = pd.to_datetime(tmp['Time'])
    plt.plot(tmp['Time'], tmp[f'Close_{symbol}'], lw=1)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{symbol}")


def basic_statistics(data):
    tmp = data.copy()
    info = tmp.describe()
    info.loc['kurtosis'] = tmp.kurtosis()
    info.loc['skew'] = tmp.skew()
    info.loc['volatility'] = [tmp[i].std() / tmp[i].mean() for i in info.columns]

    info = info.transpose()[['mean', '50%', 'std', 'volatility', 'skew', 'kurtosis']]
    info.columns = ['Średnia', 'Mediana', 'Odchylenie stand.', 'Wsp. zmienności', 'Skośność', 'Kurtoza']
    return info


def day_of_the_week_effect(data, target):
    tmp = data.copy()
    tmp['Time'] = pd.to_datetime(tmp['Time'])
    tmp = tmp[['Time', f'Close_{target}']]
    tmp['R'] = 0
    for i in tmp.index[1:]:
        tmp['R'][i] = ((tmp[f'Close_{target}'][i] / tmp[f'Close_{target}'][i - 1]) - 1) * 100

    tmp['Day'] = tmp['Time'].apply(lambda x: x.weekday())
    tmp['D0'], tmp['D1'], tmp['D2'], tmp['D3'], tmp['D4'], tmp['D5'], tmp['D6'] = 0, 0, 0, 0, 0, 0, 0
    for i in tmp.index:
        match tmp['Day'][i]:
            case 0:
                tmp['D0'][i] = 1
            case 1:
                tmp['D1'][i] = 1
            case 2:
                tmp['D2'][i] = 1
            case 3:
                tmp['D3'][i] = 1
            case 4:
                tmp['D4'][i] = 1
            case 5:
                tmp['D5'][i] = 1
            case 6:
                tmp['D6'][i] = 1

    # ANOVA

    return tmp.drop(columns=['Day'])


def volume_anlysis(data):
    pass

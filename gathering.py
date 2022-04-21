from binance.client import Client
import pandas as pd

client = Client("dEz6xgelIXues4sj4R94xFhIlkDHsUf9zRmtWCBstt48nWzIxkNo0KsigdaQQiXH",
                "ydrqbHy7NfNPIWZSL0rR2uMWrspi4Pvz1Baskt8eHEqMRfMCV6lWvVxSAImw1NSN")

daily = pd.DataFrame(client.get_historical_klines("BTCUSDT", '1d', '2018-01-01', '2021-12-31'),
                     columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                              'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                              'Ignore'])
daily = daily[['Close time', 'Open', 'High', 'Low', 'Close', 'Volume']]
daily['Close time'] = pd.to_datetime(daily['Close time'], unit='ms')
daily.to_csv("daily.csv", index=False)

min5 = pd.DataFrame(client.get_historical_klines("BTCUSDT", '5m', '2018-01-01', '2022-01-01'),
                     columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                              'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                              'Ignore'])
min5 = min5[['Close time', 'Open', 'High', 'Low', 'Close', 'Volume']]
min5['Close time'] = pd.to_datetime(min5['Close time'], unit='ms')
print(min5)
min5.to_csv("min5.csv", index=False)
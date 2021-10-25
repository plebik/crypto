import yfinance as yf
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta.trend import *
from ta.others import *

btc = yf.Ticker('BTC-USD')
df = btc.history(period="max")[['Open', 'High', 'Low', 'Close', 'Volume']]
df['awesome_oscillator'] = awesome_oscillator(df['High'], df['Low'])
df['kama'] = kama(df['Close'])
df['ppo'] = ppo(df['Close'])
df['pvo'] = pvo(df['Volume'])
df['roc'] = roc(df['Close'])
df['rsi'] = rsi(df['Close'])
df['stochrsi'] = stochrsi(df['Close'])
df['stoch'] = stoch(df['Close'], df['High'], df['Low'])
df['tsi'] = tsi(df['Close'])
df['ultimate_oscillator'] = ultimate_oscillator(df['High'], df['Low'], df['Close'])
df['williams_r'] = williams_r(df['High'], df['Low'], df['Close'])
df['acc_dist_index'] = acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
df['chaikin_money_flow'] = chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
df['ease_of_movement'] = ease_of_movement(df['High'], df['Low'], df['Volume'])
df['force_index'] = force_index(df['Close'], df['Volume'])
df['money_flow_index'] = money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
df['negative_volume_index'] = negative_volume_index(df['Close'], df['Volume'])
df['on_balance_volume'] = on_balance_volume(df['Close'], df['Volume'])
df['volume_price_trend'] = volume_price_trend(df['Close'], df['Volume'])
df['volume_weighted_average_price'] = volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
df['average_true_range'] = average_true_range(df['High'], df['Low'], df['Close'])
# bollinger bands, Donchian Channel, KeltnerChannel
df['ulcer_index'] = ulcer_index(df['Close'])
# adx, aroon
df['cci'] = cci(df['High'], df['Low'], df['Close'])
df['dpo'] = dpo(df['Close'])
df['ema_indicator'] = ema_indicator(df['Close'])
# ichimoku, kst, macd
df['mass_index'] = mass_index(df['High'], df['Low'])
# psar
df['sma'] = sma_indicator(df['Close'])
df['stc'] = stc(df['Close'])
df['trix'] = trix(df['Close'])
# vortex
df['cumulative_return'] = cumulative_return(df['Close'])
df['daily_return'] = daily_return(df['Close'])

print(df)

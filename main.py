import yfinance as yf
from stats import *

btc = yf.Ticker('BTC-USD')
df = btc.history(period="max")[['Close', 'Volume']]
info = descriptive_statistics(df)
dist_plot(df)

# period = 1
df['abs'] = growth(df['Close'], type='absolute')
df['relative'] = growth(df['Close'], type='relative')
df['log'] = growth(df['Close'], type='log')

growth_plot(df, btc.ticker)
if __name__ == '__main__':
    pass

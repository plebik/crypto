import yfinance as yf
from stats import *


btc = yf.Ticker('BTC-USD')
df = btc.history(period="max")[['Close', 'Volume']]
info = descriptive_statistics(df)
dist_plot(df)

df['abs'] = growth(df['Close'], type='absolute')
df['relative'] = growth(df['Close'], type='relative')
df['log'] = growth(df['Close'], type='log')

growth_plot(df, btc.ticker)
decomposition_plot(df['Close'])
test_adf(df['Close'])
test_kpps(df['Close'])

if __name__ == '__main__':
    pass

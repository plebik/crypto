import yfinance as yf
from stats import *
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

btc = yf.Ticker('BTC-USD')
df = btc.history(period="max")[['Close', 'Volume']]
info = descriptive_statistics(df)
dist_plot(df)

df['abs'] = growth(df['Close'], type='absolute')
df['relative'] = growth(df['Close'], type='relative')
df['log'] = growth(df['Close'], type='log')

growth_plot(df, btc.ticker)
decomposition_plot(df['Close'])
plot_acf(df['Close'])
plt.savefig('Autocorrelatiion.svg', format='svg')
test_adf(df['Close'])
test_kpps(df['Close'])
day_analysis(df['Close'])

if __name__ == '__main__':
    pass

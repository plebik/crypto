import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

plt.style.use('fivethirtyeight')

data = pd.read_csv('data.csv')
data['Time'] = pd.to_datetime(data['Time'])
clear_data = data.dropna()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(clear_data['Time'], clear_data['Close_BTC'], color='red', linewidth=1.5)
ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('BTC Price', fontsize=14)

ax1_1 = ax1.twinx()
# make a plot with different y-axis using second axis object
ax1_1.plot(clear_data['Time'], clear_data['Close_BNB'], color="blue", linewidth=1.5)
ax1_1.set_ylabel('BNB Price', fontsize=14)
ax1_1.set_ylim(top=max(clear_data['Close_BNB']) * 2)


plt.tick_params(axis='x', which='major', labelsize=10)
plt.tight_layout()
plt.grid(False)
plt.show()

# print(clear_data['Time'])

# model = ols('Close_btc ~ Close_bnb + Close_xmr + Close_bat', data=joint).fit()

# view model summary
# print(model.summary())

# print(durbin_watson(model.resid))

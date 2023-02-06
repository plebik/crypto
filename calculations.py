import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

tmp1 = pd.merge(daily_btc, daily_bnb, on='Time', how="outer", suffixes=('_btc', '_bnb'))
tmp2 = pd.merge(tmp1, daily_xmr, on='Time', how="outer")
joint = pd.merge(tmp2, daily_bat, on='Time', how="outer", suffixes=('_xmr', '_bat'))
joint.dropna(inplace=True)

for i in ['btc', 'bnb', 'xmr', 'bat']:
    joint[f'Close_{i}'] = pd.to_numeric(joint[f'Close_{i}'])

model = ols('Close_btc ~ Close_bnb + Close_xmr + Close_bat', data=joint).fit()

# view model summary
# print(model.summary())

print(durbin_watson(model.resid))

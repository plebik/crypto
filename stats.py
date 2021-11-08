import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from statistics import geometric_mean, harmonic_mean

warnings.filterwarnings("ignore")


def chronological(series, t=None):
    if t is None:
        t = len(series)
    output = np.zeros(series.shape[1])
    for i in range(t - 1):
        output[0] += (series[i][0] + series[i + 1][0]) / 2
        output[1] += (series[i][1] + series[i + 1][1]) / 2
    return output / (t - 1)


def descriptive_statistics(series):
    info = pd.DataFrame(columns=['close', 'volume'])
    info.loc['min'] = series.min().to_numpy()
    info.loc['max'] = series.max().to_numpy()
    info.loc['mean'] = series.mean().to_numpy()
    info.loc['harmonic'] = [harmonic_mean(series[x]) for x in series.columns]
    info.loc['chronological'] = chronological(series.to_numpy())
    info.loc['median'] = series.median().to_numpy()
    info.loc['range'] = info.loc['max'] - info.loc['min']
    info.loc['std'] = series.std().to_numpy()
    info.loc['var_coef'] = info.loc['std'] / info.loc['mean']
    info.loc['asymmetry'] = series.skew().to_numpy()
    info.loc['kurtosis'] = series.kurt().to_numpy()
    info.loc['G'] = [geometric_mean(series[x]) - 1 for x in series.columns]
    return info.round(2)


def dist_plot(series):
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    for i, j in enumerate(series.columns):
        _, bins, _ = axs[0, i].hist(series[j], density=True, alpha=0.5, )
        axs[0, i].set_title(j)
        mu, sigma = stats.norm.fit(series[j])
        best_fit_line = stats.norm.pdf(bins, mu, sigma)
        axs[0, i].plot(bins, best_fit_line)

    axs[1, 0].plot(series['Close'])
    axs[1, 1].plot(series['Volume'])
    plt.tight_layout()
    plt.savefig('Distribution.svg', format='svg')


def growth_plot(series, ticker=''):
    plt.style.use('seaborn')
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=300)
    axs[0].plot(series['Close'])
    axs[0].set_title(ticker)
    axs[1].plot(series['abs'])
    axs[1].set_title('Rates of Return')
    axs[2].plot(series['log'])
    axs[2].set_title('Logarithmic RoR')

    plt.tight_layout()
    plt.savefig('Rates of Return.svg', format='svg')


def growth(array, tau=1, type='absolute'):
    output = []
    for i in range(tau):
        output.append(np.NAN)
    if type == 'absolute':
        for i in range(tau, len(array)):
            output.append(array[i] - array[i - tau])
    elif type == 'relative':
        for i in range(tau, len(array)):
            output.append((array[i] - array[i - tau]) / array[i - tau])
    elif type == 'log':
        for i in range(tau, len(array)):
            output.append(np.log(array[i] / array[i - tau]))
    return output


def decomposition_plot(series):
    decomposition = seasonal_decompose(series, model='multiplicative', period=300)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    axs[0, 0].plot(series)
    axs[0, 0].set_title('Original Data')
    axs[0, 1].plot(trend)
    axs[0, 1].set_title('Trend')
    axs[1, 0].plot(seasonal)
    axs[1, 0].set_title('Seasonality')
    axs[1, 1].plot(residual)
    axs[1, 1].set_title('Residual')

    plt.tight_layout()
    plt.savefig('Decomposition.svg', format='svg')


def test_adf(series):
    test, p_value, _, _, crit, _ = adfuller(series)
    text = f"AUGMENTED DICKEY-FULLER TEST\n" \
           f"Test's statistics: {round(test, 4)}\n" \
           f"p-value: {round(p_value, 4)}\n" \
           f"Critical Values:\n" \
           f"1%: {round(crit['1%'], 4)}\n" \
           f"5%: {round(crit['5%'], 4)}\n" \
           f"10%: {round(crit['10%'], 4)}\n"

    if p_value > 0.05 and all(i < test for i in list(crit.values())):
        text += "\nNo reason to reject the null hypothesis. So, the time series is in fact non-stationary\n"
    else:
        text += "\nWe can reject the null hypothesis and take that the series is stationary\n"
    print(text)


def test_kpps(series):
    test, p_value, _, crit = kpss(series)

    text = f"KPPS TEST\nTest's statistics: {round(test, 4)}\n" \
           f"p-value: {round(p_value, 4)}\n" \
           f"Critical Values:\n" \
           f"10%: {round(crit['10%'], 4)}\n" \
           f"5%: {round(crit['5%'], 4)}\n" \
           f"2.5%: {round(crit['2.5%'], 4)}\n" \
           f"1%: {round(crit['1%'], 4)}\n"

    if all(i < test for i in list(crit.values())):
        text += "\nThere is evidence for rejecting the null hypothesis in favor of the alternative." \
                "Hence, the series is non-stationary\n"
    else:
        text += "\nNo evidence to reject null hypothesis. Hence, the series is stationary\n"
    print(text)


def day_analysis(series):
    days = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
    tmp_dict = {}

    for j in series.index[:7]:
        n = j.weekday()
        tmp_dict[days[n]] = series.values[n::7]

    frame = pd.DataFrame.from_dict(tmp_dict, orient='index').transpose()
    frame = frame[['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']]
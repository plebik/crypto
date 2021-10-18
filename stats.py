import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def harmonic_mean(series):
    output = np.zeros(series.shape[1])
    for i in series:
        output[0] += pow(i[0], -1)
        output[1] += pow(i[1], -1)
    return len(series) / output


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
    info.loc['harmonic'] = harmonic_mean(series.to_numpy())
    info.loc['chronological'] = chronological(series.to_numpy())
    info.loc['median'] = series.median().to_numpy()
    info.loc['range'] = info.loc['max'] - info.loc['min']
    info.loc['std'] = series.std().to_numpy()
    info.loc['var_coef'] = info.loc['std'] / info.loc['mean']
    info.loc['asymmetry'] = 3 * (info.loc['mean'] - info.loc['median']) / info.loc['std']
    info.loc['kurtosis'] = series.kurt().to_numpy()
    return info.round(2)


def dist_plot(series):
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=100)
    for i, j in enumerate(series.columns):
        _, bins, _ = axs[0, i].hist(series[j], density=True, alpha=0.5, )
        axs[0, i].set_title(j)
        mu, sigma = stats.norm.fit(series[j])
        best_fit_line = stats.norm.pdf(bins, mu, sigma)
        axs[0, i].plot(bins, best_fit_line)

    axs[1, 0].plot(series['Close'])
    axs[1, 1].plot(series['Volume'])
    plt.tight_layout()
    plt.savefig('distribution.png', dpi=300)


def growth_plot(series, ticker=''):
    plt.style.use('seaborn')
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=100)
    axs[0].plot(series['Close'])
    axs[0].set_title(ticker)
    axs[1].plot(series['abs'])
    axs[1].set_title('Rates of Return')
    axs[2].plot(series['log'])
    axs[2].set_title('Logarithmic RoR')

    plt.tight_layout()
    plt.savefig('Rates of Return.png', dpi=300)


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

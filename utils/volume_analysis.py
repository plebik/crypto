import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def preprocessing(crypto):
    tmp = crypto.data.copy()
    tmp.columns = ['P', 'V']
    tmp['R'] = np.log(tmp['P']) - np.log(tmp['P'].shift(1))
    tmp['D'] = [np.std(tmp.iloc[i - 20:i + 1]['R']) for i in range(tmp.shape[0])]
    tmp.columns = [f'{i}_{crypto.name}' for i in ['P', 'V', 'R', 'D']]

    return tmp, crypto.name


def basic_stats(data):
    statistics = pd.DataFrame(columns=['mean', 'median', 'std', 'volatility', 'skew', 'kurtosis'])
    tmp = data[0].describe().transpose()[['mean', '50%', 'std']]
    tmp['volatility'] = (tmp['std'] / tmp['mean']) * 100
    tmp['skew'] = data[0].skew()
    tmp['kurtosis'] = data[0].kurtosis()
    tmp = tmp.applymap(
        lambda x: '{:.2e}'.format(x) if (0.01 > x > -0.01) or x < -1000000 or x > 1000000 else x)
    tmp = tmp.applymap(lambda x: x if isinstance(x, str) else round(x, 2))
    tmp.columns = ['mean', 'median', 'std', 'volatility', 'skew', 'kurtosis']

    statistics = pd.concat([statistics, tmp])
    statistics = statistics.reindex([f'{i}_{data[1]}' for i in ['P', 'R', 'V']])

    return statistics


def pearson(data):
    data = data.dropna()
    pearson_corr, p_value = pearsonr(data[data.columns[0]], data[data.columns[1]])
    if p_value < 0.01:
        p_value = '{:.2e}'.format(p_value)
    else:
        p_value = round(p_value, 2)

    return f"{round(pearson_corr, 2)}[{p_value}]"


def corr(data, dates):
    results = pd.DataFrame(columns=[f'P_{data[1]}', f'R_{data[1]}', f'D_{data[1]}'])

    counter = 0
    while counter < len(dates):
        if counter == 0:
            try:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter]) + 1]
            except KeyError:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter + 1]) + 1]
                counter += 1
        elif counter == len(dates) - 1:
            tmp = data[0].iloc[data[0].index.get_loc(dates[counter]):]
        else:
            tmp = data[0].iloc[
                  data[0].index.get_loc(dates[counter - 1]):data[0].index.get_loc(dates[counter])]

        results.loc[f'period_{counter + 1}'] = [pearson(tmp[[f'{j}_{data[1]}', f'V_{data[1]}']]) for j in
                                                ['P', 'R', 'D']]
        counter += 1

    results.loc['all'] = [pearson(data[0][[f'{j}_{data[1]}', f'V_{data[1]}']]) for j in ['P', 'R', 'D']]

    return results


def dickey_fuller(data):
    decision = ''

    result = sm.tsa.stattools.adfuller(data.dropna(), regression='c')
    stat, p_value = round(result[0], 4), result[1]

    if p_value >= 0.05:
        decision = 'Y'
    else:
        if stat >= result[4]['5%']:
            decision = 'Y'

    if -0.01 < p_value < 0.01:
        p_value = '{:.2e}'.format(p_value)
    else:
        p_value = round(p_value, 2)

    return [stat, p_value, decision]


def auto_corr(data, dates):
    results = pd.DataFrame(columns=[f'P_{data[1]}', f'R_{data[1]}', f'D_{data[1]}', f'V_{data[1]}'])

    counter = 0
    while counter < len(dates):
        if counter == 0:
            try:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter]) + 1]
            except KeyError:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter + 1]) + 1]
                counter += 1
        elif counter == len(dates) - 1:
            tmp = data[0].iloc[data[0].index.get_loc(dates[counter]):]
        else:
            tmp = data[0].iloc[
                  data[0].index.get_loc(dates[counter - 1]):data[0].index.get_loc(dates[counter])]

        results.loc[f'period_{counter + 1}'] = [dickey_fuller(tmp[f'{j}_{data[1]}'].dropna())[2] for j in
                                                ['P', 'R', 'D', 'V']]

        counter += 1

    return results


def johansen(data, X, Y):
    result = coint_johansen(data[[X, Y]].dropna(), det_order=0, k_ar_diff=1)
    mapping = {0: [result.eig[0], result.lr1[0], result.cvt[0][1], result.lr2[0], result.cvm[0][1]],
               1: [result.eig[1], result.lr1[1], result.cvt[1][1], result.lr2[1], result.cvm[1][1]]}

    tmp = pd.DataFrame(mapping, columns=[0, 1],
                       index=['Eigenvalue', 'Trace stat', 'Trace crit', 'Lmax stat', 'Lmax crit']).transpose()
    tmp['Result_Trace'] = tmp['Trace stat'] > tmp['Trace crit']
    tmp['Result_Lmax'] = tmp['Lmax stat'] > tmp['Lmax crit']

    tmp.index = [f'{X} & {Y}_0', f'{X} & {Y}_1']
    return round(tmp[['Eigenvalue', 'Trace stat', 'Lmax stat', 'Result_Trace', 'Result_Lmax']], 3)


def order(data, X, Y):
    # model = sm.OLS(data[Y], sm.add_constant(data[X])).fit()
    # bic = model.bic

    return 4


def granger(data, X, Y, lag=7, verbose=False):

    results = grangercausalitytests(data[[Y, X]].dropna(), maxlag=lag, verbose=False)
    stat, p_value = results[lag][0]['ssr_chi2test'][:-1]
    # lrtest = [round(i, 4) for i in results[lag][0]['lrtest']]
    if verbose:
        print("\nGranger Causality\n"
              f"number of lags (no zero) {lag}\n"
              f"ssr based chi2 test:\t\tF={stat}\tp={p_value}")
        # f"likelihood ratio test:\t\tF={lrtest[0]}\tp={lrtest[1]}\tdf_num={lrtest[2]}\n")

    # causality statement
    if p_value >= 0.05:
        decision = ''
    else:
        decision = True

    if p_value < 0.01:
        p_value = '{:.2e}'.format(p_value)
    else:
        p_value = round(p_value, 4)

    return pd.Series([f'{X} → {Y}'] + [round(stat, 2), p_value, decision])


def period_granger(data, dates):
    results = pd.DataFrame(columns=[f'period_{i}' for i in range(1, len(dates) + 1)])

    counter = 0
    while counter < len(dates):
        group = []
        index = []
        if counter == 0:
            try:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter]) + 1]
            except KeyError:
                tmp = data[0].iloc[0:data[0].index.get_loc(dates[counter + 1]) + 1]
                counter += 1
        elif counter == len(dates) - 1:
            tmp = data[0].iloc[data[0].index.get_loc(dates[counter]):]
        else:
            tmp = data[0].iloc[
                  data[0].index.get_loc(dates[counter - 1]):data[0].index.get_loc(dates[counter])]

        for i in ['P', 'R', 'D']:
            index1, index2 = f'{i}_{data[1]}', f'V_{data[1]}'
            index.append(index1 + ' → ' + index2)
            index.append(index2 + ' → ' + index1)
            group.append(granger(tmp.dropna(), index1, index2).tail(1).values[0])
            group.append(granger(tmp.dropna(), index1, index2).tail(1).values[0])

        results[f'period_{counter + 1}'] = group
        results.index = index

        counter += 1

    return results


def get_plot(data, symbol='BTC'):
    tmp = data.copy()
    tmp['Time'] = pd.to_datetime(tmp['Time'])
    plt.plot(tmp['Time'], tmp[f'Close_{symbol}'], lw=1)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{symbol}")


def volume_analysis(cryptos, plot=False, verbose=True):
    names = []
    statistics = []
    correlations = []
    overall_autocorrelations = []
    period_autocorrelations = []
    johansens = []
    overall_grangers = []
    period_grangers = []
    transoformed = []
    dates = ['2018-12-15', '2019-06-26', '2020-03-12', '2021-04-13', '2021-07-20', '2021-11-08']

    for crypto in cryptos:
        names.append(crypto.name)

        # data preparation
        data = preprocessing(crypto)
        transoformed.append(data[0][f'R_{data[1]}'])

        # basic statistics
        statistic = basic_stats(data)
        statistics.append(statistic)

        # correlations
        correlation = corr(data, dates)
        correlations.append(correlation)

        # autocorrelations
        overall_autocorrelation = pd.DataFrame(0, index=['Statistic', 'p_value', 'Stationarity'],
                                               columns=[f'{i}_{data[1]}' for i in ['P', 'R', 'D', 'V']])
        for i in overall_autocorrelation.columns:
            overall_autocorrelation[i] = dickey_fuller(data[0][i].dropna())

        overall_autocorrelations.append(overall_autocorrelation)

        period_autocorrelations.append(auto_corr(data, dates))

        for i in ['P', 'R', 'D']:
            # johansen test
            johansens.append(johansen(data[0], f'{i}_{data[1]}', f'V_{data[1]}'))

            # The linear Granger causality test.
            overall_grangers.append(granger(data[0], f'{i}_{data[1]}', f'V_{data[1]}'))
            overall_grangers.append(granger(data[0], f'V_{data[1]}', f'{i}_{data[1]}'))

        period_grangers.append(period_granger(data, dates))
        # plot to choose periods
        if plot and crypto.name == 'BTC':
            plt.plot(data[0]['P_BTC'], lw=1, color='black')
            for d in dates:
                vline_date = dt.datetime.strptime(d, '%Y-%m-%d').date()
                plt.axvline(vline_date, color='black', lw=1)
            plt.tight_layout()
            plt.savefig("plots/volume_analysis.png")
            plt.show()

    transformed_df = pd.concat(transoformed, axis=1)
    # returns_granger = pd.DataFrame(columns=['Direction', 'Stat', 'p_value', 'Causality'])
    returns_granger_list = []
    for i in transformed_df.columns:
        right = transformed_df.drop(columns=[i])
        for j in right.columns:
            returns_granger_list.append(granger(transformed_df, i, j))

    returns_granger = pd.DataFrame(returns_granger_list)
    returns_granger.columns = ['Direction', 'Stat', 'p_value', 'Causality']
    print(returns_granger)


    # grouping and setting the order
    grouped_statistics = pd.concat(statistics)
    grouped_statistics = grouped_statistics.reindex([f'{i}_{j}' for i in ['P', 'R', 'V'] for j in names])

    transposed = [df.T for df in correlations]
    grouped_correlations = pd.concat(transposed)
    grouped_correlations = grouped_correlations.reindex([f'{i}_{j}' for i in ['P', 'R', 'D'] for j in names])

    transposed = [df.T for df in overall_autocorrelations]
    grouped_overall_autocorrelations = pd.concat(transposed)
    grouped_overall_autocorrelations = grouped_overall_autocorrelations.reindex(
        [f'{i}_{j}' for i in ['P', 'R', 'D', 'V'] for j in names])

    transposed = [df.T for df in period_autocorrelations]
    grouped_period_autocorrelations = pd.concat(transposed)
    grouped_period_autocorrelations = grouped_period_autocorrelations.reindex(
        [f'{i}_{j}' for i in ['P', 'R', 'D', 'V'] for j in names])

    grouped_johansens = pd.concat(johansens)

    grouped_overall_grangers = pd.DataFrame(overall_grangers)
    grouped_overall_grangers.columns = ['Direction', 'Stat', 'p_value', 'Causality']
    grouped_overall_grangers.index = grouped_overall_grangers['Direction']
    grouped_overall_grangers.drop(columns=['Direction'], inplace=True)

    grouped_period_grangers = pd.concat(period_grangers)


    if verbose:
        print()
        for i in [grouped_statistics, grouped_correlations, grouped_overall_autocorrelations,
                  grouped_period_autocorrelations, grouped_johansens, grouped_overall_grangers,
                  grouped_period_grangers]:
            print(i)
            print()

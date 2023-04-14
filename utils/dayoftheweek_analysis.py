import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA

from utils.volume_analysis import dickey_fuller


def basic_statistics(data, name):
    frame = pd.DataFrame(columns=['All days', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                  'Sunday'])

    frame['All days'] = [data['R'].mean(), data['R'].std(), (data['R'].std() / data['R'].mean()) * 100,
                         data['R'].skew(), data['R'].kurt()]

    for i in frame.columns[1:]:
        tmp = data[data[i] == 1]

        frame[i] = [tmp['R'].mean(), tmp['R'].std(), (tmp['R'].std() / tmp['R'].mean()) * 100,
                    tmp['R'].skew(), tmp['R'].kurt()]

    frame.index = [f'{name}_{i}' for i in ['mean', 'std', 'volatility', 'skew', 'kurtosis']]

    return round(frame, 2)


def yearly_returns(data, name):
    frame = pd.DataFrame(columns=['All days', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                  'Sunday'])

    frame['All days'] = [data[(data.index >= f'{i}-01-01') & (data.index <= f'{i}-12-31')]['R'].mean() for i in
                         data.index.year.unique()]
    for i in frame.columns[1:]:
        tmp = data[data[i] == 1]

        frame[i] = [tmp[(tmp.index >= f'{i}-01-01') & (tmp.index <= f'{i}-12-31')]['R'].mean() for i in
                    tmp.index.year.unique()]

    frame.index = [f'{name}_{i}' for i in data.index.year.unique()]
    return round(frame, 2)


def autocorrelation(data, name, lag=7):
    acf, q, p = sm.tsa.acf(data.dropna(), nlags=lag + 1, qstat=True)
    return pd.DataFrame([acf[:-1], q, p], index=[f'{name}_ACF', f'{name}_Q-Stat', f'{name}_p-value'])


def arch(data):
    data = data.asfreq('D')
    model = ARIMA(data.dropna(), order=(2, 1, 0)).fit()
    test_results = het_arch(model.resid)
    return test_results


def preparation(data, name):
    frame = pd.DataFrame(columns=['Stationary', 'Autocorrelation', ''])
    stationary = dickey_fuller(data['R'])[2]
    arch_results = arch(data['R'])
    # print(arch_results)


def ols(data):
    data.dropna(inplace=True)
    y = data['R']
    X = data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    results = []
    for param, p_value in zip(model.params[1:], model.pvalues[1:]):

        if -0.01 < p_value < 0.01:
            p_value = '{:.2e}'.format(p_value)
        else:
            p_value = round(p_value, 3)

        results.append(f'{round(param, 3)} [{p_value}]')

    return results, [round(model.rsquared, 3), round(sm.stats.stattools.durbin_watson(model.resid), 3),
                     round(model.llf, 3)]


def analysis(data, name):
    columns = [f'{name}_{i}' for i in ['OLS', 'GARCH()', 'GARCH()']]
    frame = pd.DataFrame('', columns=columns,
                         index=['Monday(M)', 'Tuesday(M)', 'Wednesday(M)', 'Thursday(M)', 'Friday(M)', 'Saturday(M)',
                                'Sunday(M)', 'rt-1', 'Cst(V)', 'Monday(V)', 'Tuesday(V)', 'Wednesday(V)', 'Thursday(V)',
                                'Friday(V)', 'Saturday(V)', 'Sunday(V)', 'ARCH(Alpha1)', 'GARCH(Beta1)'])
    stats = pd.DataFrame('', columns=columns, index=['R2', 'DW', 'logL', 'AIC', 'SIC', 'Alpha[1]+Beta[1]'])

    kmnk = ols(data)

    frame[f'{name}_OLS'] = kmnk[0] + ['' for w in range(11)]
    stats[f'{name}_OLS'] = kmnk[1] + ['' for w in range(3)]

    return [frame, stats]


def dayoftheweek_analysis(cryptos, verbose=True):
    basics = []
    yearly = []
    coefs = []
    stats = []
    preparations = []
    autocorrs = []
    datas = []
    for crypto in cryptos:
        name = crypto.name
        data = crypto.data[['Close']]
        data['R'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
        tmp = data.copy()
        tmp.columns = [f"Close_{name}", f"R_{name}"]
        datas.append(tmp)
        for i, j in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
            data[j] = list(map(lambda x: 1 if x == i else 0, data.index.weekday))



        basics.append(basic_statistics(data, name))
        yearly.append(yearly_returns(data, name))
        autocorrs.append(autocorrelation(data['R'], name))
        preparations.append(preparation(data, name))
        coef, stat = analysis(data, name)
        coefs.append(coef)
        stats.append(stat)

    grouped_basics = pd.concat(basics)
    grouped_yearly = pd.concat(yearly)
    grouped_coefs = pd.concat(coefs, axis=1)
    grouped_stats = pd.concat(stats, axis=1)
    grouped_autocorrs = pd.concat(autocorrs).transpose()
    joint_data = pd.concat(datas, axis=1)

    joint_data.to_csv("data/data.csv")

    if verbose:
        print()
        for i in [grouped_basics, grouped_yearly, grouped_coefs, grouped_stats, grouped_autocorrs]:
            print(i)
            print()
    #
    # df.dropna(inplace=True)
    # # Equation 1
    # X_1 = df[df.columns[1:]]
    # X_1 = sm.add_constant(X_1)
    # y_1 = df['return']
    #
    # model_1 = sm.OLS(y_1, X_1).fit()
    # print(model_1.summary())
    #
    # s = 3  # TUTAJ SPRAWDZIĆ
    # for i in range(1, s + 1):
    #     df[f'delayed_{i}'] = df['return'].shift(-i)
    #
    # df.dropna(inplace=True)
    #
    # X_2 = df[df.columns[1:]]
    # X_2 = sm.add_constant(X_2)
    # y_2 = df['return']
    #
    # model_2 = sm.OLS(y_2, X_2).fit()
    # print(model_2.summary())
    #
    # return df.head(10)

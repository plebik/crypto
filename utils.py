import datetime as dt
import os
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from binance.client import Client
from scipy.stats import pearsonr

plt.style.use('fivethirtyeight')
warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.set_option("display.max_columns", 10)

pd.options.mode.chained_assignment = None


class Crypto:
    def __init__(self, name):
        self.data = self.fetch_data(name)
        # self.data['return'] = 100 * np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data.index = pd.to_datetime(self.data.index)
        self.name = name

    def __str__(self):
        return self.name

    @staticmethod
    def fetch_data(symbol, timeframe="1d", start='2018-01-01', stop='2022-12-31'):
        with open('vars.txt') as f:
            lines = f.readlines()

        api_key, secret_key = lines[0][:-2], lines[1]

        client = Client(api_key, secret_key)
        tmp = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", timeframe, start, stop),
                           columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                    'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                    'Taker buy quote asset volume', 'Ignore'])

        tmp = tmp[['Close time', 'Close', 'Volume']]
        tmp.columns = ['Time', 'Close', 'Volume']
        tmp['Time'] = pd.to_datetime(tmp['Time'], unit='ms')
        tmp['Time'] = tmp['Time'].dt.date
        tmp.index = tmp['Time']
        tmp.drop(columns=['Time'], inplace=True)
        tmp['Close'] = pd.to_numeric(tmp['Close'])
        tmp['Volume'] = pd.to_numeric(tmp['Volume'])

        return tmp

    @staticmethod
    def daily_subsets(data):
        if isinstance(data, pd.DataFrame):
            data['day'] = list(map(lambda x: date.weekday(x), data.index))
            dictionary = {
                'all_days': data,
                'monday': data[data['day'] == 0],
                'tuesday': data[data['day'] == 1],
                'wednesday': data[data['day'] == 2],
                'thursday': data[data['day'] == 3],
                'friday': data[data['day'] == 4],
                'saturday': data[data['day'] == 5],
                'sunday': data[data['day'] == 6]
            }
            for i in dictionary.values():
                i.drop(columns=['day'], inplace=True)

            return dictionary
        else:
            raise Exception("The object passed to the function has to be a dataframe")

    @staticmethod
    def annual_subsets(data):
        if isinstance(data, pd.DataFrame):
            data['year'] = list(map(lambda x: x.year, data.index))
            dictionary = {}
            for i in np.unique(data['year']):
                dictionary[i] = data[data['year'] == i]

            for i in dictionary.values():
                i.drop(columns=['year'], inplace=True)

            return dictionary
        else:
            raise Exception("The object passed to the function has to be a dataframe")

    @staticmethod
    def basic_statistics(data):
        frame = pd.DataFrame([data.mean().values[0],
                              data.std().values[0],
                              data.std().values[0] / data.mean().values[0],
                              data.skew().values[0],
                              data.kurtosis().values[0]],
                             index=['mean', 'std', 'volatility', 'skew', 'kurtosis'],
                             columns=['value'])
        return round(frame, 2)

    @staticmethod
    def basic_statistics_for_each_day(data):
        daily_subsets = Crypto.daily_subsets(data)
        statistics = [Crypto.basic_statistics(i) for i in daily_subsets.values()]
        frame = pd.concat(statistics, axis=1)
        frame.columns = ["all_days", 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

        return frame

    @staticmethod
    def average_daily_returns_indices_by_annual_sub_periods(data):
        annual_subsets = Crypto.annual_subsets(data)
        daily_subsets = {}
        for i, j in annual_subsets.items():
            daily_subsets[i] = Crypto.daily_subsets(j)
            for k in daily_subsets[i].keys():
                daily_subsets[i][k] = round(daily_subsets[i][k].mean().values[0], 2)

        return pd.DataFrame(daily_subsets.values(), index=list(daily_subsets.keys()))

    def preliminary_analysis(self):
        # ADF

        # BOX
        pass

    def day_of_the_week_analysis(self):
        df = self.data[['return']].copy()
        df['Monday'] = list(map(lambda x: 1 if x == 0 else 0, df.index.weekday))
        df['Tuesday'] = list(map(lambda x: 1 if x == 1 else 0, df.index.weekday))
        df['Wednesday'] = list(map(lambda x: 1 if x == 2 else 0, df.index.weekday))
        df['Thursday'] = list(map(lambda x: 1 if x == 3 else 0, df.index.weekday))
        df['Friday'] = list(map(lambda x: 1 if x == 4 else 0, df.index.weekday))
        df['Saturday'] = list(map(lambda x: 1 if x == 5 else 0, df.index.weekday))
        df['Sunday'] = list(map(lambda x: 1 if x == 6 else 0, df.index.weekday))

        df.dropna(inplace=True)
        # Equation 1
        X_1 = df[df.columns[1:]]
        X_1 = sm.add_constant(X_1)
        y_1 = df['return']

        model_1 = sm.OLS(y_1, X_1).fit()
        print(model_1.summary())

        s = 3  # TUTAJ SPRAWDZIĆ
        for i in range(1, s + 1):
            df[f'delayed_{i}'] = df['return'].shift(-i)

        df.dropna(inplace=True)

        X_2 = df[df.columns[1:]]
        X_2 = sm.add_constant(X_2)
        y_2 = df['return']

        model_2 = sm.OLS(y_2, X_2).fit()
        print(model_2.summary())

        return df.head(10)

    def event_analysis(self):
        # top_10 = self.data['']
        plt.plot(np.cumsum(self.data['return']))
        plt.show()

    @staticmethod
    def volume_analysis(cryptos, plot=False):
        names = []
        statistics = []
        correlations = []
        overall_autocorrelations = []
        period_autocorrelations = []
        dates = ['2018-12-15', '2019-06-26', '2020-03-12', '2021-04-13', '2021-07-20', '2021-11-08']

        for crypto in cryptos:
            names.append(crypto.name)

            # data preparation
            data = preprocessing(crypto)

            # basic statistics
            statistic = basic_stats(data)
            statistics.append(statistic)

            # correlations
            correlation = corr(data, dates)
            correlations.append(correlation)

            # autocorrelations
            overall_autocorrelation = pd.DataFrame(0, index=['Statistic', 'p_value'],
                                                   columns=[f'{i}_{data[1]}' for i in ['P', 'R', 'D', 'V']])
            for i in overall_autocorrelation.columns:
                overall_autocorrelation[i] = dickey_fuller(data[0][i].dropna())[:2]

            overall_autocorrelations.append(overall_autocorrelation)

            period_autocorrelation = auto_corr(data, dates)
            period_autocorrelations.append(period_autocorrelation)

            # johansen test

            # The linear Granger causality test.

            # plot to choose periods
            if plot and crypto.name == 'BTC':
                plt.plot(data[0]['P_BTC'], lw=1, color='black')
                for d in dates:
                    vline_date = dt.datetime.strptime(d, '%Y-%m-%d').date()
                    plt.axvline(vline_date, color='black', lw=1)
                plt.tight_layout()
                plt.savefig("plots/volume_analysis.png")
                plt.show()

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

        return grouped_statistics, grouped_correlations, grouped_overall_autocorrelations, grouped_period_autocorrelations

        #
        # # #
        # # johansen_result = coint_johansen(df[[f'P_{self.name}', f'V_{self.name}']], det_order=0, k_ar_diff=0)
        # # print(johansen_result.trace_stat_crit_vals)

        # return stats, overall_corr, period_corr.transpose(), overall_autocorr.transpose(), period_autocorr
        # return period_autocorr
        #


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
    decision = True

    result = sm.tsa.stattools.adfuller(data.dropna(), regression='c')
    stat, p_value = round(result[0], 4), result[1]

    if p_value >= 0.05:
        decision = False
    else:
        if stat >= result[4]['5%']:
            decision = False

    if p_value < 0.01:
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


def get_plot(data, symbol='BTC'):
    tmp = data.copy()
    tmp['Time'] = pd.to_datetime(tmp['Time'])
    plt.plot(tmp['Time'], tmp[f'Close_{symbol}'], lw=1)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{symbol}")

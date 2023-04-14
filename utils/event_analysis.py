import pandas as pd


def event(data, event_date, back=2, forward=2):
    tmp = data.copy()
    tmp['R'] = (tmp['Close'] - tmp['Close'].shift(1)) * 100 / tmp['Close']

    event_window_start = pd.to_datetime(event_date) - pd.DateOffset(days=back)
    event_window_end = pd.to_datetime(event_date) + pd.DateOffset(days=forward)
    one_month_ago = pd.to_datetime(event_window_start) - pd.DateOffset(months=1)

    before_event = tmp.loc[(tmp.index >= one_month_ago) & (tmp.index < event_window_start)]
    event_window = tmp.loc[(tmp.index >= event_window_start) & (tmp.index <= event_window_end)]

    event_window['E(R)'] = before_event['R'].mean()
    event_window['AR'] = event_window['R'] - event_window['E(R)']

    return event_window['AR']


def event_analysis(alternatives, classics, event_date, back=2, forward=2, verbose=True):
    classic_ARs, alternative_ARs = [], []
    classic_CARs, alternative_CARs = [], []
    classic_names, alternative_names = [], []

    for classic, alternative in zip(classics, alternatives):
        result_classic = event(classic.data, event_date, back, forward)
        classic_ARs.append(result_classic)
        classic_CARs.append(result_classic.sum())
        classic_names.append(classic.name)

        result_alternative = event(alternative.data, event_date, back, forward)
        alternative_ARs.append(result_alternative)
        alternative_CARs.append(result_alternative.sum())
        alternative_names.append(alternative.name)

    classic_ACARs = sum(classic_CARs) / len(classic_CARs)
    alternative_ACARs = sum(alternative_CARs) / len(alternative_CARs)

    frame = pd.DataFrame(columns=alternative_names + classic_names)

    joined = alternative_ARs + classic_ARs
    for i in range(frame.shape[1]):
        frame[frame.columns[i]] = joined[i]

    frame.index = range(-back, forward + 1)
    frame.loc['CAR'] = alternative_CARs + classic_CARs

    if verbose:
        print()
        print(round(frame, 2), end="\n\n")
        print(f"Alternative ACAR: {round(alternative_ACARs, 2)}")
        print(f"Classic ACAR: {round(classic_ACARs, 2)}\n")

    return round(frame, 2), round(alternative_ACARs, 2), round(classic_ACARs, 2)

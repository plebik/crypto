import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

frame = pd.read_csv('data.csv').dropna()
frame['Time'] = pd.to_datetime(frame['Time'])


def get_plot(data, name1, name2):
    fig, ax = plt.subplots()
    ax.plot(data['Time'], data[f'Close_{name1}'], color='red', linewidth=1.5)
    ax.set_xlabel('Time', fontsize=14, weight="bold")
    ax.set_ylabel('BTC Price', fontsize=14, weight="bold")

    ax2 = ax.twinx()
    ax2.plot(data['Time'], data[f'Close_{name2}'], color="blue", linewidth=1.5)
    ax2.set_ylabel('BNB Price', fontsize=14, weight="bold")
    ax2.set_ylim(top=max(data[f'Close_{name2}']) * 2)

    ax2.set_title(f"{name1} vs {name2}\nρ={round(data[f'Close_{name1}'].corr(frame[f'Close_{name2}']), 2)}",
                  weight="bold")
    fig.autofmt_xdate()
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.tight_layout()
    plt.grid(False)

    plt.savefig(f"plots/{name1} vs {name2}")


os.makedirs("plots", exist_ok=True)
for i in ['BNB', 'XMR', 'BAT']:
    get_plot(frame, "BTC", i)

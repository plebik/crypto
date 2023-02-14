from utils import *

if __name__ == '__main__':
    # get_data()
    data = pd.read_csv('data.csv')
    get_plot(data)
    # print(day_of_the_week_effect(data, 'BTC'))
    # print(basic_statistics(data))

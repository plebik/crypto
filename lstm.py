import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

btc = yf.Ticker('BTC-USD')
df = btc.history(period="max")[['Close']]

# Period to look behind
period = 60

training_data_len = int(np.ceil(len(df) * .8))

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
# Splitting the data
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len - period:, :]
X_train, y_train = [], []

for i in range(period, len(train_data)):
    X_train.append(train_data[i - 60:i])
    y_train.append(train_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Building a model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', )
model.fit(X_train, y_train, epochs=3)

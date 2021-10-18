import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

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
X_test = []
y_test = scaled_data[training_data_len:]

for i in range(period, len(train_data)):
    X_train.append(train_data[i - 60:i])
    y_train.append(train_data[i])

for i in range(period, len(test_data)):
    X_test.append(test_data[i - 60:i])

X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)



# Building a model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=30)
mc = ModelCheckpoint('best_model', monitor='val_loss', save_best_only=True)
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es, mc])

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

train = df['Close'][:training_data_len]
test = df['Close'][training_data_len:]

plt.style.use('fivethirtyeight')

plt.figure(figsize=(10, 8))
plt.title(btc.ticker)
plt.plot(range(len(train)), train, label='train')
plt.plot(range(len(train), len(df)), test, label='test')
plt.plot(range(len(train), len(df)), predictions, label='pred')
plt.legend()

plt.savefig('LSTM.png', dpi=300)

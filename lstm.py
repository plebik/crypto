import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import numpy as np
from stats import growth
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ticker = yf.Ticker('BTC-USD')
df = ticker.history(period="max")[['Close']]
df['abs'] = growth(df['Close'], type='absolute')
df.drop(columns=['Close'], inplace=True)
df.dropna(inplace=True)

# Period to look behind
period = 30


training_data_len = int(np.ceil(len(df) * .8))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len - period:, :]
X_train, y_train, X_test = [], [], []
y_test = scaled_data[training_data_len:]

for i in range(period, len(train_data)):
    X_train.append(train_data[i - period:i])
    y_train.append(train_data[i])

for i in range(period, len(test_data)):
    X_test.append(test_data[i - period:i])

X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
# Building a model
model = Sequential()
model.add(BatchNormalization())
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(LSTM(50, return_sequences=False))
model.add(BatchNormalization())
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint('LSTM.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es, mc])

loaded_model = load_model('LSTM.h5')

train = loaded_model.predict(X_train)
test = loaded_model.predict(X_test)

plt.style.use('ggplot')

plt.figure(figsize=(10, 8))
plt.title(ticker.ticker)
plt.plot(df.index, scaled_data, label='original')
plt.plot(df.index[period:training_data_len], train, label='train')
plt.plot(df.index[training_data_len:], test, label='test')
plt.legend()

plt.savefig('LSTM.svg', format='svg')
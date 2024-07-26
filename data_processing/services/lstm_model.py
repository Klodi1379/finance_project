import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data(df, column, look_back=1):
    data = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y), scaler

def create_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df, column='Close', look_back=1):
    X, Y, scaler = prepare_data(df, column, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = create_lstm_model(look_back)
    model.fit(X, Y, epochs=10, batch_size=1, verbose=2)
    return model, scaler

def predict_lstm_model(model, df, scaler, column='Close', look_back=1):
    data = df[column].values.reshape(-1, 1)
    data = scaler.transform(data)
    X = []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

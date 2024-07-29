import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def handle_nan_values(df):
    for column in df.columns:
        df[column] = df[column].interpolate(method='linear')
        df[column] = df[column].fillna(method='bfill').fillna(method='ffill')
    return df

def prepare_data(df, target_column, look_back=60):
    df = handle_nan_values(df)  # Handle NaN values before processing
    data = df[[target_column] + [col for col in df.columns if col != target_column]]
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        Y.append(scaled_data[i, 0])
    
    if not X:  # If there's not enough data for the given look_back
        return None, None, scaler
    
    return np.array(X), np.array(Y), scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_lstm_model(df, target_column='Close', look_back=60, epochs=10, batch_size=32):
    X, Y, scaler = prepare_data(df, target_column, look_back)
    
    if X is None or Y is None:
        return None, scaler, None  # Return None for model and history if there's not enough data
    
    model = create_lstm_model((X.shape[1], X.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X, Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler, history

def predict_lstm_model(model, df, scaler, target_column='Close', look_back=60):
    if model is None:
        return pd.DataFrame(index=df.index, columns=['Prediction'])  # Return empty DataFrame if model is None
    
    df = handle_nan_values(df)  # Handle NaN values before making predictions
    data = df[[target_column] + [col for col in df.columns if col != target_column]]
    scaled_data = scaler.transform(data)
    
    X = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
    X = np.array(X)
    
    if X.size == 0:  # If there's not enough data for prediction
        return pd.DataFrame(index=df.index, columns=['Prediction'])
    
    predictions = model.predict(X)
    
    inv_predictions = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), data.shape[1]-1))]))[:, 0]
    
    predictions_df = pd.DataFrame(index=df.index, columns=['Prediction'])
    predictions_df.iloc[look_back:, 0] = inv_predictions
    
    return predictions_df

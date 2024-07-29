import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def train_arima_model(df, column='Close', order=(5,1,0)):
    # Ensure the dataframe has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            raise ValueError("DataFrame must have a 'Date' column")
    
    df.index = pd.to_datetime(df.index)
    
    # Sort the index to ensure it's in chronological order
    df = df.sort_index()
    
    # Remove any NaN values
    df = df.dropna()
    
    if df.empty:
        raise ValueError("No valid data after removing NaN values")
    
    # Use the 'Close' column directly
    model = ARIMA(df[column], order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima_model(model, steps=10):
    forecast = model.forecast(steps=steps)
    last_date = model.model.data.dates[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast, index=forecast_index)
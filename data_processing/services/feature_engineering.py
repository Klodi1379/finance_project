import pandas as pd
import numpy as np

def add_moving_average(df, window=20):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def add_daily_return(df):
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def add_technical_indicators(df):
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)

    return df
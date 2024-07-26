import pandas as pd

def add_moving_average(df, window=20):
    df['MA'] = df['Close'].rolling(window=window).mean()
    return df

def add_daily_return(df):
    df['Daily_Return'] = df['Close'].pct_change()
    return df
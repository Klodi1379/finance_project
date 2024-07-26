import pandas as pd
import numpy as np

def remove_outliers(df, column, n_sigmas=3):
    mean = df[column].mean()
    std = df[column].std()
    df = df[(df[column] <= mean + (n_sigmas * std)) & 
            (df[column] >= mean - (n_sigmas * std))]
    return df

def fill_missing_values(df):
    return df.ffill().bfill()
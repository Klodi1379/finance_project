import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

def save_to_database(df, symbol):
    from core.models import FinancialData
    for index, row in df.iterrows():
        FinancialData.objects.create(
            date=index,
            symbol=symbol,
            price=row['Close'],
            volume=row['Volume']
        )
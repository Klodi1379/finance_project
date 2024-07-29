# data_processing/services/visualization.py

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

def create_stock_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Stock Price', 'Volume'),
                        row_width=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'),
                row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                row=2, col=1)

    fig.update_layout(height=600, width=1000, title_text="Stock Price and Volume Chart")
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig.to_json()

def create_technical_indicators_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=('Price and MA', 'RSI', 'MACD'))

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], name='Moving Average'),
                row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
                row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
                row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'),
                row=3, col=1)

    fig.update_layout(height=900, width=1000, title_text="Technical Indicators")
    
    return fig.to_json()

def create_portfolio_pie_chart(weights, symbols):
    fig = go.Figure(data=[go.Pie(labels=symbols, values=weights)])
    fig.update_layout(title_text="Portfolio Allocation")
    
    return fig.to_json()
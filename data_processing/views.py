from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
import numpy as np
from .services.data_collection import fetch_stock_data, save_to_database
from .services.data_cleaning import remove_outliers, fill_missing_values
from .services.feature_engineering import add_moving_average, add_daily_return
import json
import pandas as pd

@require_http_methods(["GET", "POST"])
def process_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            # Fetch data
            df = fetch_stock_data(symbol, start_date, end_date)

            # Clean data
            df = remove_outliers(df, 'Close')
            df = fill_missing_values(df)

            # Feature engineering
            df = add_moving_average(df)
            df = add_daily_return(df)

            # Save to database
            save_to_database(df, symbol)

            # Prepare data for JSON response
            df_json = df.tail(100).reset_index().to_json(orient='records', date_format='iso')
            data = json.loads(df_json)

            return JsonResponse({'success': True, 'data': data})
        except ValueError as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f"An unexpected error occurred: {str(e)}"}, status=500)
    else:
        return render(request, 'data_processing/process.html')
    
    
    
import json
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

@csrf_exempt
@require_http_methods(["GET", "POST"])
def lstm_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            # Example DataFrame, replace this with actual data fetching and processing
            dates = pd.date_range(start=start_date, end=end_date)
            close_prices = np.random.rand(len(dates)) * 100 + 400
            df = pd.DataFrame({'Date': dates, 'Close': close_prices})

            # Clean data (example, real implementation should use actual data)
            df = df.dropna()

            # Feature engineering
            df['MA'] = df['Close'].rolling(window=5).mean()
            df['Daily_Return'] = df['Close'].pct_change() * 100

            # Prepare data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']].values)

            # Prepare training data
            X_train, y_train = [], []
            for i in range(60, len(scaled_data)):
                X_train.append(scaled_data[i-60:i, 0])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=32)

            # Make predictions
            predictions = model.predict(X_train)
            predictions = scaler.inverse_transform(predictions)
            df['Prediction'] = np.nan
            df.loc[df.index[-len(predictions):], 'Prediction'] = predictions.flatten()

            # Ensure proper date formatting
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df.fillna(0)  # Replace NaNs to avoid JSON serialization issues

            results = df.to_dict(orient='records')
            return JsonResponse({'success': True, 'results': results})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/lstm_analysis.html')

    
    
    
    
    
from .services.arima_model import train_arima_model, predict_arima_model

@require_http_methods(["GET", "POST"])
def arima_analysis(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        symbol = data.get('symbol', '').strip().upper()
        start_date = data.get('start_date', '2022-01-01')
        end_date = data.get('end_date', '2023-01-01')

        df = fetch_stock_data(symbol, start_date, end_date)
        model = train_arima_model(df)
        predictions = predict_arima_model(model, steps=len(df))

        # Prepare data for JSON response
        data = df.tail(len(predictions)).copy()
        data['Predictions'] = predictions.values
        data_json = data.to_json(orient='records', date_format='iso')
        return JsonResponse({'success': True, 'data': json.loads(data_json)})
    else:
        return render(request, 'data_processing/arima_analysis.html')







from .services.sentiment_analysis import analyze_sentiment

@require_http_methods(["GET", "POST"])
def sentiment_analysis(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')

        sentiment = analyze_sentiment(text)
        return JsonResponse({'success': True, 'sentiment': sentiment})
    else:
        return render(request, 'data_processing/sentiment_analysis.html')





from .services.portfolio_optimization import portfolio_optimization

@require_http_methods(["GET", "POST"])
def optimize_portfolio(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        symbols = data.get('symbols', [])
        start_date = data.get('start_date', '2022-01-01')
        end_date = data.get('end_date', '2023-01-01')

        returns = []
        for symbol in symbols:
            df = fetch_stock_data(symbol, start_date, end_date)
            returns.append(df['Close'].pct_change().dropna().values)

        returns = np.array(returns)
        cov_matrix = np.cov(returns)
        mean_returns = np.mean(returns, axis=1)

        optimal_weights = portfolio_optimization(mean_returns, cov_matrix)
        
        return JsonResponse({'success': True, 'weights': optimal_weights.tolist()})
    else:
        return render(request, 'data_processing/optimize_portfolio.html')
    
    
    
    
    
from .services.risk_assessment import calculate_var, calculate_cvar

@require_http_methods(["GET", "POST"])
def risk_assessment(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        symbol = data.get('symbol', '').strip().upper()
        start_date = data.get('start_date', '2022-01-01')
        end_date = data.get('end_date', '2023-01-01')

        df = fetch_stock_data(symbol, start_date, end_date)
        returns = df['Close'].pct_change().dropna().values

        var = calculate_var(returns)
        cvar = calculate_cvar(returns)
        
        return JsonResponse({'success': True, 'var': var, 'cvar': cvar})
    else:
        return render(request, 'data_processing/risk_assessment.html')
    
    
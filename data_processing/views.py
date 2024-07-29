from functools import cache
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
import json
import numpy as np
import pandas as pd
import logging

# Import necessary services
from .services import data_collection, data_cleaning, feature_engineering, lstm_model, arima_model, sentiment_analysis, portfolio_optimization, risk_assessment, visualization
from .services.risk_assessment import calculate_risk_metrics
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if obj is None or obj != obj:  # Check for NaN
            return None
        return super(NpEncoder, self).default(obj)

@require_http_methods(["GET", "POST"])
def process_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            # Fetch data
            df = data_collection.fetch_stock_data(symbol, start_date, end_date)

            # Clean data
            df = data_cleaning.remove_outliers(df, 'Close')
            df = data_cleaning.fill_missing_values(df)

            # Feature engineering
            df = feature_engineering.add_moving_average(df)
            df = feature_engineering.add_daily_return(df)
            df = feature_engineering.add_technical_indicators(df)

            # Save to database
            data_collection.save_to_database(df, symbol)

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

@require_http_methods(["GET", "POST"])
def lstm_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Try to get results from cache
            cache_key = f"lstm_analysis_{symbol}_{start_date}_{end_date}"
            cached_results = cache.get(cache_key)
            if cached_results:
                return JsonResponse({'success': True, 'results': cached_results})

            df = data_collection.fetch_stock_data(symbol, start_date, end_date)
            logger.debug(f"Raw data shape: {df.shape}")
            logger.debug(f"Raw data info:\n{df.info()}")
            logger.debug(f"Raw data head:\n{df.head()}")

            df = feature_engineering.add_technical_indicators(df)  # Add more features
            logger.debug(f"Data with indicators shape: {df.shape}")
            logger.debug(f"Data with indicators info:\n{df.info()}")
            logger.debug(f"Data with indicators head:\n{df.head()}")

            # Check for NaN values
            nan_columns = df.columns[df.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"NaN values found in columns: {nan_columns}")
                df = df.dropna()
                logger.info(f"Rows with NaN values dropped. New shape: {df.shape}")

            if df.empty:
                raise ValueError("All data was removed when dropping NaN values. Please check your data source and preprocessing steps.")

            model, scaler, _ = lstm_model.train_lstm_model(df)
            
            if model is None:
                logger.warning("Not enough data to train LSTM model")
                predictions_df = pd.DataFrame(index=df.index, columns=['Prediction'])
            else:
                predictions_df = lstm_model.predict_lstm_model(model, df, scaler)

            # Combine original data with predictions
            result_df = df.join(predictions_df)
            result_df['Date'] = result_df.index.strftime('%Y-%m-%d')
            
            # Convert DataFrame to dictionary, handling NaN values
            results = json.loads(json.dumps(result_df.reset_index(drop=True).fillna(0).to_dict(orient='records'), cls=NpEncoder))

            # Cache the results
            cache.set(cache_key, results, timeout=3600)  # Cache for 1 hour

            return JsonResponse({'success': True, 'results': results})
        except ValueError as e:
            logger.error(f"ValueError in lstm_analysis: {str(e)}")
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error in lstm_analysis: {str(e)}", exc_info=True)
            return JsonResponse({'success': False, 'error': f"An unexpected error occurred: {str(e)}"}, status=500)
    else:
        return render(request, 'data_processing/lstm_analysis.html')

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
import json
# from . import data_collection, data_cleaning
# from services.arima_model import train_arima_model, predict_arima_model
from .services.arima_model import train_arima_model, predict_arima_model
from django.core.cache import cache
import logging
import numpy as np

@require_http_methods(["GET", "POST"])
def arima_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = pd.to_datetime(data.get('start_date', '2022-01-01'))
            end_date = pd.to_datetime(data.get('end_date', '2023-01-01'))
            
            print(f"Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}")  # Logging
            
            df = data_collection.fetch_stock_data(symbol, start_date, end_date)
            print(f"Stock data shape: {df.shape}")  # Logging
            print(df.head())  # Logging
            print(df.dtypes)  # Logging
            
            df = data_cleaning.remove_outliers(df, 'Close')
            df = data_cleaning.fill_missing_values(df)
            
            if df.empty:
                return JsonResponse({'success': False, 'error': "No valid data after cleaning"})
            
            model = train_arima_model(df)
            predictions = predict_arima_model(model, steps=30)  # Predict next 30 days
            
            print(f"Predictions shape: {predictions.shape}")  # Logging
            print(predictions.head())  # Logging
            
            # Combine actual data and predictions
            combined_data = pd.concat([df['Close'], predictions.rename('Predictions')], axis=1)
            print(f"Combined data shape: {combined_data.shape}")  # Logging
            print(combined_data.head())  # Logging
            
            combined_data = combined_data.reset_index()
            combined_data['Date'] = combined_data['Date'].dt.strftime('%Y-%m-%d')
            
            # Replace NaN with None for JSON serialization
            combined_data = combined_data.where(pd.notnull(combined_data), None)
            
            data_json = combined_data.to_json(orient='records')
            print(f"JSON data length: {len(data_json)}")  # Logging
            return JsonResponse({'success': True, 'data': json.loads(data_json)})
        except ValueError as ve:
            print(f"ValueError: {ve}")  # Logging
            return JsonResponse({'success': False, 'error': str(ve)})
        except Exception as e:
            print(f"Unexpected error: {e}")  # Logging
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/arima_analysis.html')
    
    
    
@require_http_methods(["GET", "POST"])
def sentiment_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '')
            text = data.get('text', '')

            sentiment = sentiment_analysis.analyze_sentiment(text)
            news_sentiment = sentiment_analysis.fetch_news_sentiment(symbol)
            
            return JsonResponse({'success': True, 'text_sentiment': sentiment, 'news_sentiment': news_sentiment})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/sentiment_analysis.html')

@require_http_methods(["GET", "POST"])
def optimize_portfolio(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbols = data.get('symbols', [])
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            returns = []
            for symbol in symbols:
                df = data_collection.fetch_stock_data(symbol, start_date, end_date)
                returns.append(df['Close'].pct_change().dropna().values)

            returns = np.array(returns)
            cov_matrix = np.cov(returns)
            mean_returns = np.mean(returns, axis=1)

            optimal_weights = portfolio_optimization.portfolio_optimization(mean_returns, cov_matrix)
            
            return JsonResponse({'success': True, 'weights': optimal_weights.tolist()})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/optimize_portfolio.html')

@require_http_methods(["GET", "POST"])
def risk_assessment(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')
            risk_free_rate = data.get('risk_free_rate', 0.01)
            
            df = data_collection.fetch_stock_data(symbol, start_date, end_date)
            returns = df['Close'].pct_change().dropna()
            
            if len(returns) < 30:  # Ensure we have enough data points
                return JsonResponse({'success': False, 'error': 'Insufficient data for risk assessment'})
            
            risk_metrics = calculate_risk_metrics(returns, risk_free_rate)
            
            # Convert numpy types to Python native types for JSON serialization
            risk_metrics = {k: float(v) for k, v in risk_metrics.items()}
            
            return JsonResponse({'success': True, 'risk_metrics': risk_metrics})
        except ValueError as ve:
            return JsonResponse({'success': False, 'error': str(ve)})
        except Exception as e:
            return JsonResponse({'success': False, 'error': f"An unexpected error occurred: {str(e)}"})
    else:
        return render(request, 'data_processing/risk_assessment.html')
    
    
    
    
@require_http_methods(["GET", "POST"])
def visualize_stock_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', '').strip().upper()
            start_date = data.get('start_date', '2022-01-01')
            end_date = data.get('end_date', '2023-01-01')

            df = data_collection.fetch_stock_data(symbol, start_date, end_date)
            df = data_cleaning.remove_outliers(df, 'Close')
            df = data_cleaning.fill_missing_values(df)
            df = feature_engineering.add_technical_indicators(df)

            # Log DataFrame columns
            logger.debug(f"DataFrame columns after adding technical indicators: {df.columns}")

            stock_chart = visualization.create_stock_chart(df)
            technical_chart = visualization.create_technical_indicators_chart(df)

            return JsonResponse({
                'success': True, 
                'stock_chart': stock_chart,
                'technical_chart': technical_chart
            })
        except Exception as e:
            logger.error(f"Error in visualize_stock_data: {str(e)}", exc_info=True)
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/visualize_stock_data.html')


@require_http_methods(["GET", "POST"])
def visualize_portfolio(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbols = data.get('symbols', [])
            weights = data.get('weights', [])

            portfolio_chart = visualization.create_portfolio_pie_chart(weights, symbols)

            return JsonResponse({
                'success': True, 
                'portfolio_chart': portfolio_chart
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return render(request, 'data_processing/visualize_portfolio.html')

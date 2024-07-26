# data_processing/urls.py

from django.urls import path
from . import views


app_name = 'data_processing'

urlpatterns = [
    path('process/', views.process_data, name='process_data'),
    path('lstm_analysis/', views.lstm_analysis, name='lstm_analysis'),
    path('arima_analysis/', views.arima_analysis, name='arima_analysis'),
    path('sentiment_analysis/', views.sentiment_analysis, name='sentiment_analysis'),
    path('optimize_portfolio/', views.optimize_portfolio, name='optimize_portfolio'),
    path('risk_assessment/', views.risk_assessment, name='risk_assessment'),
]

from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(df, column='Close', order=(5,1,0)):
    model = ARIMA(df[column], order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima_model(model, steps=10):
    forecast = model.forecast(steps=steps)
    return forecast

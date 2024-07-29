import numpy as np
from scipy import stats
import pandas as pd

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_sharpe_ratio(returns, risk_free_rate=0.01, periods=252):
    excess_returns = returns - risk_free_rate / periods
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.01, periods=252):
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.sum(downside_returns**2) / len(returns)) * np.sqrt(periods)
    return np.sqrt(periods) * excess_returns.mean() / downside_deviation

def calculate_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_risk_metrics(returns, risk_free_rate=0.01, periods=252):
    """
    Calculate various risk metrics for a given set of returns.
    
    :param returns: pandas Series of asset returns
    :param risk_free_rate: annual risk-free rate (default: 1%)
    :param periods: number of periods in a year (default: 252 for daily returns)
    :return: dictionary containing calculated risk metrics
    """
    annualized_return = (1 + returns.mean())**periods - 1
    annualized_volatility = returns.std() * np.sqrt(periods)
    
    risk_metrics = {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'var_95': calculate_var(returns),
        'cvar_95': calculate_cvar(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods),
        'max_drawdown': calculate_max_drawdown(returns),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns)
    }
    
    return risk_metrics
import numpy as np
from scipy.optimize import minimize

def portfolio_optimization(returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(returns)
    
    def portfolio_performance(weights, returns, cov_matrix):
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
    
    def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
        p_return, p_volatility = portfolio_performance(weights, returns, cov_matrix)
        return - (p_return - risk_free_rate) / p_volatility

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(neg_sharpe_ratio, initial_guess, args=(returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

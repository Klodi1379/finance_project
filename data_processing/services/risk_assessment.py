import numpy as np


def calculate_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = abs(sorted_returns[index])
    return var

def calculate_cvar(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = abs(sorted_returns[:index].mean())
    return cvar

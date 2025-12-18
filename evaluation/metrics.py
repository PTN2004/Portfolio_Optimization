import numpy as np
import pandas as pd

def calculate_financial_metrics(portfolio_values, annualization_factor=252, risk_free_rate=0.0):
    portfolio_values = np.array(portfolio_values, dtype=np.float64)
    returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-8)
    log_returns = np.log1p(returns)

    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
    avg_daily_return = np.mean(returns)
    annualized_return = (1 + avg_daily_return) ** annualization_factor - 1.0

    average_value = np.mean(portfolio_values)

    rolling_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (rolling_max - portfolio_values) / (rolling_max + 1e-8)
    max_drawdown = np.max(drawdowns)

    excess_returns = returns - risk_free_rate / annualization_factor
    sharpe_ratio = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)) * np.sqrt(annualization_factor)

    calmar_ratio = (annualized_return + 1e-8) / (max_drawdown + 1e-8)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "average_value": average_value,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio
    }
# backend/services/run_ppo.py

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from preprocessing.reprocess_data import listing_symbols, repare_trading_data
from core.env_new import EnvironmentTrading
from evaluation.metrics import calculate_financial_metrics
from vnstock import Quote

def run_backtest_service(start_date, end_date, init_balance, fee):
    symbols = listing_symbols()
    START_DATE = "2010-01-01"

    X_features, X_prices, mask, index, symbols = repare_trading_data(
        symbols, start_date=START_DATE, end_date=end_date
    )

    start_idx = index.searchsorted(start_date)
    end_idx = index.searchsorted(end_date)

    X_feat_test = X_features[:, start_idx:end_idx, :]
    X_price_test = X_prices[:, start_idx:end_idx]
    mask_test = mask[:, start_idx:end_idx]
    index_test = index[start_idx:end_idx]

    model = PPO.load("./model/best_model.zip")

    env = EnvironmentTrading(
        X_features=X_feat_test,
        X_prices=X_price_test,
        mask=mask_test,
        index=index_test,
        symbols=symbols,
        window_size=30,
        lambda_drawdown=0.1,
        initial_balance=init_balance,
        transaction_cost=fee
    )

    obs, info = env.reset()
    done = False

    values = [init_balance]
    weights = []
    dates = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        values.append(info["portfolio_value"])
        weights.append(info["weights"])
        dates.append(info["date"])

    # Tính metrics
    metrics = calculate_financial_metrics(values)

    return {
        "dates": dates,
        "portfolio_values": values,
        "weights": weights,
        "metrics": metrics
    }

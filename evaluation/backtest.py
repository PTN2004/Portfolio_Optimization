import datetime
import os
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from vnstock import Quote

from core.environment import EnvironmentTrading
from preprocessing.reprocess_data import repare_trading_data, listing_symbols
from .metrics import calculate_financial_metrics
from .trading_evaluate import TradingEvaluator

def run_backtest():
    # symbols = ["VCB", "VHM", "VIC", "FPT", "VNM", "HPG", "POW", "SSI", "TCB", "ACB"]
    group = 'VN30'
    
    symbols = listing_symbols(group)
    START_DATE = "2010-01-01"
    END_DATE = "2025-11-10"
    
    TEST_START = "2024-01-01"
    TEST_END = "2025-11-01"
    
    MODEL_PATH = "./best_model_flexible/best_model_256.zip"

    print(f"\n[1/6] Đang tải và xử lý dữ liệu cho {len(symbols)} mã...")
    X_features, X_prices, mask, index, symbols = repare_trading_data(
        symbols, start_date=START_DATE, end_date=END_DATE
    )

    # Tách tập Test
    start_idx = index.searchsorted(TEST_START)
    end_idx = index.searchsorted(TEST_END)
    
    if start_idx >= end_idx:
        print("LỖI: Khoảng thời gian Test không hợp lệ hoặc không có dữ liệu.")
        return

    X_feat_test = X_features[:, start_idx:end_idx, :]
    X_price_test = X_prices[:, start_idx:end_idx]
    mask_test = mask[:, start_idx:end_idx]
    index_test = index[start_idx:end_idx]
    
    print(f"[INFO] Backtest từ {TEST_START} đến {TEST_END} ({len(index_test)} ngày)")

    # 3. Khởi tạo Môi trường & Mô hình
    print(f"[2/6] Khởi tạo môi trường và tải mô hình...")
    
    test_env = EnvironmentTrading(
        X_features=X_feat_test,
        X_prices=X_price_test,
        mask=mask_test,
        index=index_test,
        symbols=symbols,
        window_size=30,
        lambda_drawdown=0.1,
        initial_balance=1e8, 
        transaction_cost=0.0015 
    )

    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy file mô hình tại {MODEL_PATH}")
        return
    
    model = PPO.load(MODEL_PATH, custom_objects={
        "clip_range": lambda x: x,
        "lr_schedule": lambda x: x
    })

    print(f"[3/6] Đang chạy mô phỏng giao dịch...")
    obs, info = test_env.reset()
    done = False
    
    history = {
        "portfolio_values": [test_env.initial_balance],
        "weights": [],
        "dates": []
    }

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        
        history["portfolio_values"].append(info["portfolio_value"])
        
        if "weights" in info:
            history["weights"].append(info["weights"])
        else:
             history["weights"].append(np.maximum(action, 0) / (np.sum(np.maximum(action, 0)) + 1e-8))
             
        history["dates"].append(info["date"])

    print(f"[4/6] Tính toán Benchmark (VN30 & Equal-Weight)...")
    
    try:
        vn30_df = Quote(symbol=group).history(start=TEST_START, end=TEST_END)
        vn30_df['time'] = pd.to_datetime(vn30_df['time'])
        vn30_df = vn30_df.set_index('time')['close']
        vn30_aligned = vn30_df.reindex(pd.to_datetime(history["dates"]), method='ffill')
        vn30_values = (vn30_aligned / vn30_aligned.iloc[0]) * test_env.initial_balance
    except Exception as e:
        print(f"Cảnh báo: Không tải được VN30 ({e}). Bỏ qua benchmark VN30.")
        vn30_values = None

    dates_idx = pd.to_datetime(history["dates"])
    price_df = pd.DataFrame(X_price_test.T, index=index_test, columns=symbols)
    price_df = price_df.reindex(dates_idx, method='ffill')
    
    price_eq = price_df.copy()
    price_eq[price_eq <= 1e-8] = np.nan
    
    returns_eq = price_eq.pct_change()
    avg_returns = returns_eq.mean(axis=1).fillna(0)
    
    eq_values = (1 + avg_returns).cumprod() * test_env.initial_balance

    print(f"[5/6] Tạo báo cáo đánh giá...")
    
    weights_arr = np.array(history["weights"])
    portfolio_vals_aligned = history["portfolio_values"][1:] 

    if len(weights_arr) > len(history["dates"]):
        weights_arr = weights_arr[-len(history["dates"]):]
    
    # Khởi tạo Evaluator (Truyền VN30 vào làm benchmark chính để vẽ)
    evaluator = TradingEvaluator(
        portfolio_values=portfolio_vals_aligned,
        weights=weights_arr,
        dates=history["dates"],
        benchmark_values=vn30_values.values if vn30_values is not None else None
    )
    
    # In báo cáo Text
    evaluator.print_report()
    
    # So sánh thêm với Equal-Weight (in text phụ)
    eq_metrics = calculate_financial_metrics(eq_values.tolist())
    print(f"\n[So sánh phụ] Equal-Weight Benchmark:")
    print(f" - Total Return: {eq_metrics['total_return']*100:.2f}%")
    print(f" - Sharpe Ratio: {eq_metrics['sharpe_ratio']:.3f}")
    
    # Vẽ biểu đồ
    print(f"[6/6] Vẽ biểu đồ dashboard...")
    evaluator.plot_dashboard(save_path="final_backtest_report.png")
    
    print("\n--- HOÀN TẤT BACKTEST ---")

if __name__ == "__main__":
    run_backtest()
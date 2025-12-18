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

# Giữ nguyên các import từ core/preprocessing của bạn
from core.environment import EnvironmentTrading
from preprocessing.reprocess_data import repare_trading_data, listing_symbols
# from .metrics import calculate_financial_metrics # Tạm comment để dùng hàm mới chuẩn Mercury
from .trading_evaluate import TradingEvaluator

# ==========================================
# 1. HÀM TÍNH TOÁN METRICS CHUẨN MERCURY
# ==========================================
def calculate_mercury_metrics(portfolio_values):
    """
    Tính toán 5 chỉ số cốt lõi của Mercury: ARR, AVL, MDD, SR, CR
    """
    portfolio_values = np.array(portfolio_values)
    
    # Tính lợi nhuận ngày (Daily Returns)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Số ngày giao dịch thực tế
    n_days = len(returns)
    if n_days == 0:
        return {}

    # 1. ARR (Annualized Rate of Return) - Lợi nhuận năm hóa
    # Công thức: (Giá cuối / Giá đầu) ^ (252 / số ngày) - 1
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    arr = (1 + total_return) ** (252 / n_days) - 1

    # 2. AVL (Annualized Volatility) - Biến động năm hóa
    # Công thức: Std(daily_returns) * sqrt(252)
    avl = np.std(returns) * np.sqrt(252)

    # 3. MDD (Maximum Drawdown) - Mức sụt giảm tối đa
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / peak
    mdd = np.max(drawdowns)

    # 4. SR (Sharpe Ratio) - Tỷ lệ Sharpe
    # Giả định risk-free rate = 0 cho đơn giản hóa (như thường lệ trong các bài báo RL)
    sr = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)

    # 5. CR (Calmar Ratio) - Tỷ lệ Calmar
    # Công thức: ARR / MDD
    cr = arr / (mdd + 1e-9)

    return {
        "ARR": arr,
        "AVL": avl,
        "MDD": mdd,
        "SR": sr,
        "CR": cr
    }

def print_mercury_report(metrics, name="Model"):
    """In bảng báo cáo đẹp theo style Mercury"""
    print(f"\n{'='*40}")
    print(f"BÁO CÁO HIỆU SUẤT THEO CHUẨN MERCURY ({name})")
    print(f"{'='*40}")
    print(f"1. Profitability Indicator:")
    print(f"   - ARR (Annualized Return):  {metrics['ARR']*100:.2f}%")
    print(f"-"*40)
    print(f"2. Risk Indicators:")
    print(f"   - AVL (Volatility):         {metrics['AVL']*100:.2f}% (Thấp hơn là tốt)")
    print(f"   - MDD (Max Drawdown):       {metrics['MDD']*100:.2f}% (Thấp hơn là tốt)")
    print(f"-"*40)
    print(f"3. Risk-Profit Indicators:")
    print(f"   - SR (Sharpe Ratio):        {metrics['SR']:.3f}   (Cao hơn là tốt)")
    print(f"   - CR (Calmar Ratio):        {metrics['CR']:.3f}   (Cao hơn là tốt)")
    print(f"{'='*40}\n")

# ==========================================
# 2. HÀM RUN_BACKTEST ĐÃ SỬA ĐỔI
# ==========================================
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

    # ---------------------------------------------------------
    # TÍNH TOÁN VÀ HIỂN THỊ METRICS MERCURY CHO MODEL
    # ---------------------------------------------------------
    print(f"[4/6] Tính toán Metrics chuẩn Mercury...")
    
    # 1. Metrics của AI Model
    model_metrics = calculate_mercury_metrics(history["portfolio_values"])
    print_mercury_report(model_metrics, name="AI Agent (Mercury Style)")

    # ---------------------------------------------------------
    # BENCHMARK (VN30 & Equal-Weight)
    # ---------------------------------------------------------
    
    vn30_values = None
    try:
        vn30_df = Quote(symbol=group).history(start=TEST_START, end=TEST_END)
        vn30_df['time'] = pd.to_datetime(vn30_df['time'])
        vn30_df = vn30_df.set_index('time')['close']
        vn30_aligned = vn30_df.reindex(pd.to_datetime(history["dates"]), method='ffill')
        vn30_values = (vn30_aligned / vn30_aligned.iloc[0]) * test_env.initial_balance
        
        # Tính metrics cho VN30 để so sánh
        vn30_metrics = calculate_mercury_metrics(vn30_values.values)
        print_mercury_report(vn30_metrics, name="Benchmark VN30")
        
    except Exception as e:
        print(f"Cảnh báo: Không tải được VN30 ({e}). Bỏ qua benchmark VN30.")

    # Tính Equal-Weight Benchmark
    dates_idx = pd.to_datetime(history["dates"])
    price_df = pd.DataFrame(X_price_test.T, index=index_test, columns=symbols)
    price_df = price_df.reindex(dates_idx, method='ffill')
    price_eq = price_df.copy()
    price_eq[price_eq <= 1e-8] = np.nan
    returns_eq = price_eq.pct_change()
    avg_returns = returns_eq.mean(axis=1).fillna(0)
    eq_values = (1 + avg_returns).cumprod() * test_env.initial_balance
    
    # Tính metrics cho Equal-Weight
    eq_metrics = calculate_mercury_metrics(eq_values.values)
    print_mercury_report(eq_metrics, name="Benchmark Equal-Weight")

    print(f"[5/6] Tạo báo cáo Visual...")
    
    weights_arr = np.array(history["weights"])
    portfolio_vals_aligned = history["portfolio_values"][1:] 

    if len(weights_arr) > len(history["dates"]):
        weights_arr = weights_arr[-len(history["dates"]):]
    
    # Sử dụng lại TradingEvaluator của bạn để vẽ biểu đồ (vì nó tiện cho visual)
    evaluator = TradingEvaluator(
        portfolio_values=portfolio_vals_aligned,
        weights=weights_arr,
        dates=history["dates"],
        benchmark_values=vn30_values.values if vn30_values is not None else None
    )
    
    # Vẽ biểu đồ
    print(f"[6/6] Vẽ biểu đồ dashboard...")
    evaluator.plot_dashboard(save_path="final_backtest_report_mercury.png")
    
    print("\n--- HOÀN TẤT BACKTEST ---")

if __name__ == "__main__":
    run_backtest()
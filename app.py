import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from stable_baselines3 import PPO
from vnstock import Quote

# --- IMPORT MODULES CỦA BẠN ---
from core.new_fe import FeatureExtractor, HybridAttention
from core.environment import EnvironmentTrading
from preprocessing.reprocess_data import repare_trading_data

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Trading Pro Dashboard", layout="wide", page_icon="📊")

# CSS Tùy chỉnh
st.markdown("""
<style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4e73df; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .buy-signal { color: #28a745; font-weight: bold; }
    .sell-signal { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Portfolio Manager (Full Features)")
st.caption("Hệ thống tối ưu hóa danh mục, quản lý rủi ro và khuyến nghị giao dịch.")

# --- SIDEBAR: CẤU HÌNH ---
st.sidebar.header("🛠 Cấu hình Danh mục")
user_input = st.sidebar.text_area("Mã cổ phiếu:", value="FPT, MWG, HPG, VCB, SSI, TCB")
selected_symbols = [s.strip().upper() for s in user_input.split(",") if s.strip()]

start_date = st.sidebar.date_input("Ngày bắt đầu:", pd.to_datetime("2024-06-01"))
end_date = st.sidebar.date_input("Ngày kết thúc:", pd.to_datetime("2024-11-01"))
initial_capital = st.sidebar.number_input("Vốn đầu tư (VND):", value=100_000_000, step=10_000_000)
btn_run = st.sidebar.button("🚀 Kích Hoạt AI", type="primary")

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_ai_model():
    model_path = "./best_model_flexible/best_model_256.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        custom_objects = {"lr_schedule": lambda x: x, "clip_range": lambda x: x}
        model = PPO.load(model_path, custom_objects=custom_objects, device=device)
        return model
    except Exception as e:
        st.error(f"Không tìm thấy model tại {model_path}. Lỗi: {e}")
        return None

# --- MAIN LOGIC ---
if btn_run:
    if len(selected_symbols) < 2:
        st.error("Vui lòng nhập ít nhất 2 mã cổ phiếu.")
        st.stop()

    with st.spinner('🤖 AI đang phân tích dữ liệu thị trường và chạy mô phỏng...'):
        # 1. TẢI DỮ LIỆU
        try:
            fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            fetch_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            
            X_features, X_prices, mask, index, symbols = repare_trading_data(
                selected_symbols, start_date=fetch_start, end_date=fetch_end
            )
            
            sim_start_idx = index.searchsorted(str(start_date))
            if sim_start_idx >= len(index):
                st.error("Khoảng thời gian này không có dữ liệu giao dịch.")
                st.stop()

            X_feat_sim = X_features[:, sim_start_idx:, :]
            X_price_sim = X_prices[:, sim_start_idx:]
            mask_sim = mask[:, sim_start_idx:]
            index_sim = index[sim_start_idx:]
            
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu đầu vào: {e}")
            st.stop()

        # 2. KHỞI TẠO ENV & MODEL
        env = EnvironmentTrading(
            X_features=X_feat_sim, X_prices=X_price_sim, mask=mask_sim, index=index_sim,
            symbols=symbols, max_num_assets=100, window_size=30,
            initial_balance=initial_capital, transaction_cost=0.0015
        )
        
        model = load_ai_model()
        if not model: st.stop()

        # 3. CHẠY MÔ PHỎNG (BACKTEST LOOP)
        obs, info = env.reset()
        done = False
        
        trade_logs = []       
        portfolio_history = [] 
        history_weights = []   
        prev_weights = np.zeros(len(symbols)) 
        
        progress_bar = st.progress(0)
        total_steps = len(index_sim)
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            current_date = info["date"]
            nav = info["portfolio_value"]
            
            # --- XỬ LÝ WEIGHTS & CASH ---
            # Lấy raw weights từ model cho các mã user chọn
            raw_weights = action[:len(symbols)]
            
            # Xử lý logic Tiền mặt (CASH)
            # Nếu tổng trọng số các mã < 1.0, phần còn lại là Tiền mặt
            total_stock_weight = np.sum(np.maximum(raw_weights, 0))
            if total_stock_weight > 1.0:
                user_weights = np.maximum(raw_weights, 0) / total_stock_weight # Normalize nếu lố 100%
                cash_weight = 0.0
            else:
                user_weights = np.maximum(raw_weights, 0)
                cash_weight = 1.0 - total_stock_weight
            
            # Lưu lại để vẽ biểu đồ (Thêm cột CASH vào cuối)
            current_alloc = list(user_weights) + [cash_weight]
            history_weights.append(current_alloc)
            
            # So sánh tín hiệu (chỉ so phần cổ phiếu)
            diff = user_weights - prev_weights
            prices_today = X_price_sim[:, step] 
            
            daily_actions = []
            for i, sym in enumerate(symbols):
                change_pct = diff[i]
                if abs(change_pct) > 0.01: 
                    action_type = "MUA" if change_pct > 0 else "BÁN"
                    money_est = abs(change_pct) * nav
                    daily_actions.append({
                        "Mã": sym, "Hành động": action_type,
                        "Tỷ trọng đổi": f"{abs(change_pct)*100:.1f}%",
                        "Giá": prices_today[i], "Giá trị": money_est
                    })
            
            trade_logs.append({
                "Ngày": current_date, "NAV": nav, "Chi tiết lệnh": daily_actions,
                "Weights": user_weights, "Prices": prices_today, "Cash": cash_weight
            })
            
            prev_weights = user_weights
            portfolio_history.append({"Date": current_date, "NAV": nav})
            
            step += 1
            progress_bar.progress(min(step / total_steps, 1.0))

        # --- XỬ LÝ KẾT QUẢ ---
        df_result = pd.DataFrame(portfolio_history)
        df_result["Date"] = pd.to_datetime(df_result["Date"])
        df_result.set_index("Date", inplace=True)
        
        df_result["Daily_Return"] = df_result["NAV"].pct_change().fillna(0)
        df_result["Cum_Return"] = (df_result["NAV"] / initial_capital) - 1
        df_result["Drawdown"] = (df_result["NAV"] / df_result["NAV"].cummax()) - 1

        # Lấy Benchmark
        has_benchmark = False
        try:
            vnindex = Quote("VNINDEX").history(pd.to_datetime(start_date).strftime("%Y-%m-%d"), pd.to_datetime(end_date).strftime("%Y-%m-%d"),)
            vnindex["time"] = pd.to_datetime(vnindex["time"])
            vnindex.set_index("time", inplace=True)
            vnindex = vnindex.reindex(df_result.index, method='ffill')
            vnindex["Cum_Return"] = (vnindex["close"] / vnindex["close"].iloc[0]) - 1
            df_result["Benchmark"] = vnindex["Cum_Return"]
            has_benchmark = True
        except: pass

        # --- 4. DASHBOARD HIỂN THỊ ---
        
        # A. Metrics
        final_nav = portfolio_history[-1]["NAV"]
        profit = final_nav - initial_capital
        max_dd = df_result["Drawdown"].min()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><b>Vốn ban đầu</b><br>{initial_capital:,.0f} ₫</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><b>Tài sản hiện tại</b><br>{final_nav:,.0f} ₫</div>", unsafe_allow_html=True)
        
        color = "green" if profit > 0 else "red"
        c3.markdown(f"<div class='metric-card' style='border-left: 5px solid {color}'><b>Lợi nhuận</b><br>{profit:,.0f} ₫ ({profit/initial_capital*100:.2f}%)</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card' style='border-left: 5px solid red'><b>Max Drawdown</b><br>{max_dd*100:.2f}%</div>", unsafe_allow_html=True)

        st.divider()

        # B. Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Hiệu Suất", "🌊 Rủi Ro", "💰 Phân Bổ & Tiền Mặt", "🔥 Tương Quan", "🕯 Soi Chart"
        ])

        # TAB 1: Alpha
        with tab1:
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(x=df_result.index, y=df_result["Cum_Return"]*100, name="AI Portfolio", line=dict(color="#00CC96", width=2)))
            if has_benchmark:
                fig_alpha.add_trace(go.Scatter(x=df_result.index, y=df_result["Benchmark"]*100, name="VN-INDEX", line=dict(color="#636EFA", dash='dot')))
            fig_alpha.update_layout(title="Lợi nhuận Lũy kế (%)", yaxis_title="%", hovermode="x unified", height=450)
            st.plotly_chart(fig_alpha, use_container_width=True)

        # TAB 2: Drawdown
        with tab2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df_result.index, y=df_result["Drawdown"]*100, fill='tozeroy', line=dict(color='#EF553B'), name='Drawdown'))
            fig_dd.update_layout(title="Mức độ sụt giảm tài sản (Drawdown)", yaxis_title="%", height=450)
            st.plotly_chart(fig_dd, use_container_width=True)

        # TAB 3: Allocation (+ CASH)
        with tab3:
            if len(history_weights) > 0:
                # Tạo cột tên: Các mã + Tiền mặt
                cols = symbols + ["TIỀN MẶT (CASH)"]
                w_df = pd.DataFrame(history_weights, columns=cols)
                
                if len(portfolio_history) > 0:
                    w_df["Date"] = pd.to_datetime([p["Date"] for p in portfolio_history])
                    w_df.set_index("Date", inplace=True)
                    
                    # Vẽ biểu đồ miền
                    fig_alloc = px.area(w_df, x=w_df.index, y=cols, title="Diễn biến Tỷ trọng Danh mục (bao gồm Tiền mặt)")
                    # Tô màu Tiền mặt thành màu xám nhạt để dễ phân biệt
                    # (Plotly tự động chọn màu, nhưng ta có thể custom nếu muốn)
                    fig_alloc.update_layout(hovermode="x unified", height=450)
                    st.plotly_chart(fig_alloc, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu phân bổ.")

        # TAB 4: Correlation Matrix (TƯƠNG QUAN)
        with tab4:
            st.subheader("Độ Tương Quan Giữa Các Mã (Correlation Matrix)")
            st.caption("Màu càng sáng (Vàng) -> Tương quan càng cao. Nếu danh mục toàn màu vàng -> Rủi ro cao vì 'chết chùm'.")
            
            # Tạo DataFrame giá để tính corr
            df_prices_corr = pd.DataFrame(X_price_sim.T, columns=symbols)
            # Tính phần trăm thay đổi hàng ngày (Log return hoặc PCT change)
            df_returns_corr = df_prices_corr.pct_change().dropna()
            
            # Tính ma trận tương quan
            corr_matrix = df_returns_corr.corr()
            
            # Vẽ Heatmap
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale="RdBu_r", # Đỏ (Nghịch biến) - Xanh (Đồng biến)
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        # TAB 5: Pro Chart
        with tab5:
            c_sel, _ = st.columns([1, 3])
            stock_view = c_sel.selectbox("Chọn mã để soi chart:", symbols)
            
            try:
                df_real = Quote(symbol=stock_view).history(
                    start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                    end=pd.to_datetime(end_date).strftime("%Y-%m-%d")
                )
                df_real['time'] = pd.to_datetime(df_real['time'])
                df_real.set_index('time', inplace=True)

                fig_pro = go.Figure()
                fig_pro.add_trace(go.Candlestick(
                    x=df_real.index, open=df_real['open'], high=df_real['high'],
                    low=df_real['low'], close=df_real['close'], name='Giá'
                ))
                
                # Vẽ tín hiệu Mua/Bán
                buy_x, buy_y, sell_x, sell_y = [], [], [], []
                for log in trade_logs:
                    d = pd.to_datetime(log["Ngày"])
                    if d in df_real.index:
                        for t in log["Chi tiết lệnh"]:
                            if t["Mã"] == stock_view:
                                p = df_real.loc[d]['close']
                                if t["Hành động"] == "MUA": 
                                    buy_x.append(d); buy_y.append(p*0.98)
                                elif t["Hành động"] == "BÁN": 
                                    sell_x.append(d); sell_y.append(p*1.02)
                
                if buy_x: fig_pro.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'), name='AI MUA'))
                if sell_x: fig_pro.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='AI BÁN'))

                fig_pro.update_layout(title=f"Chart: {stock_view}", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_pro, use_container_width=True)
            except Exception as e:
                st.warning(f"Không vẽ được chart chi tiết: {e}")

        st.divider()

        # C. Nhật Ký Giao Dịch
        with st.expander("📝 Xem Nhật Ký Tín Hiệu (Signal Log)", expanded=True):
            flat_logs = []
            for log in trade_logs:
                # Hiển thị thêm cột Tiền mặt trong bảng log nếu cần
                cash_info = f"{log['Cash']*100:.1f}%"
                if log["Chi tiết lệnh"]:
                    for t in log["Chi tiết lệnh"]:
                        flat_logs.append({
                            "Ngày": log["Ngày"], "Mã": t["Mã"], "Lệnh": t["Hành động"],
                            "% Đổi": t["Tỷ trọng đổi"], "Giá": f"{t['Giá']:,.0f}", "Cash nắm giữ": cash_info
                        })
            
            if flat_logs:
                df_sig = pd.DataFrame(flat_logs)
                def color_sig(val):
                    return 'background-color: #d4edda' if val == "MUA" else 'background-color: #f8d7da'
                st.dataframe(df_sig.style.applymap(color_sig, subset=['Lệnh']), use_container_width=True)
            else:
                st.info("AI Quyết định Nắm giữ (Hold) toàn bộ thời gian này.")

else:
    st.info("👈 Hãy nhập danh mục bên trái và nhấn nút 'Kích Hoạt AI'.")